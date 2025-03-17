# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics
from tqdm import tqdm

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net
from .diffusion import Diffusion, disabled_train, create_model_config
from i2sb.VQGAN.vqgan import VQModel
from i2sb.base.modules.encoders.modules import SpatialRescaler
from torch.utils.data import DataLoader
from corruption.mixture import floodDataset
from .embedding import RainfallEmbedder

from ipdb import set_trace as debug

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        self.opt = opt
        self.model_config = create_model_config()
        if opt.latent_space:
            self.vqgan = VQModel(**vars(self.model_config.VQGAN.params)).eval()
            self.vqgan.train = disabled_train
            for param in self.vqgan.parameters():
                param.requires_grad = False
            print(f"load vqgan from {self.model_config.VQGAN.params.ckpt_path}")
            
            self.cond_stage_model = SpatialRescaler(**vars(self.model_config.CondStageParams))
            self.vqgan.to(opt.device)
            self.cond_stage_model.to(opt.device)

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        self.rainfall_emb = RainfallEmbedder(256, 1)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")
            self.rainfall_emb.load_state_dict(checkpoint['embedding'])
            log.info(f"[Embedding] Loaded embedding ckpt: {opt.load}!")
            if opt.normalize_latent:
                self.net.ori_latent_mean = checkpoint["ori_latent_mean"]
                self.net.ori_latent_std = checkpoint["ori_latent_std"]
                self.net.cond_latent_mean = checkpoint["cond_latent_mean"]
                self.net.cond_latent_std = checkpoint["cond_latent_std"]
                log.info(f"[Latent] Loaded latent mean/std ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)
        self.rainfall_emb.to(opt.device)

        self.log = log

        if opt.eval:
            self.net.ori_latent_mean = self.net.ori_latent_mean.to(opt.device)
            self.net.ori_latent_std = self.net.ori_latent_std.to(opt.device)
            self.net.cond_latent_mean = self.net.cond_latent_mean.to(opt.device)
            self.net.cond_latent_std = self.net.cond_latent_std.to(opt.device)

    def logger(self, msg, **kwargs):
        print(msg, **kwargs)

    def get_latent_mean_std(self):
        train_dataset = floodDataset(True)
        train_loader = DataLoader(train_dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        # max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_cond, _) = batch
            x = x.to(self.opt.device)
            x_cond = x_cond.to(self.opt.device)

            x_latent = self.vqgan.encoder(x)
            x_cond_latent = self.vqgan.encoder(x_cond)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_cond, _) = batch
            x = x.to(self.opt.device)
            x_cond = x_cond.to(self.opt.device)

            x_latent = self.vqgan.encoder(x)
            x_cond_latent = self.vqgan.encoder(x_cond)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        self.logger(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        self.logger(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        # self.logger(self.net.ori_latent_mean)
        # self.logger(self.net.ori_latent_std)
        # self.logger(self.net.cond_latent_mean)
        # self.logger(self.net.cond_latent_std)
    
    @torch.no_grad()
    def encode(self, x, cond=True):
        normalize = self.opt.normalize_latent 
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.net.cond_latent_mean) / self.net.cond_latent_std
            else:
                x_latent = (x_latent - self.net.ori_latent_mean) / self.net.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True):
        normalize = self.opt.normalize_latent
        if normalize:
            if cond:
                x_latent = x_latent * self.net.cond_latent_std + self.net.cond_latent_mean
            else:
                x_latent = x_latent * self.net.ori_latent_std + self.net.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out
    
    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        if opt.corrupt == "mixture":
            clean_img, corrupt_img, y, image_name = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
        # debug()
        # y is rainfall value in this case (24)
        y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        if opt.latent_space:
            x0 = self.encode(x0, cond=False)
            x1 = self.encode(x1, cond=False)
            cond = self.cond_stage_model(cond)

        return x0, x1, mask, y, cond

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = self.net
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        self.accuracy = torchmetrics.Accuracy().to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                rainfall_emb = self.rainfall_emb(y)
                pred = net(xt, step, rainfall_emb, cond=cond)
                assert xt.shape == label.shape == pred.shape

                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 2000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        'embedding': self.rainfall_emb.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            # if it == 500 or it % 3000 == 0: # 0, 0.5k, 3k, 6k 9k
            #     net.eval()
            #     self.evaluation(opt, it, val_loader, corrupt_method)
            #     net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, y, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True, eval=False):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        if opt.latent_space and eval:
            x1 = self.encode(x1, cond=False)
            cond = self.cond_stage_model(cond)
            
        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step, rainfall_emb):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, rainfall_emb, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            rainfall_emb = self.rainfall_emb(y)
            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, rainfall_emb, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        if opt.latent_space and eval:
            xs = xs[:, 0, ...].to(opt.device)
            # xs = xs.squeeze(1)
            xs = self.decode(xs, cond=False)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)

        x1 = img_corrupt.to(opt.device)
        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, y, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        # assert y.shape == (batch,)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        # def log_accuracy(tag, img):
        #     pred = self.resnet(img.to(opt.device)) # input range [-1,1]
        #     accu = self.accuracy(pred, img_clean.to(opt.device))
        #     self.writer.add_scalar(it, tag, accu)

        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

        # log.info("Logging accuracies ...")
        # log_accuracy("accuracy/clean",   img_clean)
        # log_accuracy("accuracy/corrupt", img_corrupt)
        # log_accuracy("accuracy/recon",   img_recon)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
