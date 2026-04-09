"""Training engine — optimized for Ada Lovelace (BF16)."""

import os
import csv
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from losses import CenterNet3DLoss
from decode import decode_predictions
from metrics import compute_all_metrics


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.criterion = CenterNet3DLoss(cfg).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr,
            weight_decay=cfg.weight_decay)

        def lr_lambda(epoch):
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / cfg.warmup_epochs
            factor = 1.0
            for ms in cfg.lr_milestones:
                if epoch >= ms:
                    factor *= cfg.lr_gamma
            return factor

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda)

        # ---- Detect BF16 support (Ada Lovelace, Ampere+) ----
        self.use_bf16 = (cfg.use_amp
                         and torch.cuda.is_available()
                         and torch.cuda.is_bf16_supported())
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16

        # GradScaler not needed for BF16 (no loss scaling required)
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=(cfg.use_amp and not self.use_bf16))

        if self.use_bf16:
            print(f"  ✓ Using BF16 (Ada Lovelace detected)")
        elif cfg.use_amp:
            print(f"  Using FP16 with GradScaler")

        # Also enable TF32 for matmuls (free speedup on Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.train_loader = train_loader
        self.val_loader   = val_loader

        os.makedirs(cfg.output_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(cfg.output_dir, "tb_logs"))

        self.csv_path = os.path.join(cfg.output_dir, "train_log.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "total", "hm", "off", "ctr", "edg", "cor",
                "val_ap25", "val_ap50", "val_corner_cm", "val_center_cm",
                "val_n_det", "val_n_matched", "lr",
            ])

        self.global_step = 0
        self.best_metric = -1.0

    # ==================================================================
    def train_epoch(self, epoch):
        self.model.train()
        accum, n_batch, n_nan = {}, 0, 0
        t0 = time.time()

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}", leave=False)
        for imgs, targets, metas in pbar:
            imgs = imgs.to(self.device)
            tgt  = {k: v.to(self.device) for k, v in targets.items()}

            with torch.amp.autocast('cuda', enabled=self.cfg.use_amp,
                                    dtype=self.amp_dtype):
                out = self.model(imgs)
                loss, stats = self.criterion(out, tgt)

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad(set_to_none=True)
                n_nan += 1
                pbar.set_postfix(L="NaN!")
                continue

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in stats.items():
                if not torch.isnan(v):
                    accum[k] = accum.get(k, 0.0) + v.item()
            n_batch += 1
            self.global_step += 1

            if self.global_step % self.cfg.log_interval == 0:
                for k, v in stats.items():
                    if not torch.isnan(v):
                        self.writer.add_scalar(
                            f"train/{k}", v.item(), self.global_step)

            pbar.set_postfix(L=f"{stats['total'].item():.3f}")

        avg = {k: v / max(n_batch, 1) for k, v in accum.items()}
        dt = time.time() - t0
        nan_str = f" nan_batches={n_nan}" if n_nan else ""
        print(f"  [Train] total={avg.get('total',0):.4f} "
              f"hm={avg.get('hm',0):.4f} off={avg.get('off',0):.4f} "
              f"ctr={avg.get('ctr',0):.4f} edg={avg.get('edg',0):.4f} "
              f"cor={avg.get('cor',0):.4f} | {dt:.0f}s{nan_str}")
        return avg

    # ==================================================================
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        all_dets, all_gts = [], []

        for imgs, targets, metas in tqdm(
            self.val_loader, desc="Val", leave=False
        ):
            imgs = imgs.to(self.device)
            pc   = metas["pc_raw"].to(self.device)

            with torch.amp.autocast('cuda', enabled=self.cfg.use_amp,
                                    dtype=self.amp_dtype):
                out = self.model(imgs)
            batch_dets = decode_predictions(out, pc, self.cfg)

            for b in range(imgs.shape[0]):
                all_dets.append(batch_dets[b])
                sid = metas["sample_id"][b]
                bbox3d = np.load(
                    os.path.join(self.cfg.data_root, sid, "bbox3d.npy")
                ).astype(np.float32)
                all_gts.append(bbox3d)

        metrics = compute_all_metrics(all_dets, all_gts, verbose=True)
        return metrics

    # ==================================================================
    def fit(self):
        n_p = sum(p.numel() for p in self.model.parameters())
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        print(f"Training on {gpu_name} ({gpu_mem:.0f}GB) | "
              f"{n_p/1e6:.2f}M params | "
              f"{'BF16' if self.use_bf16 else 'FP16' if self.cfg.use_amp else 'FP32'}")

        for epoch in range(1, self.cfg.max_epochs + 1):
            print(f"\n{'='*65}")
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{self.cfg.max_epochs}  lr={lr:.2e}")

            loss_avg = self.train_epoch(epoch)
            self.scheduler.step()

            vm = {}
            do_eval = (epoch % self.cfg.save_interval == 0
                       or epoch == self.cfg.max_epochs
                       or epoch <= 3)

            if do_eval:
                vm = self.validate(epoch)
                print("  [Val] " + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) and not np.isnan(v)
                    else f"{k}={v}"
                    for k, v in vm.items()
                ))
                for k, v in vm.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        self.writer.add_scalar(f"val/{k}", v, epoch)

                ap = vm.get("AP@0.25", 0)
                corner_err = vm.get("corner_err_cm", float("inf"))
                if ap > 0:
                    metric = ap
                elif not np.isnan(corner_err) and corner_err < float("inf"):
                    metric = 1.0 / (1.0 + corner_err)
                else:
                    metric = 0.0

                if metric > self.best_metric:
                    self.best_metric = metric
                    self._save("best.pth", epoch)
                    print(f"  ★ New best (metric={metric:.4f})")

            self._save("latest.pth", epoch)

            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch,
                    *[f"{loss_avg.get(k, 0):.5f}"
                      for k in ("total", "hm", "off", "ctr", "edg", "cor")],
                    f"{vm.get('AP@0.25', 0):.4f}",
                    f"{vm.get('AP@0.50', 0):.4f}",
                    f"{vm.get('corner_err_cm', 0):.2f}",
                    f"{vm.get('center_err_cm', 0):.2f}",
                    vm.get("n_det_total", 0),
                    vm.get("n_matched", 0),
                    f"{lr:.2e}",
                ])

        self.writer.close()
        print(f"\n✓ Done. Best metric = {self.best_metric:.4f}")

    def _save(self, name, epoch):
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "sched": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "cfg": self.cfg,
            "xyz_mean": self.train_loader.dataset.xyz_mean,
            "xyz_std":  self.train_loader.dataset.xyz_std,
        }, os.path.join(self.cfg.output_dir, name))