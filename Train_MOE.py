import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from .data.dataset import get_dataset
from loss.loss_function import get_loss_function
import utils as utils

import sys
from datetime import datetime
import json

from model.MOE_model import MOEModel
from tqdm import tqdm
import argparse


def diversity_loss(weights):
    num_experts = weights.shape[-1]
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
    normalized_entropy = entropy / torch.log(torch.tensor(num_experts, dtype=torch.float, device=weights.device))
    loss = 1 - normalized_entropy
    return torch.mean(loss)


def train(cfg):
    # -------- initial setup --------
    now = datetime.now().strftime("%Y%m%d_%H%M%S")  # ex: 20250907_0000
    chkpt_dir = os.path.join(cfg["output_dir"], now)  # ex: ./chkpt/20250907_0000
    os.makedirs(chkpt_dir, exist_ok=True)
    utils.json_dump(os.path.join(chkpt_dir, "config.json"), cfg)
    log_txt_path = os.path.join(chkpt_dir, "log.txt")
    sys.stdout = utils.Logger(log_txt_path)
    device = torch.device(cfg["device"])
    # -------------------------------

    # -------- dataset / dataloader --------
    train_ds = get_dataset(
        cfg["train_list"],
        os.path.join(cfg["wav_dir"], "train"),
        max_sec=cfg["max_len"],
        sr=cfg["audio_encoder"]["sample_rate"],
        org_max=10.0,
        org_min=0.0,
        is_train=True,
    )
    val_ds = get_dataset(
        cfg["validation_list"],
        os.path.join(cfg["wav_dir"], "validation"),
        max_sec=cfg["max_len"],
        sr=cfg["audio_encoder"]["sample_rate"],
        org_max=10.0,
        org_min=0.0,
        is_train=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=train_ds.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["val_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=val_ds.collate_fn,
    )
    # -------------------------------------------------

    # -------- model / loss / opt --------
    model = MOEModel(cfg, device).to(device)

    gating_parameters = list(model.gating_network.parameters())

    reg_loss_fn = get_loss_function(cfg["loss"])
    # opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    opt = torch.optim.AdamW(gating_parameters, lr=cfg["lr"], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    best_srcc, patience = -np.inf, 0
    start_epoch = 0

    if cfg.get("resume_from_checkpoint"):
        ckpt_path = cfg["resume_from_checkpoint"]
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        opt.load_state_dict(checkpoint["opt_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_srcc = checkpoint.get("best_srcc", -np.inf)
        patience = checkpoint.get("patience", 0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print("Resume from checkpoint:", ckpt_path)
        print(
            f"best_srcc: {best_srcc:.4f}, patience: {patience}, start_epoch: {start_epoch}"
        )

    # ------------------------------------
    print("train loader:", len(train_loader))
    for epoch in range(start_epoch, cfg["epochs"] + start_epoch):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs'] + start_epoch}"
        )

        train_preds, train_gts = [], []
        for batch in train_pbar:
            # ---------- setup ----------
            batch = utils.move_to_device(batch, device)
            opt.zero_grad()
            # ---------------------------

            pred, weight = model.forward(batch)

            # ---------- loss ----------
            reg_loss = reg_loss_fn(pred, batch["scores"], batch["num_class"])

            total_loss = reg_loss + diversity_loss(weight) * 0.2

            total_loss.backward()
            opt.step()
            epoch_loss += total_loss.item()

            train_pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "avg_loss": f"{epoch_loss / (train_pbar.n + 1):.4f}",
                }
            )

            train_preds.extend((pred.detach().cpu().numpy() * 5 + 5).tolist())
            train_gts.extend(batch["real_scores"].cpu().numpy().tolist())

        # ---------- Train Epoch loss ----------
        avg_train_loss = epoch_loss / len(train_loader)

        train_srcc = spearmanr(train_gts, train_preds).correlation
        train_mse = np.mean((np.array(train_gts) - np.array(train_preds)) ** 2)
        # --------------------------------------

        ################################################################
        train_pred = (pred.detach().cpu().numpy() * 5 + 5).tolist()
        # train_pred = pred
        train_gt = (batch["scores"].detach().cpu().numpy() * 5 + 5).tolist()

        print(f"weight: {weight}")

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = utils.move_to_device(batch, device)
                # pred  = model.forward(batch, normalizer_eval)
                pred, _ = model.forward(batch)
                loss = reg_loss_fn(pred, batch["scores"], batch["num_class"])
                val_loss += loss.item()
                preds.extend((pred.cpu().numpy() * 5 + 5).tolist())
                gts.extend((batch["scores"].cpu().numpy() * 5 + 5).tolist())
        avg_val_loss = val_loss / len(val_loader)
        srcc = spearmanr(gts, preds).correlation
        mse = np.mean((np.array(gts) - np.array(preds)) ** 2)
        # --------------------------------

        ################################################################
        val_pred = (pred.detach().cpu().numpy() * 5 + 5).tolist()
        val_gt = (batch["scores"].detach().cpu().numpy() * 5 + 5).tolist()
        ################################################################

        # ---------- timer ----------
        end_time = time.time()
        elapsed_time = end_time - start_time
        # ---------------------------

        # ---------- epoch summary ----------
        print(
            f"Epoch {epoch:04d} completed in {elapsed_time:.2f} seconds | Train Loss : {avg_train_loss:.4f}\tVal Loss : {avg_val_loss:.4f}\tVal SRCC / MSE: {srcc:.4f} , {mse:.4f}"
        )
        # -----------------------------------

        ########################################################################
        train_pred = [f"{v: 05.2f}" for v in train_pred]
        train_gt = [f"{v: 05.2f}" for v in train_gt]
        val_pred = [f"{v: 05.2f}" for v in val_pred]
        val_gt = [f"{v: 05.2f}" for v in val_gt]
        print(f"\ttrain pred   : {train_pred}")
        print(f"\ttrain_gt     : {train_gt}")
        print(f"\tval pred     : {val_pred}")
        print(f"\tval_gt       : {val_gt}")
        ########################################################################

        # ---------- Scheduler ----------
        scheduler.step(avg_val_loss)
        # -------------------------------

        # ---------- early-stop ----------
        if srcc > best_srcc:
            best_srcc, patience = srcc, 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_srcc": best_srcc,
                "patience": patience,
            }
            torch.save(checkpoint, os.path.join(chkpt_dir, "best_model.pt"))
            print("✅  best model updated")
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print("⛔️ Early stopping (patience exhausted)")
                break
        # --------------------------------

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = utils.load_config(args.config)

    if args.resume is not None:
        config["resume_from_checkpoint"] = args.resume
    if args.device is not None:
        config["device"] = args.device

    train(config)
