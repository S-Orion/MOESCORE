import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.MOE_model import MOEModel

from data.dataset import get_infdataset

import utils as utils
from tqdm import tqdm
import csv
import sys
import os
from ruamel import yaml
from easydict import EasyDict
import argparse


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def inference(
    cfg, split="test", ckpt_path=None, output_path=None, device_override=None
):
    dataset_label = f"{split}_list"
    dataset_list = cfg[dataset_label]
    dataset_wav_dir = cfg["wav_dir"]

    if device_override is not None:
        cfg["device"] = device_override
    device = torch.device(cfg["device"])

    if ckpt_path is None:
        ckpt_path = cfg.moe_ckpt_path

    print("Perform inference with:")
    print(f"\tcheckpoint:   {ckpt_path}")
    print(f"\tdata list:    {dataset_list}")
    print(f"\twav dir:      {dataset_wav_dir}")
    print(f"\tdevice:       {device}")

    test_ds = get_infdataset(
        txt_file_path=dataset_list,
        wav_dir=dataset_wav_dir,
        max_sec=cfg["max_len"],
        sr=cfg["sample_rate"],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=test_ds.collate_fn,
    )
    # -------------------------------------------------

    model = MOEModel(cfg, device).to(device)

    chkpt = torch.load(cfg.moe_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(chkpt.get("model_state_dict"), strict=False)
    model.eval()
    # ------------------------------------
    print(f"\tbest SRCC in the checkpoint: ", chkpt.get("best_srcc", "N/A"))
    # -------- run inference --------
    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = utils.move_to_device(batch, device)
            pred, _ = model.forward(batch)
            pred = pred.detach().cpu().item()
            pred_mos = pred * 5.0 + 5.0
            rows.append(
                {
                    "wav_file_name": os.path.basename(batch["wav_paths"][0]),
                    "pred_score": round(pred_mos, 2),
                }
            )
    # -------------------------------

    if output_path is None:
        save_dir = getattr(cfg, "moe_model_dir", "./results")
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"MOE_result_{split}.csv")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Inference completed. Saving to:\n\t{output_path}")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_file_name", "pred_score"])
        writer.writeheader()
        writer.writerows(rows)
    # -------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "validation", "test"]
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    inference(
        cfg,
        split=args.split,
        ckpt_path=args.ckpt,
        output_path=args.output,
        device_override=args.device,
    )
