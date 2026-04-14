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


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def inference(cfg):
    chkpt_dir = cfg.moe_model_dir

    # dataset_key = "validation"
    dataset_key = "test"

    dataset_label = f"{dataset_key}_list"
    dataset_list = cfg[dataset_label]
    # dataset_wav_dir = os.path.join(cfg["wav_dir"], dataset_key)
    dataset_wav_dir = cfg["wav_dir"]

    # dataset_wav_dir = "/root/autodl-tmp/src/Data/Clotho_data/evaluation"
    print("Perform inference on the following dataset with following checkpoint.")
    # print(f"\tchkpt:        {chkpt_path}")
    print(f"\tdata list:    {dataset_list}")
    print(f"\twav dir:      {dataset_wav_dir}")
    device = torch.device(cfg["device"])
    # -------------------------------

    test_ds = get_infdataset(
        txt_file_path=dataset_list,
        wav_dir=dataset_wav_dir,
        max_sec=cfg["max_len"],
        sr=16000,
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

    # -------- write results --------
    # result_path = os.path.join(chkpt_dir, f"new_inference_result_for_{dataset_key}.csv")
    result_path = os.path.join(chkpt_dir, f"MOE_test_result_for_{dataset_key}.csv")
    print(
        f"Inference has completed. Results will be written to the following file: \n\t{result_path}"
    )
    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_file_name", "pred_score"])
        writer.writeheader()
        writer.writerows(rows)
    # # # -------------------------------
    return
    # -------------------------------


if __name__ == "__main__":

    cfg_path = "config.yaml"
    cfg = load_config(cfg_path)
    inference(cfg)
