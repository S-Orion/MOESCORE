import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio
import numpy as np
from torch.nn.functional import pad as pad1d


def get_dataset(
    txt_file_path,
    wav_dir,
    max_sec,
    sr,
    org_max=10.0,
    org_min=0.0,
    is_train=True,
):
    return XACLEDataset(
        txt_file_path,
        wav_dir,
        max_sec,
        sr,
        org_max,
        org_min,
        is_train=is_train,
    )


def get_infdataset(txt_file_path, wav_dir, max_sec, sr):
    return XACLEINFDataset(txt_file_path, wav_dir, max_sec, sr)


class XACLEDataset(Dataset):
    def __init__(
        self,
        txt_file_path: str,
        wav_dir: str,
        max_sec: int = 10,
        sr: int = 16_000,
        org_max: float = 10.0,
        org_min: float = 0.0,
        is_train: bool = True,
    ):
        super().__init__()
        df = pd.read_csv(txt_file_path)
        self.wav_dir = wav_dir
        self.sr = sr
        self.clap_sr = 48000
        self.mga_sr = 32000
        self.wav_max_len = int(max_sec * self.sr)
        self.mga_wav_max_len = int(max_sec * self.mga_sr)
        self.org_mid = (org_max + org_min) / 2
        self.norm_denom = 5.0
        bins = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        labels = [f"{i}-{i+1}" for i in range(0, 10)]

        self.df = df
        self.is_train = is_train
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=self.mga_sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav_path = os.path.join(self.wav_dir, r["wav_file_name"].lstrip("/"))
        wav, sr = torchaudio.load(wav_path)
        audio_data = wav
        if sr != self.mga_sr:
            mga_wav = self.resampler(wav)
        mga_audio_data = mga_wav.float()

        mos = float(r["average_score"])
        mos_norm = (mos - self.org_mid) / self.norm_denom
        caption = r["text"]
        num_class = int(r["num_class"])

        return dict(
            wav=audio_data,
            score=mos_norm,
            caption=caption,
            num_class=num_class,
            wav_path=wav_path,
            score_raw=mos,
            mga_wav=mga_audio_data,
        )

    def collate_fn(self, batch):
        # --- wav padding / trimming --------------------------------------
        wav_max_len = self.wav_max_len
        wavs = [b["wav"] for b in batch]
        wav_fixed = []
        wav_real_lens = []
        for wav in wavs:
            pad_len = wav_max_len - wav.shape[1]
            if pad_len > 0:
                padded = pad1d(wav, (0, pad_len), mode="constant", value=0.0)
            else:
                padded = wav[:, :wav_max_len]
            wav_fixed.append(padded)
            wav_real_lens.append(wav.shape[1])
        wav_batch = torch.stack(wav_fixed)
        wav_lens = torch.tensor(wav_real_lens, dtype=torch.long)


        mga_wav_max_len = self.mga_wav_max_len
        mga_wavs = [b["mga_wav"] for b in batch]
        wav_fixed = []
        for wav in mga_wavs:
            pad_len = mga_wav_max_len - wav.shape[1]
            if pad_len > 0:
                padded = pad1d(wav, (0, pad_len), mode="constant", value=0.0)
            else:
                padded = wav[:, :mga_wav_max_len]
            wav_fixed.append(padded)
        mga_wav_batch = torch.stack(wav_fixed)

        # --- simple tensors ----------------------------------------------
        mos_score = torch.tensor([b["score"] for b in batch], dtype=torch.float)
        num_class = torch.tensor([b["num_class"] for b in batch], dtype=torch.long)

        # --- caption tokens ----------------------------------------------
        captions = [b["caption"] for b in batch]

        return dict(
            wavs=wav_batch,
            mga_wavs=mga_wav_batch,
            scores=mos_score,
            captions=captions,
            num_class=num_class,
            wav_paths=[b["wav_path"] for b in batch],
            wav_lens=wav_lens,
            scores_raw=torch.tensor([b["score_raw"] for b in batch], dtype=torch.float),
        )


class XACLEINFDataset(Dataset):
    def __init__(
        self,
        txt_file_path: str,
        wav_dir: str,
        max_sec: int = 10,
        sr: int = 16_000,
    ):
        super().__init__()
        df = pd.read_csv(txt_file_path)
        self.wav_dir = wav_dir
        self.wav_max_len = int(max_sec * sr)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav_path = os.path.join(self.wav_dir, r["wav_file_name"].lstrip("/"))
        wav, _ = torchaudio.load(wav_path)
        caption = r["text"]
        return dict(wav=wav, caption=caption, wav_path=wav_path)

    def collate_fn(self, batch):
        # --- wav padding / trimming --------------------------------------
        wav_max_len = self.wav_max_len
        wavs = [b["wav"] for b in batch]
        wav_fixed = []
        for wav in wavs:
            pad_len = wav_max_len - wav.shape[1]
            if pad_len > 0:
                padded = pad1d(wav, (0, pad_len), mode="constant", value=0.0)
            else:
                padded = wav[:, :wav_max_len]
            wav_fixed.append(padded)
        wav_batch = torch.stack(wav_fixed)

        # --- caption tokens ----------------------------------------------
        captions = [b["caption"] for b in batch]

        return dict(
            wavs=wav_batch,
            caption_tokens=captions,
            wav_paths=[b["wav_path"] for b in batch],
        )