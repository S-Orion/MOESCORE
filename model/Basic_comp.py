import torch
import torch.nn as nn
import yaml
from mga_model import ASE
import torch.nn.functional as F
from m2d_model import PortableM2D
import laion_clap
from wrapper import TextEncoder, AudioEncoder


class TemporalAlignModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.text_encoder = TextEncoder(cfg, freeze_pretrained=False)
        self.audio_encoder = AudioEncoder(cfg)
        self.seq_coatt = SeqCoAttn(512, device)

        self.projection = Projection(
            input_dim=cfg["model"]["projection"]["input_dim"],
            hidden_dim=cfg["model"]["projection"]["hidden_dim"],
            activation=cfg["model"]["projection"]["activation"],
            range_clipping=cfg["model"]["projection"]["range_clipping"],
            dropout=cfg["model"]["projection"]["dropout"],
        )

    def forward(self, batch: dict):
        audio_seq, audio_mask = self.audio_encoder.get_audio_sequence(
            batch["wavs"], batch["wav_lens"]
        )

        text_seq, text_mask = self.text_encoder.get_text_sequence(
            batch["captions"]
        )

        cond_emb = self.seq_coatt(audio_seq, text_seq, audio_mask, text_mask)

        # score predict
        mos_hat = self.projection(cond_emb)
        return mos_hat, cond_emb



class Projection(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        activation: str = "ReLU",
        range_clipping: bool = False,
        dropout: float = 0.3,
        task_type: str = "regression",
    ):
        super().__init__()
        self.range_clipping = range_clipping
        self.task_type = task_type
        self.act = (
            getattr(nn, activation)() if isinstance(activation, str) else activation
        )

        output_dim = 1 if task_type == "regression" else 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        if self.range_clipping and task_type == "regression":
            self.out_act = nn.Tanh()

    def forward(self, x):
        """
        x : (B, 1024)
        """
        x = self.net(x)  # (B, 1)

        if self.range_clipping and self.task_type == "regression":
            x = self.out_act(x)
        x = x.squeeze(-1)  # (B, )
        return x


class SeqCoAttn(nn.Module):
    def __init__(self, dim, device, nhead=8, dropout=0.1, activation=nn.ReLU()):
        super().__init__()
        self.dim = dim
        self.device = device

        self.audio_to_text_attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.text_to_audio_attn = nn.MultiheadAttention(dim, nhead, batch_first=True)

        self.audio_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)

        self.audio_res_proj = nn.Linear(dim, dim)
        self.text_res_proj = nn.Linear(dim, dim)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.fusion_norm = nn.LayerNorm(2 * dim)

    def forward(self, audio_seq, text_seq, audio_mask=None, text_mask=None):
        # (B, )
        audio_attended, _ = self.audio_to_text_attn(
            audio_seq, text_seq, text_seq, key_padding_mask=text_mask
        )

        # audio_attended = self.audio_norm(audio_attended + self.audio_res_proj(audio_seq))

        text_attended, _ = self.text_to_audio_attn(
            text_seq, audio_seq, audio_seq, key_padding_mask=audio_mask
        )

        # text_attended = self.text_norm(text_attended + self.text_res_proj(text_seq))

        audio_pooled = self.pool(audio_attended.transpose(1, 2)).squeeze(-1)
        text_pooled = self.pool(text_attended.transpose(1, 2)).squeeze(-1)

        fused = torch.cat([audio_pooled, text_pooled], dim=-1)  # [B, dim]
        # fused = self.dropout(self.activation(fused))
        # fused = self.fusion_norm(fused)
        return fused



class MGA_audio_text_encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device

        mga_cfg_path = cfg.mga_encoder.cfg_path
        checkpoint_path = cfg.mga_encoder.checkpoint_path

        with open(mga_cfg_path, "r") as f:
            mga_config = yaml.safe_load(f)

        self.model = ASE(mga_config).to(self.device)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()

    def forward(self, audio, text):
        text_feats, word_embeds, attn_mask = self.model.encode_text(text)
        text_embeds = self.model.msc(word_embeds, self.model.codebook, attn_mask)
        text_embeds = F.normalize(text_embeds, dim=-1)

        audio = audio.mean(1)
        audio_feats, frame_embeds = self.model.encode_audio(audio)
        audio_embeds = self.model.msc(frame_embeds, self.model.codebook)
        audio_embeds = F.normalize(audio_embeds, dim=-1)

        return audio_embeds, text_embeds



class M2D_audio_text_encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device

        weight_path = cfg.m2d_encoder.weight_path

        self.model = PortableM2D(weight_file=weight_path, flat_features=True).to(
            self.device
        )

        for name, param in self.model.named_parameters():
            if "patch_embed" in name:
                param.requires_grad = False
            elif "audio_proj" in name or "text_proj" in name:
                param.requires_grad = True
            elif (
                "backbone.transformer.layers.10" in name
                or "backbone.transformer.layers.11" in name
                or "text_encoder.transformer.layers.10" in name
                or "text_encoder.transformer.layers.11" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, audio, text):
        audio_embs = self.model.encode_clap_audio_multiscale(audio)
        text_embs = self.model.encode_clap_text(text)

        text_embeds = F.normalize(text_embs, dim=-1)
        audio_embeds = F.normalize(audio_embs, dim=-1)

        return text_embeds, audio_embeds



class CLAP_audio_text_encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        ckpt_path = cfg.clap_encoder.ckpt_path
        enable_fusion = cfg.clap_encoder.enable_fusion

        self.model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion, device=self.device
        )
        self.model.load_ckpt(ckpt_path)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.model.named_parameters():
            if "projection" in name or "mlp" in name:
                param.requires_grad = True

        for name, param in self.model.model.audio_branch.named_parameters():
            if (
                "layers.3" in name
                or "head" in name
                or "norm" in name
                or "layers.2" in name
            ):
                param.requires_grad = True

        self.model.train()

    def forward(self, audio, text):
        audio_embed = self.model.get_audio_embedding_from_data(
            audio.squeeze(1), use_tensor=True
        )
        text_embed = self.model.get_text_embedding(text, use_tensor=True)

        return audio_embed, text_embed



class Expert1(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.model = CLAP_audio_text_encoder(cfg).to(device)

    def forward(self, batch):
        audio_feats, text_feats = self.model(
            batch["clap_wavs"], batch["caption_tokens"]
        )

        mos_hat = torch.sum(audio_feats * text_feats, dim=-1, keepdim=True)
        fused_feats = torch.cat([audio_feats, text_feats], dim=-1)
        return mos_hat, fused_feats



class Expert2(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.model = MGA_audio_text_encoder(cfg).to(device)

    def forward(self, batch):
        audio_feats, text_feats = self.model(batch["wavs"], batch["caption_tokens"])

        mos_hat = torch.sum(audio_feats * text_feats, dim=-1, keepdim=True)
        fused_feats = torch.cat([audio_feats, text_feats], dim=-1)
        return mos_hat, fused_feats



class Expert3(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.model = M2D_audio_text_encoder(cfg).to(device)

        in_dim = 3072
        self.alignment_head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
            return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

        for m in self.alignment_head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=2e-5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch):
        audio_feats, text_feats = self.model(batch["wavs"], batch["caption_tokens"])

        cos = (audio_feats * text_feats).sum(dim=-1, keepdim=True)
        diff = torch.abs(audio_feats - text_feats)
        prod = audio_feats * text_feats
        cross_feats = torch.cat(
            [
                audio_feats,
                text_feats,
                diff,
                prod,
                cos.expand(-1, audio_feats.size(-1)),
            ],
            dim=-1,
        )

        mos_hat = self.alignment_head(cross_feats)
        fused_feats = cross_feats
        return mos_hat, fused_feats



class Expert4(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.model = TemporalAlignModel(cfg, device).to(device)

    def forward(self, batch):
        mos_hat, fused_feats = self.model(batch)
        return mos_hat, fused_feats
