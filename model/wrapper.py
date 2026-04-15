import torch
import torch.nn as nn
from unilm.beats.BEATs import BEATs, BEATsConfig
import laion_clap


class AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device

        beats_ckpt_path = cfg.audio_wrapper.beats_ckpt_path

        ckpt = torch.load(beats_ckpt_path, map_location=self.device)
        beats_config = BEATsConfig(ckpt["cfg"])
        self.model = BEATs(beats_config)
        self.model.load_state_dict(ckpt["model"])
        self.model.train()
        self.model.to(self.device)

        self.sr = 16000
        self.num_encoder_layers = len(self.model.encoder.layers)

        self.layer_weights = nn.Parameter(
            torch.ones(self.num_encoder_layers) / self.num_encoder_layers
        )

        proj_output_dim = cfg.audio_wrapper.proj_output_dim
        self.audio_feats_proj = nn.Sequential(
            nn.LayerNorm(self.model.cfg.encoder_embed_dim),
            nn.Linear(self.model.cfg.encoder_embed_dim, proj_output_dim),
            nn.GELU(),
            nn.LayerNorm(proj_output_dim),
        ).to(self.device)

        for i, layer in enumerate(self.model.encoder.layers):
            if i < len(self.model.encoder.layers) - 2:
                for p in layer.parameters():
                    p.requires_grad_(False)
            else:
                for p in layer.parameters():
                    p.requires_grad_(True)


    def get_audio_sequence(self, wavs, wavs_real_len):
        batch = wavs.squeeze(1).to(self.device)
        batch = (batch - batch.mean(dim=-1, keepdim=True)) / (
            batch.std(dim=-1, keepdim=True) + 1e-8
        )
        max_len = batch.shape[-1]
        time_padding_mask = torch.arange(max_len, device=self.device).expand(
            batch.shape[0], max_len
        ) >= wavs_real_len.unsqueeze(1)

        x = self.model.preprocess(batch)
        if time_padding_mask is not None:
            time_padding_mask = self.model.forward_padding_mask(x, time_padding_mask)
        x = x.unsqueeze(1)
        x = self.model.patch_embedding(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.transpose(1, 2)
        x = self.model.layer_norm(x)

        if time_padding_mask is not None:
            time_padding_mask = self.model.forward_padding_mask(x, time_padding_mask)


        if self.model.post_extract_proj is not None:
            x = self.model.post_extract_proj(x)
        x = self.model.dropout_input(x)

        hidden_states = []
        features, layer_results = self.model.encoder(
            x, padding_mask=time_padding_mask, layer=12
        )
        for item in layer_results[1:]:
            hidden_states.append(item[0].transpose(0, 1))

        hidden_states = torch.stack(hidden_states, dim=0)
        weights = torch.softmax(self.layer_weights, dim=0)
        audio_features = torch.sum(hidden_states * weights.view(-1, 1, 1, 1), dim=0)
        audio_features = self.audio_feats_proj(audio_features)

        if time_padding_mask is None:
            raise RuntimeError("BEATs failed to generate the sequence mask.")

        return audio_features.to(torch.float32), time_padding_mask.bool()


class TextEncoder(nn.Module):
    def __init__(self, cfg, freeze_pretrained=True):
        super().__init__()
        self.device = cfg.device

        self.model = laion_clap.CLAP_Module(enable_fusion=False).to(self.device)
        self.model.load_ckpt(cfg.clap_encoder.ckpt_path)

        embed_dim = self.model.model_cfg["embed_dim"]
        self.ln_before_proj = nn.LayerNorm(embed_dim).to(self.device)

        self.text_feats_proj = nn.Sequential(
            nn.LayerNorm(self.model.model_cfg["embed_dim"]),
            nn.Linear(self.model.model_cfg["embed_dim"], 512),
            nn.GELU(),
            nn.LayerNorm(512),
        ).to(self.device)


        for param in self.model.model.parameters():
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

        num_layers = len(self.model.model.text_branch.encoder.layer)
        for p in self.model.model.text_branch.parameters():
            p.requires_grad = False
        
        if not freeze_pretrained:
            for i in range(num_layers - 2, num_layers):
                for p in self.model.model.text_branch.encoder.layer[i].parameters():
                    p.requires_grad = True


    def get_text_sequence(self, text_data):
        # text_embed = self.model.get_text_embedding(text_data, use_tensor=True)
        inputs = self.model.tokenizer(text_data).to(self.device)
        attn = inputs["attention_mask"]  # 1 valid, 0 pad
        text_pad_mask = (attn == 0).bool()  # True=pad


        # with torch.no_grad():
        text_out = self.model.model.text_branch(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )

        text_embed = text_out.hidden_states[-1]
        # text_embed = torch.stack(text_embed[-4:], dim=0).mean(dim=0)
        # text_embed = self.ln_before_proj(text_embed)
        # text_feats = self.text_feats_proj(text_embed).float()

        # return text_feats.to(torch.float32), text_pad_mask
        return text_embed.to(torch.float32), text_pad_mask

    def get_audio(self, audio_data):
        audio_embed = self.model.get_audio_embedding_from_data(audio_data.squeeze(1), use_tensor=True)
        return audio_embed
    
    def get_text(self, text_data):
        text_embed = self.model.get_text_embedding(text_data, use_tensor=True)
        return text_embed

    def get_text_sequence_projected(self, text_data):
        inputs = self.model.tokenizer(
            text_data
        ).to(self.device)

        text_out = self.model.model.text_branch(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )

        hidden = text_out.hidden_states[-1]  # [B, seq_len, 768]
        proj = self.model.model.text_projection  # 768 → 512

        # [B, seq_len, 768] → [B, seq_len, 512]
        word_embeds_proj = proj(hidden[:, 1:, :])
        pad_mask = (inputs["attention_mask"][:, 1:] == 0).bool()

        return word_embeds_proj.float(), pad_mask