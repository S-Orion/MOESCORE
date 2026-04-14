import torch.nn as nn
import torch
from Basic_comp import Expert1, Expert2, Expert3, Expert4

def load_pretrain_weight(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model



class GatingNetwork(nn.Module):
    def __init__(self, input_feature_dim, num_experts=3):
        super().__init__()
        self.net = nn.Sequential(
                    nn.Linear(input_feature_dim, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(512, num_experts),
                    nn.Softmax(dim=-1)
                )

    def forward(self, combined_features):
        return self.net(combined_features)



class MOEModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device

        print("Initializing Expert 1:")
        self.expert1 = Expert1(cfg, device).to(device)
        self.expert1 = load_pretrain_weight(
            self.expert1, cfg.clap_encoder.pretrain_best_ckpt
        )

        print("Initializing Expert 2:")
        self.expert2 = Expert2(cfg, device).to(device)
        self.expert2 = load_pretrain_weight(
            self.expert2, cfg.mga_encoder.pretrain_best_ckpt
        )

        print("Initializing Expert 3:")
        self.expert3 = Expert3(cfg, device).to(device)
        self.expert3 = load_pretrain_weight(
            self.expert3, cfg.m2d_encoder.pretrain_best_ckpt
        )

        print("Initializing Expert 4:")
        self.expert4 = Expert4(cfg, device).to(device)
        self.expert4 = load_pretrain_weight(self.expert4, cfg.pretrain_best_ckpt)

        for expert in [self.expert1, self.expert2, self.expert3, self.expert4]:
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False

        self.gate_input_dim = 7168
        print(f"Gating Network Input Dimension: {self.gate_input_dim}")

        print("Initializing Gating Network...")
        self.gating_network = GatingNetwork(self.gate_input_dim, num_experts=4).to(
            device
        )
        self.gating_network.train()

        print("MOE Model initialization complete.")

    def forward(self, batch):
        with torch.no_grad():
            score1, feat1 = self.expert1(batch)
            score2, feat2 = self.expert2(batch)
            score3, feat3 = self.expert3(batch)
            score4, feat4 = self.expert4(batch)

        combined_features = torch.cat([feat1, feat2, feat3, feat4], dim=-1)

        weights = self.gating_network(combined_features)

        final_score = (
            score1 * weights[:, 0:1]
            + score2 * weights[:, 1:2]
            + score3 * weights[:, 2:3]
            + score4 * weights[:, 3:4]
        )

        return final_score, weights
