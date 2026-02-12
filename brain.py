import torch
import torch.nn as nn

VISION_DIM = 128
AUDIO_DIM = 64

LATENT_DIM = 96
STATE_DIM = 192

ACTION_DIM = 16
GOAL_DIM = 32


# -------------------------
# Projector
# -------------------------

class Projector(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, LATENT_DIM),
            nn.ReLU(),
            nn.Linear(LATENT_DIM, LATENT_DIM)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Modality Gate
# -------------------------

class ModalityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 1)

    def forward(self, z):
        w = torch.sigmoid(self.fc(z))
        return z * w, w


# -------------------------
# Core
# -------------------------

class Core(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRUCell(LATENT_DIM, STATE_DIM)
        self.norm = nn.LayerNorm(STATE_DIM)

        self.pred_head = nn.Linear(STATE_DIM, LATENT_DIM)

        self.goal_head = nn.Sequential(
            nn.Linear(STATE_DIM, GOAL_DIM),
            nn.Tanh()
        )

        self.action_head = nn.Sequential(
            nn.Linear(STATE_DIM + GOAL_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )

        # sensor confidence
        self.conf_head = nn.Sequential(
            nn.Linear(STATE_DIM, 1),
            nn.Sigmoid()
        )

    def forward(self, z, state):

        s = self.gru(z, state)
        s = self.norm(s)

        pred = self.pred_head(s)
        goal = self.goal_head(s)

        act_in = torch.cat([s, goal], dim=1)
        action = self.action_head(act_in)

        conf = self.conf_head(s)

        return s, pred, goal, action, conf


# -------------------------
# Unified Brain v7
# -------------------------

class UnifiedBrain(nn.Module):
    def __init__(self):
        super().__init__()

        self.projectors = nn.ModuleDict({
            "v": Projector(VISION_DIM),
            "a": Projector(AUDIO_DIM),
        })

        self.gate = ModalityGate()
        self.core = Core()

    def forward(self, modality, feat, state):

        z = self.projectors[modality](feat)

        z, gate_w = self.gate(z)

        state, pred, goal, action, conf = self.core(z, state)

        return state, pred, goal, action, conf, z, gate_w


model = UnifiedBrain()


def init_state(batch=1):
    return torch.zeros(batch, STATE_DIM)
