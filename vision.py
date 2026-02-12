import cv2
import torch
import torch.nn as nn
from config import VISION_SIZE


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)


encoder = VisionEncoder()


# ---------- CAMERA AUTO DETECT ----------

def get_cam():
    for i in range(4):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print("Kamera ochilgan indeks:", i)
            return cam

    print("KAMERA TOPILMAGAN — ko'rish qobiliyati o'chirilgan")
    return None


# ---------- FEATURE READER ----------

def read_feature(cam):

    if cam is None:
        return None

    ok, frame = cam.read()
    if not ok:
        return None

    frame = cv2.resize(frame, (VISION_SIZE, VISION_SIZE))
    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255
    frame = frame.unsqueeze(0)

    with torch.no_grad():
        feat = encoder(frame)

    return feat
