import sounddevice as sd
import numpy as np
import torch
import torch.nn as nn
import librosa

from config import AUDIO_SR, AUDIO_CHUNK


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(16, 64)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)


encoder = AudioEncoder()


def read_feature():

    try:
        n = int(AUDIO_SR * AUDIO_CHUNK)

        audio = sd.rec(
            n,
            samplerate=AUDIO_SR,
            channels=1,
            blocking=True
        )

        audio = audio.flatten()

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=AUDIO_SR,
            n_mels=32
        )

        mel = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            feat = encoder(mel)

        return feat

    except Exception as e:
        print("AUDIO DATCHIK NOTO'G'RISI — o'chirilgan:", e)
        return None
