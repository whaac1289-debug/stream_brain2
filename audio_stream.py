# audio_stream.py

import subprocess
import numpy as np
import torch
import librosa
import time

AUDIO_DIM = 64
SR = 22050
CHUNK_SEC = 1.0
CHUNK_SAMPLES = int(SR * CHUNK_SEC)

STREAM_URL = "http://live.hitfm.uz/hitfmuz"


# ---------- FFMPEG LIVE PIPE ----------

def start_ffmpeg():
    print("AUDIO: ffmpeg live pipe start...")

    cmd = [
        "ffmpeg",
        "-loglevel", "quiet",
        "-i", STREAM_URL,
        "-ac", "1",              # mono
        "-ar", str(SR),          # sample rate
        "-f", "f32le",           # float32 raw
        "-"
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        bufsize=10**7
    )


ffmpeg_proc = start_ffmpeg()


# ---------- FEATURE READER ----------

def read_feature():

    try:
        need_bytes = CHUNK_SAMPLES * 4  # float32 = 4 byte

        raw = ffmpeg_proc.stdout.read(need_bytes)

        if len(raw) < need_bytes:
            print("AUDIO: yetarli data kelmadi")
            return None

        samples = np.frombuffer(raw, dtype=np.float32)

        #print("AUDIO: live samples =", len(samples))

        # ----- MEL -----

        t0 = time.time()

        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=SR,
            n_mels=32
        )

        #print("AUDIO: mel time =", round(time.time()-t0, 3))

        feat = torch.tensor(mel).mean(dim=1)
        feat = torch.nn.functional.pad(feat, (0, AUDIO_DIM - 32))

        #print("AUDIO: feature tayyor", feat.shape)

        return feat.unsqueeze(0)

    except Exception as e:
        print("AUDIO PIPE XATO:", e)
        return None
