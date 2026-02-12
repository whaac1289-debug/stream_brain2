import numpy as np
import sounddevice as sd
import torch

SR = 16000


def speak_latent(z, decision="react"):

    # -------------------------
    # Tensor → numpy → 1D
    # -------------------------

    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()

    z = np.asarray(z).reshape(-1)   # <<< MUHIM TUZATISH

    # -------------------------
    # Control params
    # -------------------------

    energy = float(np.mean(np.abs(z)))
    pitch_base = 180 + float(z[0]) * 120
    dur = 0.25 + energy * 0.4

    if decision == "investigate":
        pitch_base *= 1.4
        dur *= 1.3
    elif decision == "uncertain":
        pitch_base *= 0.7
        dur *= 0.6
    elif decision == "react":
        pitch_base *= 1.1

    # -------------------------
    # Wave generate
    # -------------------------

    t = np.linspace(0, dur, int(SR * dur), False)
    tone = np.zeros_like(t)

    n = min(6, len(z))

    for i in range(n):
        f = pitch_base * (i + 1)
        amp = float(z[i]) * 0.2   # <<< skalyar qilish
        tone += amp * np.sin(2 * np.pi * f * t)

    # envelope
    tone *= np.exp(-3 * t)

    # normalize
    tone /= (np.max(np.abs(tone)) + 1e-6)
    tone *= 0.4

    # -------------------------
    # Play
    # -------------------------

    try:
        sd.play(tone, SR, blocking=False)
    except Exception as e:
        print("AUDIO OUT ERROR:", e)
