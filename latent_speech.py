import numpy as np
import sounddevice as sd
import torch

SR = 16000


# -------------------------
# vowel-like formant filter
# -------------------------

def vowel_filter(tone, formant):
    t = np.linspace(0, 1, len(tone))
    env = np.sin(np.pi * t) ** 2
    return tone * env * formant


# -------------------------
# syllable generator
# -------------------------

def make_syllable(pitch, dur, color):

    t = np.linspace(0, dur, int(SR * dur), False)

    # harmonic stack
    sig = (
        np.sin(2*np.pi*pitch*t) +
        0.5*np.sin(2*np.pi*pitch*2*t) +
        0.3*np.sin(2*np.pi*pitch*3*t)
    )

    sig = vowel_filter(sig, color)
    return sig


# -------------------------
# main speak
# -------------------------

def speak_latent_syllables(z, decision="react"):

    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()

    z = np.asarray(z).reshape(-1)

    energy = float(np.mean(np.abs(z)))

    # -------------------------
    # speech structure
    # -------------------------

    base_pitch = 140 + z[0]*100
    syllables = int(2 + abs(z[1])*4)
    syllables = max(2, min(6, syllables))

    dur = 0.12 + abs(z[2])*0.15
    color = 0.6 + abs(z[3])*0.6

    if decision == "investigate":
        base_pitch *= 1.3
        syllables += 1

    if decision == "uncertain":
        base_pitch *= 0.7
        dur *= 0.7

    # -------------------------
    # build word
    # -------------------------

    chunks = []

    for i in range(syllables):

        pitch = base_pitch * (1 + 0.15*np.sin(i))
        syl = make_syllable(pitch, dur, color)

        pause = np.zeros(int(SR * 0.03))
        chunks.append(syl)
        chunks.append(pause)

    out = np.concatenate(chunks)

    out /= (np.max(np.abs(out)) + 1e-6)
    out *= 0.5

    try:
        sd.play(out, SR, blocking=False)
    except Exception as e:
        print("speech error:", e)
