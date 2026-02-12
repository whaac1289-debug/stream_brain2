import numpy as np
import sounddevice as sd
from latent_vocab import latent_vocab

SR = 16000


# -------------------------
# Safe tone play
# -------------------------

def play_tone(freq, dur, amp):

    try:
        t = np.linspace(0, dur, int(SR * dur), False)
        tone = amp * np.sin(2 * np.pi * freq * t)

        sd.play(tone, SR, blocking=True)

    except Exception as e:
        print("AUDIO OUT ERROR:", e)


# -------------------------
# Latent → sound + word
# -------------------------

def speak_from_latent(z):

    try:
        cid = latent_vocab.get_cluster(z)
        latent_vocab.push(cid)

        f, d, a = latent_vocab.cluster_to_sound(cid)

        # syllable chiqarish
        play_tone(f, d, a)

        # pattern → word
        word = latent_vocab.get_word_pattern()
        if word:
            print("SELF WORD:", word)

    except Exception as e:
        print("SELF SPEECH ERROR:", e)
