import signal
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from threading import Thread
from queue import Queue
import numpy as np

from config import LR, GRAD_CLIP
from vision import get_cam, read_feature as read_v

# audio stream
from audio_stream import read_feature as read_a

# decision layer
from decision import decision_engine

# latent self speech
from self_speech import speak_from_latent

from brain import model, init_state
from memory import (
    maybe_save, try_load,
    remember, retrieve,
    save_ltm, load_ltm,
    decay_and_forget,
    consolidate_memory
)

# -------------------------
# Settings
# -------------------------

AUDIO_ENABLED = True
SPEECH_INTERVAL = 1.2   # sekund
SAVE_INTERVAL = 300     # 5 minut

# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# State load
# -------------------------

state = try_load(model)
if state is None:
    state = init_state()

load_ltm()

# -------------------------
# IO setup
# -------------------------

q = Queue()
cam = get_cam()

# -------------------------
# Threads
# -------------------------

def vision_loop():
    if cam is None:
        print("KAMERA YO‘Q — vision o‘chiq")
        return

    while True:
        f = read_v(cam)
        if f is not None:
            q.put(("v", f))


def audio_loop():
    while True:
        f = read_a()
        if f is not None:
            q.put(("a", f))


# -------------------------
# Safe shutdown
# -------------------------

RUNNING = True

def safe_shutdown(*args):
    global RUNNING
    RUNNING = False

    print("\n>>> SAFE SHUTDOWN — model saqlanyapti...")

    try:
        maybe_save(model, state)
        save_ltm()
        print(">>> SAVE OK")
    except Exception as e:
        print(">>> SAVE ERROR:", e)

    sys.exit(0)


signal.signal(signal.SIGINT, safe_shutdown)
signal.signal(signal.SIGTERM, safe_shutdown)

# -------------------------
# Start threads
# -------------------------

Thread(target=vision_loop, daemon=True).start()
Thread(target=audio_loop, daemon=True).start()

print("CORE v10 adaptive ishlayapti")

# -------------------------
# Stats
# -------------------------

curiosity_hist = []
step = 0

last_save = time.time()
last_speech = 0

# -------------------------
# Main loop
# -------------------------

while RUNNING:

    typ, feat = q.get()

    state, pred, goal, action, conf, z, gate_w = model(typ, feat, state)
    state = state.detach()

    # -------------------------
    # Loss
    # -------------------------

    pred_loss = F.mse_loss(pred, z.detach())
    action_energy = action.pow(2).mean()

    loss = pred_loss * 0.01 + 0.01 * action_energy

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    # -------------------------
    # Curiosity
    # -------------------------

    curiosity = pred_loss.item()
    curiosity_hist.append(curiosity)

    if len(curiosity_hist) > 200:
        curiosity_hist.pop(0)

    mean_c = np.mean(curiosity_hist)
    std_c = np.std(curiosity_hist) + 1e-6
    dyn_thresh = mean_c + 2 * std_c

    # -------------------------
    # Memory write
    # -------------------------

    if curiosity > dyn_thresh:
        remember(z, curiosity)
        print("!!! YANGI SIGNAL", round(curiosity, 3))

    # -------------------------
    # Reaction logic
    # -------------------------

    act_power = action.abs().mean().item()
    conf_v = conf.item()

    if act_power > 0.4 * conf_v:
        print(">>> REAKSIYA |", typ,
              "power", round(act_power, 3),
              "conf", round(conf_v, 3))

    # -------------------------
    # Decision layer
    # -------------------------

    decision = decision_engine.decide(
        curiosity=curiosity,
        dyn_thresh=dyn_thresh,
        act_power=act_power,
        conf=conf_v,
        gate=gate_w.item()
    )

    if decision != "idle":
        print("QAROR:", decision)

        now = time.time()
        if AUDIO_ENABLED and now - last_speech > SPEECH_INTERVAL:
            speak_from_latent(z)
            last_speech = now

    # -------------------------
    # Decision reaction map
    # -------------------------

    if decision == "investigate":
        print(">>> yangi signalni chuqur o‘rganish rejimi")

    elif decision == "react":
        print(">>> faol javob rejimi")

    elif decision == "focus":
        print(">>> sensor fokus kuchaydi")

    elif decision == "uncertain":
        print(">>> noaniqlik holati")

    # -------------------------
    # Memory decay
    # -------------------------

    if step % 50 == 0:
        decay_and_forget()

    # -------------------------
    # Memory retrieve test
    # -------------------------

    if step % 500 == 0:
        mem = retrieve(z, 3)
        if mem is not None:
            print("xotira topildi", mem.shape)

    # -------------------------
    # Consolidation
    # -------------------------

    if step % 2000 == 0:
        consolidate_memory()

    # -------------------------
    # Debug
    # -------------------------

    if step % 200 == 0:
        print(
            "qadam", step,
            "loss", round(loss.item(), 4),
            "cur", round(curiosity, 4),
            "thr", round(dyn_thresh, 4),
            "gate", round(gate_w.item(), 3)
        )

    # -------------------------
    # Step save
    # -------------------------

    if step % 1000 == 0:
        maybe_save(model, state)

    if step % 3000 == 0:
        save_ltm()

    # -------------------------
    # Time autosave
    # -------------------------

    if time.time() - last_save > SAVE_INTERVAL:
        print(">>> AUTO SAVE (time)")
        maybe_save(model, state)
        save_ltm()
        last_save = time.time()

    step += 1
