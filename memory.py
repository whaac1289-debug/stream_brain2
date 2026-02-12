# memory.py
import torch
from config import SAVE_PATH, MEM_NOVELTY_THRESHOLD

MEM_FILE = "ltm.pt"

# (vec, score, age)
memory_bank = []


# -------------------------
# SAVE / LOAD CORE
# -------------------------

def maybe_save(model, state):
    torch.save({
        "model": model.state_dict(),
        "state": state
    }, SAVE_PATH)


def try_load(model):
    try:
        d = torch.load(SAVE_PATH)
        model.load_state_dict(d["model"])
        print("CORE LOADED")
        return d["state"]
    except:
        return None


# -------------------------
# EXPERIENCE SCORE
# -------------------------

def experience_score(z, pred_loss, action):
    novelty = z.std().item()
    energy = action.pow(2).mean().item()
    return pred_loss + 0.5 * novelty + 0.1 * energy


# -------------------------
# UTIL
# -------------------------

def _flat(x):
    return x.view(-1)


# -------------------------
# NOVELTY CHECK
# -------------------------

def _novel(vec):
    if not memory_bank:
        return True

    v = _flat(vec.cpu())

    mem = torch.stack([_flat(m[0]) for m in memory_bank])
    sims = torch.nn.functional.cosine_similarity(
        v.unsqueeze(0),
        mem,
        dim=1
    )

    return sims.max().item() < (1 - MEM_NOVELTY_THRESHOLD)


# -------------------------
# REMEMBER
# -------------------------

def remember(vec, score):
    global memory_bank

    if _novel(vec):
        memory_bank.append((vec.detach().cpu(), float(score), 0))

    if len(memory_bank) > 6000:
        memory_bank.sort(key=lambda x: x[1], reverse=True)
        memory_bank = memory_bank[:4000]


# -------------------------
# RETRIEVE (SAFE)
# -------------------------

def retrieve(query_vec, k=5):

    if not memory_bank:
        return None

    q = _flat(query_vec.cpu())

    mem_vecs = torch.stack([_flat(m[0]) for m in memory_bank])

    sims = torch.nn.functional.cosine_similarity(
        q.unsqueeze(0),
        mem_vecs,
        dim=1
    )

    k = min(k, len(mem_vecs))
    vals, idx = torch.topk(sims, k=k)

    idx = idx.clamp(max=len(mem_vecs)-1)

    return mem_vecs[idx]


# -------------------------
# DECAY
# -------------------------

def decay_and_forget():
    global memory_bank

    new_bank = []

    for vec, score, age in memory_bank:
        age += 1
        score *= 0.995

        if score > 0.05:
            new_bank.append((vec, score, age))

    memory_bank = new_bank


# -------------------------
# CONSOLIDATE
# -------------------------

def consolidate_memory():

    global memory_bank

    if len(memory_bank) < 50:
        return

    print("SLEEP: memory consolidation")

    zs = torch.stack([_flat(m[0]) for m in memory_bank])
    center = zs.mean(dim=0)

    memory_bank = [(center, 1.0, 0)]


# -------------------------
# LTM SAVE/LOAD
# -------------------------

def save_ltm():
    torch.save(memory_bank, MEM_FILE)
    print("LTM saqlandi:", len(memory_bank))


def load_ltm():
    global memory_bank
    try:
        memory_bank = torch.load(MEM_FILE)
        print("LTM yuklandi:", len(memory_bank))
    except:
        memory_bank = []
