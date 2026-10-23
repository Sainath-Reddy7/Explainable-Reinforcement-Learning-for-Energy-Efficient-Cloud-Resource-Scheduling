"""Cloud service: runs the trained Dueling Double DQN scheduler live.

Endpoints
  GET  /                 dashboard (static)
  GET  /api/health       liveness + model info
  GET  /api/decision     next dispatch decision, computed live (with
                         Gradient x Input attribution and the safety mask)
  POST /api/reset        restart the episode
  GET  /api/schedulers   the three-trace comparison shipped with the paper
"""
import json
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.qnetwork import QNetwork
from core.env import CloudSchedulingEnv
from core.dqn_agent import DQNAgent

HERE = Path(__file__).parent
STATE_DIM, ACTION_DIM = 45, 8
EPISODE_SEED = 7

FEATURES = ["task_cpu", "task_mem", "task_dur", "task_prio", "task_class"]
for i in range(8):
    FEATURES += [f"vm{i}_util", f"vm{i}_power", f"vm{i}_price",
                 f"vm{i}_queue", f"vm{i}_fit"]

app = FastAPI(title="RL-MOTS-XAI live scheduler")

_net = None
_env = None
_state = None
_pool = None
_manifest = {}
_lock = threading.Lock()
_step = 0


@app.on_event("startup")
def load():
    global _net, _env, _state, _pool, _manifest
    ckpt = np.load(HERE / "model" / "agent_checkpoint.npz")
    _net = QNetwork(STATE_DIM, ACTION_DIM)
    _net.set_weights({k: ckpt[k] for k in ckpt.files})
    _pool_npz = np.load(HERE / "model" / "borg_task_pool.npz")
    _pool = {k: _pool_npz[k] for k in _pool_npz.files if _pool_npz[k].ndim == 1}
    _env = CloudSchedulingEnv(_pool, num_vms=8, num_tasks=200, seed=0,
                              adaptive_weights=False, vm_seed=42,
                              deadline_tightness=1.0, use_shaping=True,
                              energy_model=None)
    _state = _env.reset(episode_seed_offset=EPISODE_SEED)
    mf = HERE / "model" / "manifest.json"
    _manifest = json.loads(mf.read_text()) if mf.exists() else {}
    print("model loaded:", len(ckpt.files), "tensors; pool",
          len(next(iter(_pool.values()))), "tasks")


def _decide():
    global _state, _step
    q = _net.forward(_state[None, :], train=False)[0]
    mask = DQNAgent.action_mask(_state)
    masked = q + mask
    action = int(np.argmax(masked))
    phi = _state * _net.input_gradient(_state, action)
    top = np.argsort(-np.abs(phi))[:5]
    next_state, reward, done, info = _env.step(action)
    _state = next_state
    _step += 1
    if done:
        _state = _env.reset(episode_seed_offset=EPISODE_SEED + _step)
        _step = 0
    order = np.argsort(-q)
    return {
        "step": _step,
        "chosen_vm": action,
        "q_values": [round(float(v), 3) for v in q],
        "ranking": [int(i) for i in order[:3]],
        "masked_vms": [int(i) for i in np.where(mask < 0)[0]],
        "reward": round(float(reward), 3),
        "deadline_met": bool(info.get("deadline_met", True)),
        "cost_usd": round(float(info.get("cost", 0.0)), 3),
        "energy_wh": round(float(info.get("energy_wh", 0.0)), 3),
        "top_features": [{"feature": FEATURES[i], "phi": round(float(phi[i]), 3)}
                         for i in top],
    }


@app.get("/api/health")
def health():
    return {"ok": _net is not None, "model": _manifest, "episode_step": _step}


@app.get("/api/decision")
def decision():
    with _lock:
        return JSONResponse(_decide())


@app.post("/api/reset")
def reset():
    global _state, _step
    with _lock:
        _state = _env.reset(episode_seed_offset=EPISODE_SEED)
        _step = 0
    return {"ok": True, "step": 0}


@app.get("/api/schedulers")
def schedulers():
    p = HERE / "static" / "data.json"
    return JSONResponse(json.loads(p.read_text())) if p.exists() else {}


app.mount("/", StaticFiles(directory=str(HERE / "static"), html=True), name="static")
