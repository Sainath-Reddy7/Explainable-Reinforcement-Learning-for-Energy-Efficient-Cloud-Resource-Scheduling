"""One-shot patcher: adds ablation toggles (dueling / mask / PER / shaping)."""
import io

def patch(path, pairs):
    src = io.open(path, encoding='utf-8').read()
    for old, new in pairs:
        if new in src:
            print(f"  {path}: already patched ({old[:40]!r}...)")
            continue
        assert old in src, f"{path}: pattern not found: {old[:80]!r}"
        src = src.replace(old, new)
    io.open(path, 'w', encoding='utf-8').write(src)
    print(f"  {path}: patched")

# ---------- env.py: use_shaping flag ----------
patch('env.py', [
    ("def __init__(self, task_pool, num_vms=8, num_tasks=200, seed=0,\n                 adaptive_weights=True, vm_seed=42, deadline_tightness=1.0):",
     "def __init__(self, task_pool, num_vms=8, num_tasks=200, seed=0,\n                 adaptive_weights=True, vm_seed=42, deadline_tightness=1.0,\n                 use_shaping=True):"),
    ("        self.gamma_shaping = 0.95   # discount for potential-based shaping",
     "        self.use_shaping = use_shaping\n        self.gamma_shaping = 0.95   # discount for potential-based shaping"),
    ("        shaped = self.gamma_shaping * phi_after - phi_before\n        reward += 0.5 * shaped",
     "        if self.use_shaping:\n            shaped = self.gamma_shaping * phi_after - phi_before\n            reward += 0.5 * shaped"),
])

# ---------- dqn_agent.py: dueling + use_mask flags ----------
patch('dqn_agent.py', [
    ("from qnetwork import QNetwork",
     "from qnetwork import QNetwork, VanillaQNetwork"),
    ("                 dropout=0.1, seed=0):\n        self.action_dim = action_dim",
     "                 dropout=0.1, seed=0, dueling=True, use_mask=True):\n        self.action_dim = action_dim\n        self.use_mask = use_mask"),
    ("""        self.q = QNetwork(state_dim, action_dim, hidden=hidden[:2], lr=lr,
                          dropout=dropout, seed=seed)
        self.target_q = QNetwork(state_dim, action_dim, hidden=hidden[:2], lr=lr,
                                 dropout=dropout, seed=seed + 1)""",
     """        NetCls = QNetwork if dueling else VanillaQNetwork
        self.q = NetCls(state_dim, action_dim, hidden=hidden[:2], lr=lr,
                        dropout=dropout, seed=seed)
        self.target_q = NetCls(state_dim, action_dim, hidden=hidden[:2], lr=lr,
                               dropout=dropout, seed=seed + 1)"""),
    ("    def act(self, state, greedy=False, use_mask=True):",
     "    def act(self, state, greedy=False, use_mask=None):\n        if use_mask is None: use_mask = self.use_mask"),
])

# ---------- train.py: plumb flags ----------
patch('train.py', [
    ('                             deadline_tightness=env_cfg.get("deadline_tightness", 1.0))\n    agent = DQNAgent(',
     '                             deadline_tightness=env_cfg.get("deadline_tightness", 1.0),\n                             use_shaping=env_cfg.get("use_shaping", True))\n    agent = DQNAgent('),
    ('                     eps_decay=ag_cfg["eps_decay"],\n                     seed=seed)',
     '                     eps_decay=ag_cfg["eps_decay"],\n                     seed=seed,\n                     dueling=ag_cfg.get("dueling", True),\n                     use_mask=ag_cfg.get("use_mask", True))'),
    ('                             seed=seed, vm_seed=vm_seed,\n                             deadline_tightness=env_cfg.get("deadline_tightness", 1.0))\n    results = []',
     '                             seed=seed, vm_seed=vm_seed,\n                             deadline_tightness=env_cfg.get("deadline_tightness", 1.0),\n                             use_shaping=env_cfg.get("use_shaping", True))\n    results = []'),
])

print("core patches done")
