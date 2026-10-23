# Cloud deployment — live scheduler service

This folder deploys the **actual trained scheduler** as a cloud service:
the Dueling Double DQN runs in the cloud, dispatching real Borg-pool tasks
with its safety mask and Gradient×Input attribution computed live.

## What's inside

| Path | What |
|---|---|
| `app.py` | FastAPI service (live decision API + dashboard) |
| `core/` | The project's pure-NumPy modules (env, Q-network, agent) |
| `model/agent_checkpoint.npz` | Trained weights (seed 0, 300 episodes) |
| `model/borg_task_pool.npz` | The cleaned 5,000-task Borg pool |
| `static/` | The results dashboard (same data as the GitHub Pages site) |
| `Dockerfile` / `render.yaml` | One-command cloud deployment |

## API

- `GET /api/health` — liveness + model manifest
- `GET /api/decision` — dispatch one task live; returns chosen VM, all eight
  Q-values, masked VMs (>90% utilization), reward, deadline outcome, cost,
  energy, and the top-5 attribution features
- `POST /api/reset` — restart the episode
- `GET /api/schedulers` — the three-trace comparison data

## Deploy on Render (recommended, free, no credit card)

1. Sign up at [render.com](https://render.com) (GitHub login).
2. New → **Web Service** → connect the repository
   `Sainath-Reddy7/Explainable-Reinforcement-Learning-for-Energy-Efficient-Cloud-Resource-Scheduling`.
3. Branch: `cloud-deploy` · Root directory: `cloud_service`.
   Render auto-detects the Dockerfile.
4. Instance type: **Free**. Create Web Service.
5. In ~2 minutes the service is live at `https://<name>.onrender.com` —
   the dashboard's "Try the live scheduler" panel activates automatically.

## Run locally

```bash
cd cloud_service
pip install -r requirements.txt
uvicorn app:app --port 8000
# http://localhost:8000
```

## Why Render over the alternatives

- **Railway**: trial credit requires a card eventually; Render's free tier
  does not.
- **Fly.io**: requires a credit card at signup.
- **AWS/GCP/Azure**: real clusters need billing setup — overkill for a demo
  service; this app is pure NumPy and runs comfortably in a 512 MB free
  container.
- Free-tier note: the service sleeps after ~15 min idle; the first request
  after sleep takes a few seconds to wake it.
