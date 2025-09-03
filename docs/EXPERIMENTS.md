# Experiments Log

Template for tracking runs. Add one entry per training/eval session. Keep short and high-signal.

## Example Entry
- Date: 2025-09-03
- Commit: <hash>
- Grid size: 9
- Algo: MaskablePPO
- Steps: 50k (4 envs)
- Reward weights: lines=10.0, lines_sq=5.0, cells=0.05, holes=0.1, bumpiness=0.01, height=0.02
- Key settings: n_steps=2048 (default), lr=3e-4
- Results (TensorBoard):
  - rollout/ep_rew_mean: ~34–35
  - rollout/ep_len_mean: ~19
- Artifacts:
  - Model: models/ppo_blockpuzzle.zip
  - Logs: logs/ppo/PPO_4
- Notes: First masked run; stable metrics, no clear learning yet at 50k.

## Entries
- Add below this line in reverse chronological order.

---
