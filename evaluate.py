import argparse, numpy as np, pandas as pd, os
from stable_baselines3 import PPO, DQN
from inv_env import InventoryEnv
from baselines import naive_policy, sS_policy

def rollout_model(model, episodes=20):
    env = InventoryEnv(seed=999)
    rows = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        sales = unmet = inv_sum = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            sales += info["sales"]
            unmet += info["unmet"]
            inv_sum += obs[0]  # current inventory after step
            steps += 1
            done = term or trunc
        rows.append({
            "return": ep_ret,
            "fill_rate": sales / (sales + unmet + 1e-9),
            "avg_inventory": inv_sum / max(steps,1),
            "stockouts": unmet,
        })
    return pd.DataFrame(rows)

def rollout_baseline(policy_fn, params, episodes=20):
    env = InventoryEnv(seed=999)
    rows = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        sales = unmet = inv_sum = 0
        steps = 0
        while not done:
            action = policy_fn(obs, params)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            sales += info["sales"]
            unmet += info["unmet"]
            inv_sum += obs[0]
            steps += 1
            done = term or trunc
        rows.append({
            "return": ep_ret,
            "fill_rate": sales / (sales + unmet + 1e-9),
            "avg_inventory": inv_sum / max(steps,1),
            "stockouts": unmet,
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo","dqn"], default="ppo")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--model", type=str, default="models/best_model.zip")
    ap.add_argument("--out", type=str, default="runs/eval_metrics.csv")
    args = ap.parse_args()

    Model = PPO if args.algo == "ppo" else DQN
    model = Model.load(args.model)

    rl = rollout_model(model, episodes=args.episodes)
    naive = rollout_baseline(lambda o,p: max(int(p["T"] - o[0]),0), {"T": 50}, episodes=args.episodes)
    ss = rollout_baseline(lambda o,p: (max(int(p["S"]-o[0]),0) if o[0] < p["s"] else 0), {"s":20,"S":80}, episodes=args.episodes)

    rl["policy"] = "RL"
    naive["policy"] = "Naive(T=50)"
    ss["policy"] = "sS(20,80)"

    df = pd.concat([rl, naive, ss], ignore_index=True)
    print(df.groupby("policy").agg(["mean","std"]))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
