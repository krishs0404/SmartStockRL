import argparse
import os
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
from inv_env import InventoryEnv
from datetime import datetime

def make_env(seed = None):
    env = InventoryEnv(seed = seed)
    return Monitor(env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices = ["ppo", "dqn"], default = "ppo")
    ap.add_argument("--steps", type = int, default = 200_000)
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--eval-every", type = int, default = 10_000)
    ap.add_argument("--eval-episodes", type = int, default = 5)
    ap.add_argument("--models-dir", type = str, default = "models")
    ap.add_argument("--logs-dir", type = str, default = "logs")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #directories
    os.makedirs(args.models_dir, exist_ok = True)
    os.makedirs(args.logs_dir, exist_ok = True)

    #Repro
    set_random_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Envs
    env = make_env(seed = args.seed)
    eval_env = make_env(seed = args.seed + 1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = args.models_dir,
        log_path = args.logs_dir,
        eval_freq = args.eval_every,
        n_eval_episodes = args.eval_episodes,
        deterministic = True,
    )

    #TensorBoard under logs/tb/
    tb_logdir = os.path.join(args.logs_dir, "tb")

    #Algo
    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose = 1,
            gamma = 0.99,
            gae_lambda = 0.95,
            ent_coef = 0.01,
            learning_rate = 3e-4,
            n_steps = 1024,
            tensorboard_log = tb_logdir,
            seed = args.seed,
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            verbose = 1,
            learning_rate = 1e-3,
            buffer_size = 50_000,
            learning_starts = 1_000,
            target_update_interval = 1_000,
            train_freq = 4,
            batch_size = 256,
            gamma = 0.99,
            tensorboard_log = tb_logdir,
            seed = args.seed,
        )

    #Learn
    model.learn(total_timesteps = args.steps, callback = eval_cb)
    algo_name = args.algo.upper()
    base_name = f"SmartStockRL_{algo_name}_{timestamp}"
   #Save final checkpoint
    final_path = os.path.join(args.models_dir, base_name)
    model.save(final_path)

    env.close()
    eval_env.close()
    print(f"[done] saved best model")
    print(f"[logs] evals logs + Tensorboard at {args.logs_dir}")

if __name__ == "__main__":
    main()