#!/usr/bin/env python3
"""
SAC training script for SO101 active vision
"""

import os
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

from src.rl_training_wrapper import create_rl_env, create_eval_env

def create_training_env(n_envs: int = 4, render: bool = False):
    """Create vectorized training environment"""
    
    def make_env():
        render_mode = "human" if render else "rgb_array"
        env = create_rl_env(render_mode=render_mode)
        env = Monitor(env)
        return env
    
    # Create vectorized environment
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Frame stacking for temporal information
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    return vec_env

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SAC for SO101 active vision")
    parser.add_argument("--timesteps", "-t", type=int, default=200000,
                       help="Total training timesteps")
    parser.add_argument("--envs", "-e", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=5000,
                       help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=10000,
                       help="Model save frequency")
    parser.add_argument("--model-name", "-n", type=str, default="so101_sac_active_vision",
                       help="Model name for saving")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--render", action="store_true",
                       help="Show PyBullet GUI during training")
    parser.add_argument("--multi-robot", action="store_true",
                       help="Use multi-robot environment instead of parallel envs")
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'total_timesteps': args.timesteps,
        'n_envs': args.envs,
        'eval_freq': args.eval_freq,
        'save_freq': args.save_freq,
        'log_interval': 100,
        'model_name': args.model_name,
        'use_wandb': not args.no_wandb,
        'render_training': args.render,
        'multi_robot': args.multi_robot
    }
    
    # Initialize wandb for logging
    if config['use_wandb']:
        run = wandb.init(
            project="so101-active-vision",
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
    else:
        run = None
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create environments
    if config['multi_robot']:
        print(f"🏗️  Creating multi-robot environment with {config['n_envs']} robots...")
        from src.multi_robot_env import SO101MultiRobotEnv
        from stable_baselines3.common.monitor import Monitor
        
        train_env = SO101MultiRobotEnv(
            n_robots=config['n_envs'],
            render_mode="human" if config['render_training'] else "rgb_array"
        )
        train_env = Monitor(train_env)
    else:
        print("🏗️  Creating vectorized training environment...")
        train_env = create_training_env(n_envs=config['n_envs'], render=config['render_training'])
    
    print("🔬 Creating evaluation environment...")  
    eval_env = create_eval_env(render_mode="rgb_array")
    eval_env = Monitor(eval_env)
    
    # SAC hyperparameters optimized for vision tasks
    sac_config = {
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'learning_starts': 5000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'target_update_interval': 1,
        'target_entropy': 'auto',
        'use_sde': False,
        'sde_sample_freq': -1,
        'policy_kwargs': {
            'net_arch': dict(pi=[256, 256], qf=[256, 256]),
            'activation_fn': torch.nn.ReLU,
            'normalize_images': True,  # Important for image observations
        },
        'tensorboard_log': "./logs/",
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'verbose': 1
    }
    
    print(f"🚀 Training SAC on {sac_config['device']}")
    print(f"📊 Policy networks: {sac_config['policy_kwargs']['net_arch']}")
    
    # Create SAC model
    model = SAC(
        "MultiInputPolicy",  # For dict observations with images
        train_env,
        **sac_config
    )
    
    # Callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/eval",
        eval_freq=config['eval_freq'],
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['save_freq'],
        save_path="./models/checkpoints",
        name_prefix=config['model_name']
    )
    callbacks.append(checkpoint_callback)
    
    # Wandb callback (only if wandb enabled)
    if config['use_wandb'] and run:
        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    print(f"🎯 Starting training for {config['total_timesteps']:,} timesteps")
    print("=" * 60)
    
    try:
        # Train the model
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            log_interval=config['log_interval']
        )
        
        # Save final model
        final_model_path = f"models/{config['model_name']}_final"
        model.save(final_model_path)
        print(f"💾 Final model saved to {final_model_path}")
        
        # Upload model to wandb
        wandb.save(f"{final_model_path}.zip")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        interrupt_model_path = f"models/{config['model_name']}_interrupted"
        model.save(interrupt_model_path)
        print(f"💾 Model saved to {interrupt_model_path}")
        
    finally:
        # Cleanup
        train_env.close()
        eval_env.close()
        if config['use_wandb']:
            wandb.finish()
        
    print("✅ Training completed!")

if __name__ == "__main__":
    main()