#!/usr/bin/env python3
"""
Main training script for RL search optimization

This script trains a PPO agent to learn intelligent search behavior
when the target object is lost from view.
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl import SearchRLEnv, ComprehensiveMetricsTracker
from src.rl.search_agent import SearchPPOAgent, create_vectorized_env
from src.so_arm_gym_env import SO101CameraTrackingEnv


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PPO agent for SO101 search behavior')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps (default: 500000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='PPO learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Mini-batch size (default: 64)')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Steps per rollout (default: 2048)')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Optimization epochs per rollout (default: 10)')
    
    # Environment parameters
    parser.add_argument('--max-search-steps', type=int, default=300,
                       help='Maximum steps per search episode (default: 300)')
    parser.add_argument('--n-envs', type=int, default=1,
                       help='Number of parallel environments (default: 1)')
    parser.add_argument('--static-target', action='store_true', default=True,
                       help='Use static target position (default: True)')
    parser.add_argument('--target-position', type=float, nargs=3, default=None,
                       help='Custom target position [x, y, z] (default: auto-calculated)')
    
    # Evaluation and saving
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluation frequency (default: 10000)')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--save-freq', type=int, default=50000,
                       help='Model save frequency (default: 50000)')
    parser.add_argument('--checkpoint-freq', type=int, default=100000,
                       help='Checkpoint save frequency (default: 100000)')
    
    # Paths and logging
    parser.add_argument('--log-dir', type=str, default='search_rl_logs',
                       help='Directory for training logs (default: search_rl_logs)')
    parser.add_argument('--model-path', type=str, default='models/search_ppo_model',
                       help='Path to save trained model (default: models/search_ppo_model)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to pre-trained model to continue training')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during training (slower)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning (progressive difficulty)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'],
                       help='Device for training: cpu (recommended for MLP), cuda, or auto (default: cpu)')
    
    return parser.parse_args()


def create_search_environment(max_search_steps=300, render_mode="rgb_array", static_target=True, target_position=None):
    """Create search RL environment"""
    print("🏗️  Creating search environment...")
    
    # Create base SO101 environment
    base_env = SO101CameraTrackingEnv(render_mode=render_mode)
    
    # Wrap in search RL environment
    search_env = SearchRLEnv(
        base_env=base_env,
        max_search_steps=max_search_steps,
        static_target=static_target,
        target_position=target_position
    )
    
    print("✅ Search environment created")
    if static_target:
        print("   📍 Using static target position for simplified training")
    else:
        print("   🎲 Using dynamic target placement")
    
    return search_env


def setup_curriculum_learning(agent, total_timesteps):
    """
    Setup curriculum learning stages
    
    Progressively increases difficulty during training
    """
    print("📚 Setting up curriculum learning...")
    
    curriculum_stages = [
        {
            'name': 'Stage 1: Basic Search',
            'timesteps': int(total_timesteps * 0.3),
            'description': 'Simple scenarios, targets in easy positions',
            'config': {
                'max_search_steps': 200,
                'target_distance_range': (0.4, 0.6),
                'difficulty_multiplier': 0.5
            }
        },
        {
            'name': 'Stage 2: Intermediate Search', 
            'timesteps': int(total_timesteps * 0.4),
            'description': 'Medium difficulty, varied target positions',
            'config': {
                'max_search_steps': 250,
                'target_distance_range': (0.3, 0.7),
                'difficulty_multiplier': 0.75
            }
        },
        {
            'name': 'Stage 3: Advanced Search',
            'timesteps': int(total_timesteps * 0.3),
            'description': 'Full difficulty, challenging scenarios',
            'config': {
                'max_search_steps': 300,
                'target_distance_range': (0.3, 0.8),
                'difficulty_multiplier': 1.0
            }
        }
    ]
    
    for i, stage in enumerate(curriculum_stages):
        print(f"   {stage['name']}: {stage['timesteps']:,} timesteps")
        print(f"     {stage['description']}")
    
    return curriculum_stages


def train_with_curriculum(agent, curriculum_stages, args):
    """Train agent using curriculum learning"""
    print("🎓 Starting curriculum-based training...")
    
    total_trained = 0
    
    for stage_idx, stage in enumerate(curriculum_stages):
        print(f"\n📖 {stage['name']} ({stage['timesteps']:,} timesteps)")
        print(f"   {stage['description']}")
        
        # Update environment configuration if needed
        # (This would require modifying the environment to support dynamic config)
        
        # Train for this stage
        stage_results = agent.train(
            total_timesteps=stage['timesteps'],
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_freq=args.save_freq,
            checkpoint_freq=args.checkpoint_freq
        )
        
        total_trained += stage['timesteps']
        progress = total_trained / sum(s['timesteps'] for s in curriculum_stages)
        
        print(f"✅ {stage['name']} completed!")
        print(f"   Overall progress: {progress:.1%}")
        
        # Save stage checkpoint
        stage_model_path = f"{args.model_path}_stage_{stage_idx+1}"
        agent.save_model(stage_model_path)
        print(f"   Stage model saved: {stage_model_path}")
    
    print("\n🎓 Curriculum training completed!")


def main():
    """Main training function"""
    args = parse_arguments()
    
    print("🚀 SO101 Search RL Training")
    print("=" * 50)
    print(f"Training PPO agent for intelligent search behavior")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Log directory: {args.log_dir}")
    print(f"Model save path: {args.model_path}")
    
    if args.static_target:
        if args.target_position:
            print(f"Target position: [{args.target_position[0]:.3f}, {args.target_position[1]:.3f}, {args.target_position[2]:.3f}] (custom)")
        else:
            print(f"Target position: Static (75° left, 0.75m distance)")
    else:
        print(f"Target position: Dynamic (random placement)")
    
    if args.curriculum:
        print(f"Using curriculum learning: ✅")
    
    print("=" * 50)
    
    try:
        # Create environment(s)
        render_mode = "human" if args.render else "rgb_array"
        
        if args.n_envs > 1:
            print(f"🌐 Creating {args.n_envs} parallel environments...")
            
            env_kwargs = {
                'max_search_steps': args.max_search_steps,
                'static_target': args.static_target,
                'target_position': args.target_position
            }
            
            env = create_vectorized_env(
                env_class=lambda **kwargs: create_search_environment(
                    max_search_steps=kwargs['max_search_steps'],
                    render_mode="rgb_array",  # No rendering for parallel envs
                    static_target=kwargs['static_target'],
                    target_position=kwargs['target_position']
                ),
                env_kwargs=env_kwargs,
                n_envs=args.n_envs,
                seed=args.seed
            )
            
        else:
            print("🌐 Creating single environment...")
            env = create_search_environment(
                max_search_steps=args.max_search_steps,
                render_mode=render_mode,
                static_target=args.static_target,
                target_position=args.target_position
            )
        
        # Create PPO agent
        print("🤖 Creating PPO agent...")
        if args.device == 'cpu':
            print("   💻 Using CPU (recommended for MLP policy)")
        elif args.device == 'cuda':
            print("   🚀 Using GPU (may be slower for MLP policy)")
        
        agent = SearchPPOAgent(
            env=env,
            log_dir=args.log_dir,
            model_save_path=args.model_path,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=args.device,
            verbose=args.verbose
        )
        
        # Load pre-trained model if specified
        if args.load_model:
            print(f"📁 Loading pre-trained model: {args.load_model}")
            agent.load_model(args.load_model)
        else:
            # Create new model
            agent.create_model(seed=args.seed)
        
        # Training
        start_time = time.time()
        
        if args.curriculum:
            # Curriculum learning
            curriculum_stages = setup_curriculum_learning(agent, args.timesteps)
            train_with_curriculum(agent, curriculum_stages, args)
        else:
            # Standard training
            print("🏋️  Starting standard PPO training...")
            training_results = agent.train(
                total_timesteps=args.timesteps,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                save_freq=args.save_freq,
                checkpoint_freq=args.checkpoint_freq,
                seed=args.seed
            )
        
        training_time = time.time() - start_time
        
        # Final evaluation (disabled - run separately)
        # print("\n📊 Running final evaluation...")
        # eval_results = agent.evaluate(
        #     n_episodes=50,
        #     deterministic=True,
        #     render=False
        # )
        
        # Print final results
        print("\n" + "=" * 50)
        print("🏁 TRAINING COMPLETED!")
        print(f"⏱️  Total training time: {training_time/3600:.1f} hours")
        print(f"💾 Model saved: {args.model_path}")
        print(f"📊 Run evaluation separately with:")
        print(f"   python scripts/evaluate_search.py --model-path {args.model_path}")
        print("=" * 50)
        
        # Save training completion info
        training_log_path = os.path.join(args.log_dir, "training_completed.txt")
        with open(training_log_path, 'w') as f:
            f.write("Training Completion Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Total timesteps: {args.timesteps:,}\n")
            f.write(f"Training time: {training_time/3600:.2f} hours\n")
            f.write(f"Model saved: {args.model_path}\n")
            f.write(f"Evaluation command: python scripts/evaluate_search.py --model-path {args.model_path}\n")
        
        print(f"📄 Training summary saved: {training_log_path}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'env' in locals():
                env.close()
        except:
            pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
