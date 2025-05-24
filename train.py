import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import quad, nquad
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from typing import List, Tuple, Callable, Dict
import scipy.special as sp
import os
import time
import traceback
from stable_baselines3.common.vec_env import VecNormalize
import torch.nn as nn
# from adaptiveenv import EnhancedAdaptiveIntegrationEnv  # Import the enhanced environment

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func

# Define challenging integration functions
def define_test_functions() -> Dict[str, Tuple[Callable, float, float]]:
    """Define a variety of challenging functions with their integration bounds
    Returns a dictionary mapping function names to (function, lower_bound, upper_bound)"""

    functions = {}

    # 1. Highly oscillatory functions
    functions["sin_high_freq"] = (lambda x: np.sin(50 * x), 0.0, 2.0)
    functions["cos_increasing_freq"] = (lambda x: np.cos(x**2), 0.0, 10.0)
    functions["sin_exp"] = (lambda x: np.sin(np.exp(x)), 0.0, 3.0)

    # 2. Functions with rapid changes
    functions["steep_sigmoid"] = (lambda x: 1 / (1 + np.exp(-100 * (x - 0.5))), 0.0, 1.0)
    functions["runge"] = (lambda x: 1 / (1 + 25 * x**2), -1.0, 1.0)

    # 3. Discontinuous functions
    functions["step"] = (lambda x: 1.0 if x > 0.5 else 0.0, 0.0, 1.0)
    functions["sawtooth"] = (lambda x: x - np.floor(x), 0.0, 5.0)

    # 4. Functions with singularities
    functions["sqrt_singularity"] = (lambda x: 1 / np.sqrt(x), 1e-6, 1.0)  # Singularity at x=0
    functions["log_singularity"] = (lambda x: np.log(x), 1e-6, 2.0)  # Singularity at x=0
    functions["inverse_singularity"] = (lambda x: 1 / (x - 0.5)**2 if abs(x - 0.5) > 1e-6 else 0, 0.0, 1.0)  # Singularity at x=0.5

    # 5. Combined challenging behaviors
    functions["oscillating_with_peaks"] = (lambda x: np.sin(10 * x) + 5 * np.exp(-100 * (x - 0.5)**2), 0.0, 1.0)
    functions["discontinuous_oscillatory"] = (lambda x: np.sin(20 * x) * (1 if x > 0.5 else 0.5), 0.0, 1.0)

    return functions


def define_2d_test_functions() -> Dict[str, Tuple[Callable, float, float, float, float]]:
    """Define comprehensive set of 2D test functions"""
    functions = {}

    # 1. Standard smooth functions
    functions["gaussian_2d"] = (
        lambda x, y: np.exp(-(x*2 + y*2)),
        -3.0, 3.0, -3.0, 3.0
    )
    functions["sinc_2d"] = (
        lambda x, y: np.sinc(x) * np.sinc(y),
        -4.0, 4.0, -4.0, 4.0
    )
    functions["polynomial_2d"] = (
        lambda x, y: x*2 * y3 - x*y + y*2,
        -2.0, 2.0, -2.0, 2.0
    )

    # 2. Highly oscillatory functions
    functions["oscillatory_2d"] = (
        lambda x, y: np.sin(50*x) * np.cos(50*y),
        0.0, 2.0, 0.0, 2.0
    )
    functions["bessel_2d"] = (
        lambda x, y: sp.j0(np.sqrt(x*2 + y*2)),
        -10.0, 10.0, -10.0, 10.0
    )
    functions["frequency_modulated"] = (
        lambda x, y: np.sin(x * (1 + y*2)) * np.cos(y * (1 + x*2)),
        -2.0, 2.0, -2.0, 2.0
    )
    functions["wave_packet"] = (
        lambda x, y: np.exp(-(x*2 + y2)) * np.sin(10(x + y)),
        -3.0, 3.0, -3.0, 3.0
    )

    # 3. Functions with rapid changes
    functions["peaks_2d"] = (
        lambda x, y: 3*(1-x)*2 * np.exp(-x2 - (y+1)*2) -
                    10*(x/5 - x*3 - y5) * np.exp(-x2 - y*2) -
                    1/3 * np.exp(-(x+1)*2 - y*2),
        -3.0, 3.0, -3.0, 3.0
    )
    functions["gaussian_peaks"] = (
        lambda x, y: sum(np.exp(-((x-xi)*2 + (y-yi)*2)/0.1)
                        for xi, yi in [(-1,-1), (1,1), (-1,1), (1,-1)]),
        -2.0, 2.0, -2.0, 2.0
    )

    # 4. Discontinuous functions
    functions["step_2d"] = (
        lambda x, y: 1.0 if x > 0 and y > 0 else 0.0,
        -1.0, 1.0, -1.0, 1.0
    )
    functions["checkerboard"] = (
        lambda x, y: 1.0 if (int(2*x) + int(2*y)) % 2 == 0 else 0.0,
        0.0, 2.0, 0.0, 2.0
    )
    functions["circular_step"] = (
        lambda x, y: 1.0 if x*2 + y*2 < 1 else 0.0,
        -2.0, 2.0, -2.0, 2.0
    )
    functions["sawtooth_2d"] = (
        lambda x, y: (x - np.floor(x)) * (y - np.floor(y)),
        0.0, 3.0, 0.0, 3.0
    )

    # 5. Functions with singularities
    functions["inverse_r"] = (
        lambda x, y: 1.0 / (np.sqrt(x*2 + y*2) + 1e-10),
        -1.0, 1.0, -1.0, 1.0
    )
    functions["log_singularity_2d"] = (
        lambda x, y: np.log(x*2 + y*2 + 1e-10),
        -1.0, 1.0, -1.0, 1.0
    )
    functions["pole_singularity"] = (
        lambda x, y: 1.0 / ((x*2 + y2 - 0.52)*2 + 0.1),
        -1.0, 1.0, -1.0, 1.0
    )

    # 6. Combined challenging behaviors
    functions["oscillating_peaks_2d"] = (
        lambda x, y: np.sin(10*x) * np.cos(10*y) * np.exp(-((x-0.5)*2 + (y-0.5)*2)),
        0.0, 2.0, 0.0, 2.0
    )
    functions["mixed_features"] = (
        lambda x, y: (np.sin(20*x*y) / (1 + x*2 + y*2) +
                     np.exp(-((x-0.5)*2 + (y-0.5)*2) * 10)),
        -2.0, 2.0, -2.0, 2.0
    )
    functions["complex_oscillatory"] = (
        lambda x, y: np.sin(30*x) * np.cos(30*y) + np.exp(-((x-0.5)*2 + (y-0.5)*2) * 5),
        -1.0, 1.0, -1.0, 1.0
    )
    functions["hybrid_singularity"] = (
        lambda x, y: np.sin(10*x*y) / (0.1 + x*2 + y*2),
        -2.0, 2.0, -2.0, 2.0
    )

    return functions


# Function to create a custom environment factory for a specific function
def make_env_factory(func, ax, bx, ay, by, max_intervals=40):
    def _init():
        env = EnhancedAdaptiveIntegrationEnv(
            ax=ax, bx=bx, ay=ay, by=by,
            max_intervals=max_intervals,
            function=func
        )
        return env
    return _init


# Training function
# Modified version of the train_model function to ensure training stops at 2000 steps

def train_model(functions, training_steps=200000, save_dir="models", evaluate=True):
    """Train a model on sequence of 2D functions with enhanced local error focus"""
    os.makedirs(save_dir, exist_ok=True)
    model = None

    # Enhanced training configuration with adjusted parameters for longer training
    schedule = linear_schedule(5e-4, 1e-5)  # Adjusted learning rates for longer training
    early_stop = EarlyStopCallback(
        check_freq=1000,  # Adjusted for longer training
        min_improvement=1e-5,  # More stringent improvement threshold
        min_local_improvement=1e-6,  # More stringent local improvement threshold
        patience=10,  # Increased patience for longer training
        min_episodes=20,  # Increased minimum episodes
        verbose=1
    )

    for i, (func_name, (func, ax, bx, ay, by)) in enumerate(functions.items()):
        print(f"\n{'-'*50}")
        print(f"Training on 2D function: {func_name}")

        # Create environment with enhanced local error handling
        def make_env():
            env = EnhancedAdaptiveIntegrationEnv(
                ax=ax, bx=bx, ay=ay, by=by,
                max_intervals=40,  # Increased max intervals
                function=func
            )
            return env

        # Create vectorized environment with enhanced normalization
        vec_env = SubprocVecEnv([make_env for _ in range(8)])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.995  # Increased gamma for better long-term planning
        )

        if model is None:
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                learning_rate=schedule,
                n_steps=2048,  # Increased for longer training
                batch_size=256,  # Increased batch size
                gamma=0.99,
                tensorboard_log=f"{save_dir}/tensorboard/",
                policy_kwargs={
                    'net_arch': [256, 256, 128],  # Deeper network
                    'log_std_init': -2.0,
                    'ortho_init': True,
                    'activation_fn': nn.ReLU,
                    'use_expln': True,
                    'full_std': True,
                },
                use_sde=True,
                sde_sample_freq=8,
                max_grad_norm=0.5,
                clip_range=0.2,
                clip_range_vf=0.2,
                ent_coef=0.01
            )
        else:
            # Reset normalization stats for new function
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.995
            )
            model.set_env(vec_env)

        try:
            print(f"Training for {training_steps} steps...")
            start_time = time.time()

            # Custom callback for monitoring local errors
            class LocalErrorMonitor(BaseCallback):
                def _init_(self, verbose=0):
                    super()._init_(verbose)
                    self.local_errors = []

                def _on_step(self):
                    if len(self.model.ep_info_buffer) > 0:
                        info = self.model.ep_info_buffer[-1]
                        if 'local_errors' in info:
                            self.local_errors.append(np.mean(info['local_errors']))
                    return True

            local_monitor = LocalErrorMonitor()

            # Important fix: Use the correct callback for stopping after specific steps
            # Modified callback to ensure hard stop at training_steps
            class StrictStepLimitCallback(BaseCallback):
                def _init_(self, total_steps: int, verbose: int = 0):
                    super()._init_(verbose)
                    self.total_steps = total_steps
                    self.training_start = time.time()

                def _on_step(self) -> bool:
                    # Check if we've reached the desired number of steps
                    if self.num_timesteps >= self.total_steps:
                        elapsed = time.time() - self.training_start
                        if self.verbose > 0:
                            print(f"\nReached {self.num_timesteps}/{self.total_steps} steps after {elapsed:.1f}s")
                            print("Stopping training as requested.")
                        # Return False to stop training
                        return False
                    return True

            # Create the step limit callback with the correct max steps
            step_limit = StrictStepLimitCallback(total_steps=training_steps, verbose=1)

            # Use CallbackList to combine callbacks
            from stable_baselines3.common.callbacks import CallbackList
            callbacks = CallbackList([early_stop, local_monitor, step_limit])

            model.learn(
                total_timesteps=training_steps + 100,  # Add a buffer to ensure our callback has control
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=True  # Start counting from 0 for each function
            )

            # Save normalization statistics
            vec_env.save(f"{save_dir}/vec_normalize_{i}_{func_name}.pkl")
            print("\nTraining Summary:")
            # Enhanced training summary
            summary = early_stop.get_training_summary()
            print(f"Best reward: {summary['best_reward']:.6f}")
            print(f"Best local error: {summary['best_local_error']:.6e}")
            print(f"Mean local error (last 10): {np.mean(local_monitor.local_errors[-10:] if local_monitor.local_errors else [0]):.6e}")
            print(f"Episodes completed: {summary['episodes']}")
            print(f"Training time: {time.time() - start_time:.1f}s")

            # Save model with metadata
            model_path = f"{save_dir}/adaptive_integration_2d_{i}_{func_name}"
            model.save(model_path)

            if evaluate:
                mean_reward, std_reward = evaluate_policy(
                    model, vec_env, n_eval_episodes=15,
                    deterministic=True
                )
                print(f"\nEvaluation results:")
                print(f"Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            traceback.print_exc()  # Added to show full error trace
        finally:
            vec_env.close()

    # Save final model
    final_model_path = f"{save_dir}/adaptive_integration_final"
    if model is not None:
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

    return model


# Visualization function to evaluate and visualize results
def evaluate_and_visualize(model_path, functions, max_intervals=20):
    """Evaluate the trained model"""
    model = PPO.load(model_path)
    results = {}

    for func_name, (func, a, b) in functions.items():
        print(f"\nEvaluating on {func_name}...")

        # Create environment using base AdaptiveIntegrationEnv
        env = EnhancedAdaptiveIntegrationEnv(a=a, b=b, max_intervals=max_intervals, function=func)

        # Reset environment
        obs, _ = env.reset()
        total_reward = 0
        # Track progress
        done = False
        total_reward = 0
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        # Store results
        results[func_name] = {
            'true_value': env.true_value,
            'approximation': info['approximation'],
            'error': info['error'],
            'num_intervals': len(env.intervals),
            'num_evaluations': len(env.evals)
        }
        print(f"  True value:     {env.true_value:.8f}")
        print(f"  Approximation:  {info['approximation']:.8f}")
        print(f"  Error:          {info['error']:.8e}")
        print(f"  Intervals used: {len(env.intervals)}")
        print(f"  Evaluations:    {len(env.evals)}")
    return results

def evaluate_and_visualize_2d(model_path, functions, vec_normalize_path=None, max_intervals=30):
    """Evaluate the trained model on 2D functions"""
    model = PPO.load(model_path)
    results = {}

    for func_name, (func, ax, bx, ay, by) in functions.items():
        print(f"\nEvaluating on {func_name}...")

        # Create environment using EnhancedAdaptiveIntegrationEnv
        env = EnhancedAdaptiveIntegrationEnv(
            ax=ax, bx=bx, ay=ay, by=by,
            max_intervals=max_intervals,
            function=func
        )

        # Wrap in VecNormalize if path provided
        if vec_normalize_path:
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False  # Don't update statistics during evaluation
            env.norm_reward = False  # Don't normalize rewards during evaluation

        # Reset environment
        obs, _ = env.reset()
        total_reward = 0
        # Track progress
        done = False
        total_reward = 0
        step_count = 0
        error_history = []
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            error_history.append(info['error'])
            step_count += 1

        # Store detailed results
        results[func_name] = {
            'true_value': env.true_value,
            'approximation': info['approximation'],
            'error': info['error'],
            'relative_error': info['error']/abs(env.true_value),
            'num_regions': len(env.regions),
            'num_evaluations': len(env.evals),
            'total_reward': total_reward,
            'steps': step_count,
            'error_history': error_history,
            'efficiency': info['efficiency']
        }
        print(f"\nResults for {func_name}:")
        print(f"  True value:      {env.true_value:.10e}")
        print(f"  Approximation:   {info['approximation']:.10e}")
        print(f"  Absolute Error:  {info['error']:.10e}")
        print(f"  Relative Error:  {info['error']/abs(env.true_value):.10e}")
        print(f"  Regions used:    {len(env.regions)}")
        print(f"  Evaluations:     {len(env.evals)}")
        print(f"  Total reward:    {total_reward:.4f}")
        print(f"  Steps taken:     {step_count}")
        print(f"  Efficiency:      {info['efficiency']:.4e}")

        # Visualize solution
        env.visualize_solution(num_points=100)
    return results

class EarlyStopCallback(BaseCallback):
    """
    Enhanced callback for early stopping with improved monitoring and stopping criteria.
    Tracks both global and local error improvements.
    """
    def _init_(self, check_freq: int = 5000,
                 min_improvement: float = 1e-6,
                 min_local_improvement: float = 1e-7,
                 patience: int = 5,
                 min_episodes: int = 20,
                 verbose: int = 1):
        super()._init_(verbose)
        self.check_freq = check_freq
        self.min_improvement = min_improvement
        self.min_local_improvement = min_local_improvement
        self.patience = patience
        self.min_episodes = min_episodes
        self.verbose = verbose

        # Initialize tracking variables
        self.best_mean_reward = -float('inf')
        self.best_local_error = float('inf')
        self.no_improvement_count = 0
        self.episode_count = 0
        self.reward_history = []
        self.error_history = []
        self.local_error_history = []
        self.training_start = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get current metrics
            mean_reward = self.model.logger.name_to_value.get('rollout/ep_rew_mean', None)
            ep_count = self.model.logger.name_to_value.get('time/episodes', 0)
            mean_error = self.model.logger.name_to_value.get('rollout/ep_error_mean', float('inf'))
            local_error = self.model.logger.name_to_value.get('rollout/local_error_mean', float('inf'))

            if mean_reward is not None:
                self.reward_history.append(mean_reward)
                self.error_history.append(mean_error)
                self.local_error_history.append(local_error)
                self.episode_count = ep_count

                # Calculate improvements
                reward_improvement = mean_reward - self.best_mean_reward
                local_error_improvement = self.best_local_error - local_error

                # Check for significant improvement in either metrics
                if (reward_improvement > self.min_improvement or
                    local_error_improvement > self.min_local_improvement):
                    self.best_mean_reward = max(mean_reward, self.best_mean_reward)
                    self.best_local_error = min(local_error, self.best_local_error)
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        elapsed_time = time.time() - self.training_start
                        print(f"\nImprovement at episode {ep_count} ({elapsed_time:.1f}s):")
                        print(f"  Mean reward:     {mean_reward:.6f}")
                        print(f"  Local error:     {local_error:.6e}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"\nNo significant improvement: {self.no_improvement_count}/{self.patience}")
                        print(f"  Current reward:  {mean_reward:.6f}")
                        print(f"  Current local error: {local_error:.6e}")

                # Check stopping conditions
                if self.episode_count < self.min_episodes:
                    return True

                # Stop if no improvement for too long
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print("\nEarly stopping triggered:")
                        print(f"  Episodes:        {self.episode_count}")
                        print(f"  Final reward:    {mean_reward:.6f}")
                        print(f"  Best reward:     {self.best_mean_reward:.6f}")
                        print(f"  Final local error: {local_error:.6e}")
                        print(f"  Training time:   {time.time() - self.training_start:.1f}s")
                    return False

                # Check for performance degradation
                if len(self.reward_history) > 5:
                    recent_reward_mean = np.mean(self.reward_history[-5:])
                    recent_error_mean = np.mean(self.local_error_history[-5:])
                    if (recent_reward_mean < self.best_mean_reward * 0.5 or
                        recent_error_mean > self.best_local_error * 2.0):
                        if self.verbose > 0:
                            print("\nStopping due to performance degradation:")
                            print(f"  Recent reward mean: {recent_reward_mean:.6f}")
                            print(f"  Recent error mean:  {recent_error_mean:.6e}")
                        return False
        return True

    def get_training_summary(self) -> Dict:
        """Return summary of training progress"""
        return {
            'best_reward': self.best_mean_reward,
            'best_local_error': self.best_local_error,
            'episodes': self.episode_count,
            'training_time': time.time() - self.training_start,
            'reward_history': self.reward_history,
            'error_history': self.error_history,
            'local_error_history': self.local_error_history
        }

class StepLimitCallback(BaseCallback):
    """Callback to force training to stop after exactly 200000 steps"""
    def _init_(self, total_steps: int = None, verbose: int = 0):
        super()._init_(verbose)
        # Hardcode to exactly 200000 steps regardless of passed parameter
        self.total_steps = 200000  # Changed from 2000 to 200000
        self.step_count = 0
        self.verbose = verbose
        self.start_time = time.time()
        print(f"StepLimitCallback initialized with hardcoded limit of {self.total_steps} steps")

    def _on_step(self) -> bool:
        self.step_count += 1
        # Print progress every 20000 steps (10% of 200000)
        if self.verbose > 0 and self.step_count % 20000 == 0:
            elapsed = time.time() - self.start_time
            print(f"Training progress: {self.step_count}/{self.total_steps} steps ({self.step_count/self.total_steps*100:.1f}%) - {elapsed:.1f}s elapsed")

        # Force stop at exactly 200000 steps
        if self.step_count >= self.total_steps:
            if self.verbose > 0:
                print(f"\n>>> STOPPING: Reached exactly {self.total_steps} steps <<<")
                print(f"Total training time: {time.time() - self.start_time:.1f} seconds")

            # This will definitely stop the training
            self.training_env.reset()  # Reset environment to avoid potential errors
            return False  # Return False to stop training

        return True

    def on_training_end(self) -> None:
        print(f"Training ended at exactly {self.step_count} steps")

if _name_ == "_main_":
    # Get all 2D test functions
    all_functions = define_2d_test_functions()

    # Training functions with progressive difficulty
    training_functions = {
        k: all_functions[k] for k in [
            # Start with simpler functions
            "gaussian_2d"
            , "sinc_2d", "polynomial_2d",
            # Progress to oscillatory functions
            "wave_packet", "oscillatory_2d", "bessel_2d",
            # Add rapid changes
            "peaks_2d", "gaussian_peaks",
            # Include discontinuities
            "step_2d", "circular_step",
            # Add singularities
            "inverse_r", "log_singularity_2d",
            # Finish with combined challenges
            "oscillating_peaks_2d", "complex_oscillatory",
            "mixed_features", "hybrid_singularity"
        ]
    }

    # Train with increased steps
    model = train_model(
        training_functions,
        training_steps=200000,  # Changed from 2000 to 200000
        save_dir="adaptive_integration_2d_models"
    )

    # Test functions for evaluation
    test_functions = {
        k: all_functions[k] for k in [
            "gaussian_2d"         # Smooth function
            "bessel_2d",            # Oscillatory
            "gaussian_peaks",        # Multiple peaks
            "circular_step",         # Discontinuous
            "pole_singularity",     # Singularity
            "complex_oscillatory"    # Combined features
        ]
    }

    # Evaluate and visualize results
    results = evaluate_and_visualize_2d(
        "adaptive_integration_2d_models/adaptive_integration_final",
        test_functions
    )

    # Print summary of all results
    print("\n" + "="*60)
    print("SUMMARY OF 2D INTEGRATION RESULTS")
    print("="*60)
    avg_rel_error = np.mean([r['relative_error'] for r in results.values()])
    avg_efficiency = np.mean([r['efficiency'] for r in results.values()])
    total_evals = sum(r['num_evaluations'] for r in results.values())

    for func_name, result in results.items():
        print(f"\n{func_name}:")
        print(f"  Relative error: {result['relative_error']:.2e}")
        print(f"  Efficiency:     {result['efficiency']:.2e}")
        print(f"  Regions/Evals:  {result['num_regions']}/{result['num_evaluations']}")

    print("\nAggregate Performance:")
    print(f"Average Relative Error: {avg_rel_error:.2e}")
    print(f"Average Efficiency:     {avg_efficiency:.2e}")
    print(f"Total Evaluations:      {total_evals}")