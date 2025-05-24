import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import quad, nquad
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn
from typing import Callable, Dict, List, Tuple, Optional, Union
import time
from scipy.stats import skew, kurtosis
from scipy.interpolate import UnivariateSpline


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:

    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end)
        :param progress_remaining:
        :return: current learning rate
        """
        return final_value + progress_remaining * (initial_value - final_value)
    return func


class EarlyStopCallback(BaseCallback):
    """
    Enhanced callback for early stopping with improved monitoring and stopping criteria.
    Tracks both global and local error improvements.
    """
    def _init_(self,
                 check_freq: int = 5000,
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

                # Check for significant improvement in either metric
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


class Particle:
    """Represents a particle for PFEM integration"""
    def _init_(self, x: float, y: float, weight: float = 1.0):
        self.x = x
        self.y = y
        self.weight = weight
        self.value = None
        self.neighbors = []
        self.error_estimate = 0.0

class PFEMIntegrator:
    """Handles PFEM-based integration"""
    def _init_(self, function, min_particles=20, max_particles=100):
        self.function = function
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.particles = []

    def initialize_particles(self, x0, x1, y0, y1, n_initial=20):
        """Initialize particles in the region with jittered grid distribution"""
        nx = ny = int(np.sqrt(n_initial))
        self.particles = []

        for i in range(nx):
            for j in range(ny):
                # Add jitter to avoid regular grid artifacts
                jitter_x = np.random.uniform(-0.1, 0.1) * (x1 - x0) / nx
                jitter_y = np.random.uniform(-0.1, 0.1) * (y1 - y0) / ny

                x = x0 + (i + 0.5) * (x1 - x0) / nx + jitter_x
                y = y0 + (j + 0.5) * (y1 - y0) / ny + jitter_y

                self.particles.append(Particle(x, y))

    def update_particle_values(self, eval_cache):
        """Update function values at particle locations using cache"""
        for p in self.particles:
            if (p.x, p.y) in eval_cache:
                p.value = eval_cache[(p.x, p.y)]
            else:
                p.value = self.function(p.x, p.y)
                eval_cache[(p.x, p.y)] = p.value

    def find_neighbors(self, max_dist):
        """Find neighbors for each particle within max_dist"""
        for p1 in self.particles:
            p1.neighbors = []
            for p2 in self.particles:
                if p1 != p2:
                    dist = np.sqrt((p1.x - p2.x)*2 + (p1.y - p2.y)*2)
                    if dist < max_dist:
                        p1.neighbors.append(p2)

    def estimate_local_error(self):
        """Estimate error for each particle based on neighbor value differences"""
        for p in self.particles:
            if p.neighbors:
                values = [n.value for n in p.neighbors]
                p.error_estimate = np.std(values)

    def adapt_particles(self, x0, x1, y0, y1):
        """Adapt particle distribution based on error estimates"""
        # Remove particles with low error estimates
        self.particles = [p for p in self.particles if p.error_estimate > np.median([p.error_estimate for p in self.particles])]

        # Add particles in high error regions
        new_particles = []
        for p in self.particles:
            if p.error_estimate > np.percentile([p.error_estimate for p in self.particles], 75):
                # Add new particles around high error particle
                for _ in range(2):
                    radius = 0.1 * min(x1 - x0, y1 - y0)
                    angle = np.random.uniform(0, 2 * np.pi)
                    new_x = np.clip(p.x + radius * np.cos(angle), x0, x1)
                    new_y = np.clip(p.y + radius * np.sin(angle), y0, y1)
                    new_particles.append(Particle(new_x, new_y))

        self.particles.extend(new_particles)

        # Limit total number of particles
        if len(self.particles) > self.max_particles:
            self.particles = sorted(self.particles, key=lambda p: p.error_estimate, reverse=True)[:self.max_particles]

    def integrate(self, x0, x1, y0, y1, eval_cache):
        """Perform PFEM integration"""
        area = (x1 - x0) * (y1 - y0)
        self.update_particle_values(eval_cache)
        max_dist = 0.2 * min(x1 - x0, y1 - y0)
        self.find_neighbors(max_dist)
        self.estimate_local_error()

        # Weighted sum of particle values
        total_weight = sum(p.weight for p in self.particles)
        integral = area * sum(p.value * p.weight for p in self.particles) / total_weight

        # Error estimate based on particle distribution
        error = np.mean([p.error_estimate for p in self.particles])

        return integral, error


class EnhancedAdaptiveIntegrationEnv(gym.Env):
    """
    Advanced environment for adaptive numerical integration using reinforcement learning.
    Includes normalization, enhanced error estimation, adaptive splitting, and advanced rewards.
    """
    def _init_(self,
                 ax: float = 0.0,
                 bx: float = 1.0,
                 ay: float = 0.0,
                 by: float = 1.0,
                 max_intervals: int = 20,
                 function: Callable[[float, float], float] = lambda x, y: np.sin(x) * np.cos(y),
                 function_params: Optional[Dict] = None):
        """
        Initialize the advanced adaptive integration environment for 2D.

        Args:
            ax, bx (float): Lower and upper bounds of x domain
            ay, by (float): Lower and upper bounds of y domain
            max_intervals (int): Maximum number of rectangular regions
            function (callable): 2D function to integrate
            function_params (dict, optional): Parameters of the function
        """
        super()._init_()

        # Domain boundaries
        self.ax, self.bx = ax, bx
        self.ay, self.by = ay, by
        self.x_width = bx - ax
        self.y_width = by - ay
        self.max_intervals = max_intervals

        # Function and parameters
        self.f = function
        self.function_params = function_params if function_params is not None else {}
        self.param_values = list(self.function_params.values()) if self.function_params else []

        # Calculate true value using high-precision integration
        self.true_value, _ = nquad(self.f, [[ax, bx], [ay, by]])

        # Normalization factors
        self.value_scale = 1.0

        # Action space: [region_idx, split_ratio, dimension, strategy]
        # dimension: 0 for x-split, 1 for y-split
        self.action_space = spaces.Box(
            low=np.array([0, 0.1, 0, 0]),
            high=np.array([1, 0.9, 1, 1]),
            shape=(4,), dtype=np.float32
        )

        # Enhanced observation space for 2D
        # Features per region: 15 base + 1 Richardson
        # Global stats: 5 base + function parameters
        n_params = len(self.param_values)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(max_intervals * 16 + 5 + n_params,),
            dtype=np.float32
        )

        # Initialize storage for rectangular regions
        self.regions = []  # List of (x0, x1, y0, y1) tuples
        self.evals = {}
        self.center_cache = {}
        self.region_history = []

        # Add integration method parameters
        self.mc_samples = 1000  # Base number of Monte Carlo samples
        self.method_history = []  # Track which method was used for each region

        # Initialize PFEM integrator
        self.pfem_integrator = PFEMIntegrator(function)
        self.use_pfem = False

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        # Initialize with single rectangle covering whole domain
        self.regions = [(self.ax, self.bx, self.ay, self.by)]
        self.evals = {}
        self.center_cache = {}
        self.steps = 0
        self.region_history = [(self.ax, self.bx, self.ay, self.by)]

        # Sample points for value scale calculation
        x_points = np.linspace(self.ax, self.bx, 5)
        y_points = np.linspace(self.ay, self.by, 5)
        function_values = []
        for x in x_points:
            for y in y_points:
                val = self.f(x, y)
                self.evals[(x, y)] = val
                function_values.append(val)

        value_range = max(abs(np.max(function_values) - np.min(function_values)), 1e-10)
        self.value_scale = max(1.0, value_range)

        obs = self._get_observation()
        return obs, {}

    def _eval(self, x, y):
        """
        Evaluate function with caching.

        Args:
            x, y (float): Points to evaluate

        Returns:
            float: Function value at (x, y)
        """
        if (x, y) not in self.evals:
            self.evals[(x, y)] = self.f(x, y)
        return self.evals[(x, y)]

    def _gauss_legendre_2d(self, x0, x1, y0, y1):
        """
        2D Gauss-Legendre quadrature.

        Args:
            x0, x1, y0, y1 (float): Region bounds

        Returns:
            float: Integral approximation
        """
        # 5-point GL weights and points (same as before)
        weights = np.array([0.236926885056189, 0.478628670499366, 0.568888888888889,
                          0.478628670499366, 0.236926885056189])
        points = np.array([-0.906179845938664, -0.538469310105683, 0.0,
                          0.538469310105683, 0.906179845938664])

        # Transform points to intervals
        x_points = 0.5 * (x1 - x0) * points + 0.5 * (x1 + x0)
        y_points = 0.5 * (y1 - y0) * points + 0.5 * (y1 + y0)

        # 2D integration
        result = 0.0
        for i, wx in enumerate(weights):
            for j, wy in enumerate(weights):
                if (x_points[i], y_points[j]) not in self.evals:
                    self.evals[(x_points[i], y_points[j])] = self.f(x_points[i], y_points[j])
                result += wx * wy * self.evals[(x_points[i], y_points[j])]

        return result * 0.25 * (x1 - x0) * (y1 - y0)

    def _richardson_extrapolation(self, x0, x1, y0, y1):
        """
        Estimate error using Richardson extrapolation.

        Args:
            x0, x1, y0, y1 (float): Region bounds

        Returns:
            tuple: (error_estimate, improved_estimate)
        """
        # Calculate first approximation (coarse)
        I1 = self._gauss_legendre_2d(x0, x1, y0, y1)

        # Calculate second approximation (finer, splitting region)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        I2 = (self._gauss_legendre_2d(x0, xm, y0, ym) +
              self._gauss_legendre_2d(xm, x1, y0, ym) +
              self._gauss_legendre_2d(x0, xm, ym, y1) +
              self._gauss_legendre_2d(xm, x1, ym, y1))

        # Richardson extrapolation formula for error estimation
        # For 5-point Gauss-Legendre, error should decrease as O(h^10)
        k = 10  # Order of convergence
        error_est = abs(I2 - I1) / (2**k - 1)

        # Also return improved estimate
        improved_est = I2 + (I2 - I1) / (2**k - 1)

        return error_est, improved_est

    def _analyze_function_behavior(self, x0, x1, y0, y1, n_samples=100):
        """Analyze function behavior in a region to determine best integration method"""
        x_samples = np.random.uniform(x0, x1, n_samples)
        y_samples = np.random.uniform(y0, y1, n_samples)
        values = np.array([self.f(x, y) for x, y in zip(x_samples, y_samples)])

        # Calculate statistical measures
        oscillation = np.std(np.diff(values))
        smoothness = np.mean(np.abs(np.diff(values, 2)))
        value_skew = skew(values)
        value_kurt = kurtosis(values)

        return {
            'oscillation': oscillation,
            'smoothness': smoothness,
            'skewness': value_skew,
            'kurtosis': value_kurt,
            'mean': np.mean(values),
            'std': np.std(values)
        }

    def _monte_carlo_integrate(self, x0, x1, y0, y1, n_samples=None):
        """Monte Carlo integration with importance sampling"""
        if n_samples is None:
            n_samples = self.mc_samples

        # Generate samples with importance sampling near high-gradient regions
        behavior = self._analyze_function_behavior(x0, x1, y0, y1)
        if behavior['oscillation'] > 1.0 or behavior['kurtosis'] > 3.0:
            # Use more samples for challenging regions
            n_samples *= 2

        x_samples = np.random.uniform(x0, x1, n_samples)
        y_samples = np.random.uniform(y0, y1, n_samples)

        values = np.array([self._eval(x, y) for x, y in zip(x_samples, y_samples)])
        area = (x1 - x0) * (y1 - y0)
        integral = area * np.mean(values)
        error_est = area * np.std(values) / np.sqrt(n_samples)

        return integral, error_est

    def _simpson_2d(self, x0, x1, y0, y1, level=0, tol=1e-6, max_level=10):
        """Adaptive Simpson's rule for 2D integration"""
        # Function to compute 2D Simpson's rule on a single rectangle
        def single_simpson(x0, x1, y0, y1):
            hx = (x1 - x0) / 2
            hy = (y1 - y0) / 2
            
            # Evaluate at corners and midpoints
            f00 = self._eval(x0, y0)
            f10 = self._eval(x1, y0)
            f01 = self._eval(x0, y1)
            f11 = self._eval(x1, y1)
            
            # Midpoints
            f20 = self._eval(x0 + hx, y0)
            f21 = self._eval(x0 + hx, y1)
            f02 = self._eval(x0, y0 + hy)
            f12 = self._eval(x1, y0 + hy)
            
            # Center point
            f22 = self._eval(x0 + hx, y0 + hy)
            
            # Simpson's rule
            area = (x1 - x0) * (y1 - y0)
            return area * (f00 + f10 + f01 + f11 + 4*(f20 + f21 + f02 + f12) + 16*f22) / 36

        # Base case: compute single rectangle
        single = single_simpson(x0, x1, y0, y1)
        
        if level >= max_level:
            return single
        
        # Split into four sub-rectangles
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2
        
        # Recursive calls on sub-rectangles
        s1 = self._simpson_2d(x0, xm, y0, ym, level+1, tol/4)
        s2 = self._simpson_2d(xm, x1, y0, ym, level+1, tol/4)
        s3 = self._simpson_2d(x0, xm, ym, y1, level+1, tol/4)
        s4 = self._simpson_2d(xm, x1, ym, y1, level+1, tol/4)
        
        composite = s1 + s2 + s3 + s4
        
        # Error estimation
        error = abs(composite - single)
        if error < tol:
            return composite
        return composite

    def _is_polynomial_like(self, x0, x1, y0, y1):
        """Check if function behaves like a polynomial in the region"""
        # Sample points for polynomial behavior analysis
        nx = ny = 7
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self._eval(xi, yi) for xi in x] for yi in y])
        
        # Check smoothness using finite differences
        dx2 = np.diff(Z, n=2, axis=0)
        dy2 = np.diff(Z, n=2, axis=1)
        
        # If second derivatives are nearly constant, likely polynomial
        dx2_variation = np.std(dx2) / (np.mean(abs(dx2)) + 1e-10)
        dy2_variation = np.std(dy2) / (np.mean(abs(dy2)) + 1e-10)
        
        return dx2_variation < 0.1 and dy2_variation < 0.1

    def _choose_integration_method(self, x0, x1, y0, y1):
        """Choose between Monte Carlo, Gaussian quadrature, and Simpson's rule"""
        behavior = self._analyze_function_behavior(x0, x1, y0, y1)
        
        # Check for polynomial-like behavior first
        if self._is_polynomial_like(x0, x1, y0, y1):
            return 'simpson'
            
        # Original criteria for other methods
        use_monte_carlo = (
            behavior['oscillation'] > 2.0 or
            behavior['kurtosis'] > 5.0 or
            abs(behavior['skewness']) > 2.0 or
            behavior['smoothness'] > 1.0
        )
        
        return 'monte_carlo' if use_monte_carlo else 'gaussian'

    def _should_use_pfem(self, x0, x1, y0, y1):
        """Determine if PFEM should be used for this region"""
        try:
            # Sample points to analyze function behavior
            behavior = self._analyze_function_behavior(x0, x1, y0, y1)
            curvature = self._analyze_curvature(x0, x1, y0, y1)

            # Use PFEM if function is highly oscillatory or has sharp gradients
            return (behavior['oscillation'] > 2.0 or
                    behavior['kurtosis'] > 5.0 or
                    curvature['gradient_mag'] > 10.0)
        except:
            return False

    def _adaptive_integrate(self, x0, x1, y0, y1):
        """Adaptively choose and apply integration method"""
        if self._should_use_pfem(x0, x1, y0, y1):
            self.use_pfem = True
            self.pfem_integrator.initialize_particles(x0, x1, y0, y1)
            integral, error = self.pfem_integrator.integrate(x0, x1, y0, y1, self.evals)
            self.pfem_integrator.adapt_particles(x0, x1, y0, y1)
            self.method_history.append('pfem')
        else:
            method = self._choose_integration_method(x0, x1, y0, y1)
            self.method_history.append(method)

            if method == 'simpson':
                integral = self._simpson_2d(x0, x1, y0, y1)
                # Error estimate for Simpson's rule
                coarse = self._simpson_2d(x0, x1, y0, y1, max_level=2)
                error = abs(integral - coarse)
            elif method == 'monte_carlo':
                integral, error = self._monte_carlo_integrate(x0, x1, y0, y1)
            else:
                integral = self._gauss_legendre_2d(x0, x1, y0, y1)
                _, error = self._richardson_extrapolation(x0, x1, y0, y1)

        return integral, error

    def _get_region_features(self, x0, x1, y0, y1):
        """
        Extract features from a 2D region.

        Args:
            x0, x1, y0, y1 (float): Region bounds

        Returns:
            ndarray: Feature vector
        """
        x_width = x1 - x0
        y_width = y1 - y0
        area = x_width * y_width

        # Basic evaluations
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2

        if (center_x, center_y) not in self.center_cache:
            self.center_cache[(center_x, center_y)] = self.f(center_x, center_y)
        f_center = self.center_cache[(center_x, center_y)]

        # Integration estimates
        gauss_integral = self._gauss_legendre_2d(x0, x1, y0, y1)

        # Error estimates for both dimensions
        x_variation = self._estimate_variation(x0, x1, center_y)
        y_variation = self._estimate_variation(y0, y1, center_x, is_y_dim=True)

        # Combine features
        features = np.array([
            x0, x1, y0, y1,        # Region boundaries
            x_width, y_width,      # Dimensions
            area,                  # Area
            f_center,             # Center value
            gauss_integral,       # Integration estimate
            x_variation,          # X-direction variation
            y_variation,          # Y-direction variation
            max(x_variation, y_variation),  # Max variation
            min(x_variation, y_variation),  # Min variation
            x_variation/y_variation if y_variation > 1e-10 else 1.0,  # Variation ratio
            self._estimate_total_error(x0, x1, y0, y1)  # Total error estimate
        ], dtype=np.float32)

        # Add integration method results
        integral, error = self._adaptive_integrate(x0, x1, y0, y1)
        behavior = self._analyze_function_behavior(x0, x1, y0, y1)

        # Enhanced feature vector
        features = np.array([
            x0, x1, y0, y1,        # Region boundaries
            x_width, y_width,      # Dimensions
            area,                  # Area
            integral,              # Adaptive integration result
            error,                 # Error estimate
            behavior['oscillation'],
            behavior['smoothness'],
            behavior['skewness'],
            behavior['kurtosis'],
            1.0 if self.method_history[-1] == 'monte_carlo' else 0.0,  # Method indicator
            error / (abs(integral) + 1e-10)  # Relative error
        ], dtype=np.float32)

        return features

    def _estimate_variation(self, a, b, fixed_coord, is_y_dim=False):
        """
        Estimate variation along one dimension.

        Args:
            a, b (float): Interval bounds
            fixed_coord (float): Fixed coordinate
            is_y_dim (bool): Whether the dimension is y

        Returns:
            float: Variation estimate
        """
        points = np.linspace(a, b, 5)
        values = []
        for p in points:
            coord = (fixed_coord, p) if is_y_dim else (p, fixed_coord)
            if coord not in self.evals:
                self.evals[coord] = self.f(*coord)
            values.append(self.evals[coord])
        return max(abs(np.diff(values))) / (b - a)

    def _estimate_total_error(self, x0, x1, y0, y1):
        """
        Estimate total error for a region using multiple refinements.

        Args:
            x0, x1, y0, y1 (float): Region bounds

        Returns:
            float: Total error estimate
        """
        coarse = self._gauss_legendre_2d(x0, x1, y0, y1)

        # Split region into four and compare
        xm, ym = (x0 + x1)/2, (y0 + y1)/2
        fine = (self._gauss_legendre_2d(x0, xm, y0, ym) +
                self._gauss_legendre_2d(xm, x1, y0, ym) +
                self._gauss_legendre_2d(x0, xm, ym, y1) +
                self._gauss_legendre_2d(xm, x1, ym, y1))

        return abs(fine - coarse)

    def _get_observation(self):
        """
        Create normalized observation vector from current state.

        Returns:
            ndarray: Normalized observation vector
        """
        features = []

        # First pass: collect all raw features and calculate total error
        raw_features = []
        total_error = 0
        max_error = 0

        for x0, x1, y0, y1 in self.regions:
            # Get raw features for this region
            feature = self._get_region_features(x0, x1, y0, y1)
            raw_features.append(feature)

            # Track total and max error for normalization
            error = feature[14]  # Richardson error (more accurate)
            total_error += error
            max_error = max(max_error, error)

        # Second pass: normalize features and add relative error information
        for feature in raw_features:
            # Extract components for normalization
            x0, x1, y0, y1 = feature[0], feature[1], feature[2], feature[3]
            x_width, y_width = feature[4], feature[5]
            area = feature[6]
            f_center = feature[7]
            gauss_integral = feature[8]
            x_variation, y_variation = feature[9], feature[10]
            max_variation, min_variation = feature[11], feature[12]
            variation_ratio = feature[13]
            richardson_error = feature[14]

            # Normalize position to [0,1] within domain
            norm_x0 = (x0 - self.ax) / self.x_width
            norm_x1 = (x1 - self.ax) / self.x_width
            norm_y0 = (y0 - self.ay) / self.y_width
            norm_y1 = (y1 - self.ay) / self.y_width

            # Normalize dimensions relative to domain
            norm_x_width = x_width / self.x_width
            norm_y_width = y_width / self.y_width

            # Normalize integration values by value scale and area
            scale_factor = max(area, 1e-10) * self.value_scale
            norm_gauss_integral = gauss_integral / scale_factor

            # Normalize error estimates relative to max error
            if max_error > 1e-10:
                norm_richardson_error = richardson_error / max_error
            else:
                norm_richardson_error = 0.0

            # Normalize variations
            norm_x_variation = np.tanh(x_variation / 10.0)  # Tanh keeps in [-1, 1]
            norm_y_variation = np.tanh(y_variation / 10.0)
            norm_max_variation = np.tanh(max_variation / 10.0)
            norm_min_variation = np.tanh(min_variation / 10.0)
            norm_variation_ratio = np.tanh(variation_ratio / 10.0)

            # Calculate relative error contribution
            rel_error_contribution = richardson_error / (total_error + 1e-10)

            # Create normalized feature vector
            normalized_feature = np.array([
                norm_x0, norm_x1, norm_y0, norm_y1,
                norm_x_width, norm_y_width,
                area,
                f_center,
                norm_gauss_integral,
                norm_x_variation, norm_y_variation,
                norm_max_variation, norm_min_variation,
                norm_variation_ratio,
                norm_richardson_error,
                rel_error_contribution
            ], dtype=np.float32)

            features.append(normalized_feature)

        # Sort regions by error contribution (highest error first)
        if features:
            features.sort(key=lambda x: x[15], reverse=True)

            # Store mapping from sorted indices to original indices
            self.sorted_to_original_idx = {}
            for i, (x0, x1, y0, y1) in enumerate(self.regions):
                for j, feature in enumerate(features):
                    # Match original region to sorted feature using normalized positions
                    orig_x0_norm = (x0 - self.ax) / self.x_width
                    orig_x1_norm = (x1 - self.ax) / self.x_width
                    orig_y0_norm = (y0 - self.ay) / self.y_width
                    orig_y1_norm = (y1 - self.ay) / self.y_width
                    if (abs(feature[0] - orig_x0_norm) < 1e-6 and
                        abs(feature[1] - orig_x1_norm) < 1e-6 and
                        abs(feature[2] - orig_y0_norm) < 1e-6 and
                        abs(feature[3] - orig_y1_norm) < 1e-6):
                        self.sorted_to_original_idx[j] = i
                        break

        # Pad to max_intervals with zeros
        while len(features) < self.max_intervals:
            features.append(np.zeros(16, dtype=np.float32))

        # Calculate current approximation and error
        approx = sum(self._gauss_legendre_2d(x0, x1, y0, y1) for x0, x1, y0, y1 in self.regions)
        error = abs(approx - self.true_value)

        # Normalize global statistics
        norm_region_count = len(self.regions) / self.max_intervals
        norm_approx = approx / (self.value_scale * self.x_width * self.y_width)

        # Normalize error on log scale to handle wide range of errors
        if self.true_value != 0:
            rel_error = min(error / abs(self.true_value), 1.0)  # Cap at 100% error
        else:
            rel_error = min(error, 1.0)  # If true value is 0, use absolute error capped at 1
        norm_error = np.log1p(rel_error * 10) / np.log(11)  # Maps [0,1] to [0,1] with log scaling

        # Normalized evaluation count
        norm_evals = len(self.evals) / (self.max_intervals * 25)  # Assuming ~25 evals per region max

        # Normalized step count
        norm_steps = self.steps / 50.0  # Assuming max steps of 50

        # Global stats with normalization
        global_stats = np.array([
            norm_region_count,
            norm_approx,
            norm_error,
            norm_evals,
            norm_steps
        ] + self.param_values, dtype=np.float32)

        # Return flattened array of all features and global stats
        output = np.concatenate([np.concatenate(features), global_stats])
        return np.clip(output, -10.0, 10.0)  # Clip to observation space bounds

    def _analyze_curvature(self, x0, x1, y0, y1, n_points=20):
        """Analyze function curvature in the region"""
        x = np.linspace(x0, x1, n_points)
        y = np.linspace(y0, y1, n_points)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self._eval(xi, yi) for xi in x] for yi in y])

        # Calculate gradient and curvature
        dx, dy = np.gradient(Z)
        dx2, _ = np.gradient(dx)
        _, dy2 = np.gradient(dy)

        # Mean absolute curvature
        curvature = np.mean(np.abs(dx2 + dy2))
        # Gradient magnitude
        gradient_mag = np.mean(np.sqrt(dx*2 + dy*2))
        # Oscillation measure
        oscillation = np.std(np.diff(Z.flatten()))

        return {
            'curvature': curvature,
            'gradient_mag': gradient_mag,
            'oscillation': oscillation
        }

    def _calculate_enhanced_reward(self, prev_error, new_error, evals_used, old_features, new_features):
        """Calculate reward with enhanced metrics"""
        try:
            # Basic error reduction reward
            error_reduction = prev_error - new_error
            efficiency = error_reduction / max(np.sqrt(evals_used), 1.0)
            base_reward = 10.0 * efficiency

            # Function behavior rewards
            behavior_old = self._analyze_function_behavior(
                old_features[0], old_features[1],
                old_features[2], old_features[3]
            )

            # Calculate average behavior for new regions
            behavior_new = []
            for features in new_features:
                b = self._analyze_function_behavior(
                    features[0], features[1],
                    features[2], features[3]
                )
                behavior_new.append(b)

            # Curvature analysis for old and new regions
            curv_old = self._analyze_curvature(
                old_features[0], old_features[1],
                old_features[2], old_features[3]
            )

            curv_new = [
                self._analyze_curvature(f[0], f[1], f[2], f[3])
                for f in new_features
            ]

            # Reward components based on function properties
            oscillation_factor = max(
                1.0,
                behavior_old['oscillation'] / (np.mean([b['oscillation'] for b in behavior_new]) + 1e-10)
            )

            smoothness_factor = max(
                1.0,
                behavior_old['smoothness'] / (np.mean([b['smoothness'] for b in behavior_new]) + 1e-10)
            )

            curvature_factor = max(
                1.0,
                curv_old['curvature'] / (np.mean([c['curvature'] for c in curv_new]) + 1e-10)
            )

            # Additional rewards for handling challenging regions well
            complexity_bonus = 2.0 * (
                oscillation_factor +
                smoothness_factor +
                curvature_factor
            ) / 3.0

            # Gradient-based reward
            gradient_improvement = max(
                0,
                curv_old['gradient_mag'] - np.mean([c['gradient_mag'] for c in curv_new])
            )
            gradient_reward = 3.0 * gradient_improvement

            # Combine all reward components
            total_reward = (
                base_reward +
                complexity_bonus +
                gradient_reward -
                0.1  # Small constant penalty
            )

            # Scale reward based on region properties
            if behavior_old['oscillation'] > 2.0 or curv_old['curvature'] > 5.0:
                total_reward *= 1.5  # Bonus for handling difficult regions

            # Terminal rewards
            if len(self.regions) >= self.max_intervals:
                if new_error < prev_error:
                    accuracy_ratio = min(self.max_intervals / len(self.regions), 1.0)
                    difficulty_factor = max(
                        1.0,
                        np.mean([b['oscillation'] * c['curvature']
                               for b, c in zip(behavior_new, curv_new)])
                    )
                    terminal_bonus = 5.0 * accuracy_ratio * difficulty_factor
                    total_reward += terminal_bonus

            return np.clip(total_reward, -10.0, 10.0)  # Clip reward for stability

        except Exception as e:
            print(f"Warning: Error calculating enhanced reward: {str(e)}")
            return -1.0

    def step(self, action):
        """
        Take a step by splitting a region with enhanced strategy.

        Args:
            action (ndarray): [region_idx_normalized, split_ratio, dimension, strategy]

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.steps += 1

        # Extract action components with fallback for compatibility
        if len(action) == 4:
            region_idx_normalized, split_ratio, dimension, strategy = action
        else:
            region_idx_normalized, split_ratio, dimension = action
            strategy = 0.5  # Default strategy

        # Store previous state for reward calculation
        prev_evals_count = len(self.evals)
        prev_approx = sum(self._gauss_legendre_2d(x0, x1, y0, y1) for x0, x1, y0, y1 in self.regions)
        prev_error = abs(prev_approx - self.true_value)

        # Determine which region to split based on strategy
        if strategy > 0.7:  # Use highest-error region
            # Sort regions by estimated error
            sorted_regions = sorted(
                range(len(self.regions)),
                key=lambda i: self._get_region_features(*self.regions[i])[14],  # Richardson error
                reverse=True
            )
            region_idx = sorted_regions[0] if sorted_regions else 0
        else:  # Use selected region
            region_idx = int(region_idx_normalized * (len(self.regions) - 0.001))

            # Map through sorted_to_original_idx if available
            if hasattr(self, 'sorted_to_original_idx') and region_idx in self.sorted_to_original_idx:
                region_idx = self.sorted_to_original_idx[region_idx]

        # Handle out-of-bounds index
        if region_idx >= len(self.regions):
            return self._get_observation(), -1.0, True, False, {
                "error": prev_error,
                "approximation": prev_approx,
                "evals": len(self.evals)
            }

        # Extract the region to split
        x0, x1, y0, y1 = self.regions.pop(region_idx)
        x_width = x1 - x0
        y_width = y1 - y0

        # Get features of the region being split for reward calculation
        old_features = self._get_region_features(x0, x1, y0, y1)
        old_error_estimate = old_features[14]  # Richardson error estimate

        # Apply adaptive splitting based on strategy
        if 0.3 < strategy <= 0.7:
            # Adaptive split based on function behavior
            # Sample more points to find where function changes most
            if dimension == 0:  # x-split
                n_samples = 5
                sample_points = np.linspace(x0, x1, n_samples+2)[1:-1]  # Skip endpoints
                sample_values = [self._eval(x, (y0 + y1) / 2) for x in sample_points]

                # Find largest change in function values
                changes = [abs(sample_values[i+1] - sample_values[i])
                           for i in range(len(sample_values)-1)]
                if changes:
                    max_change_idx = np.argmax(changes)
                    split_point = (sample_points[max_change_idx] +
                                   sample_points[max_change_idx+1]) / 2
                else:
                    # Default to midpoint
                    split_point = (x0 + x1) / 2
                new_regions = [(x0, split_point, y0, y1), (split_point, x1, y0, y1)]
            else:  # y-split
                n_samples = 5
                sample_points = np.linspace(y0, y1, n_samples+2)[1:-1]  # Skip endpoints
                sample_values = [self._eval((x0 + x1) / 2, y) for y in sample_points]

                # Find largest change in function values
                changes = [abs(sample_values[i+1] - sample_values[i])
                           for i in range(len(sample_values)-1)]
                if changes:
                    max_change_idx = np.argmax(changes)
                    split_point = (sample_points[max_change_idx] +
                                   sample_points[max_change_idx+1]) / 2
                else:
                    # Default to midpoint
                    split_point = (y0 + y1) / 2
                new_regions = [(x0, x1, y0, split_point), (x0, x1, split_point, y1)]
        else:
            # User-specified split ratio
            if dimension == 0:  # x-split
                split_point = x0 + split_ratio * x_width
                new_regions = [(x0, split_point, y0, y1), (split_point, x1, y0, y1)]
            else:  # y-split
                split_point = y0 + split_ratio * y_width
                new_regions = [(x0, x1, y0, split_point), (x0, x1, split_point, y1)]

        # Add new regions
        self.regions.extend(new_regions)

        # Keep track of split history
        self.region_history.append((x0, x1, y0, y1, split_point, dimension))

        # Calculate new approximation and error
        new_approx = sum(self._adaptive_integrate(x0, x1, y0, y1)[0]
                         for x0, x1, y0, y1 in self.regions)
        new_error = abs(new_approx - self.true_value)

        # Calculate error reduction and evaluations used
        error_reduction = prev_error - new_error
        evals_used = len(self.evals) - prev_evals_count

        # Calculate efficiency metrics
        efficiency_factor = 1.0 / np.sqrt(max(evals_used, 1))
        error_reduction = prev_error - new_error
        total_error = sum(self._get_region_features(*region)[14] for region in self.regions)
        max_error = max(self._get_region_features(*region)[14] for region in self.regions)

        # Calculate immediate reward
        immediate_reward = 10.0 * error_reduction * efficiency_factor

        # Calculate local metrics
        local_errors = [self._get_region_features(*region)[14] for region in new_regions]
        local_error_improvement = (max(old_error_estimate, 1e-10) - max(local_errors, default=0)) / max(old_error_estimate, 1e-10)
        local_error_ratio = max(local_errors, default=0) / (total_error + 1e-10)

        # Calculate reward components
        local_reward = 5.0 * local_error_improvement * efficiency_factor
        smoothness_reward = 5.0 * (1.0 - np.std(local_errors) / (np.mean(local_errors) + 1e-10))
        exploration_reward = 2.5 * old_error_estimate * (1 + local_error_ratio)

        # Combine rewards with weights
        global_weight = 0.6
        local_weight = 0.4
        base_reward = (
            global_weight * (immediate_reward + exploration_reward) +
            local_weight * (local_reward + smoothness_reward)
        )

        # Apply urgency factor
        regions_left = max(0, self.max_intervals - len(self.regions))
        urgency_factor = np.exp(-regions_left / (self.max_intervals * 0.3))
        total_reward = base_reward * (1.0 + urgency_factor) - 0.1  # Small constant penalty

        done = len(self.regions) >= self.max_intervals

        # Terminal rewards with local error consideration
        if done:
            if new_error < prev_error:
                accuracy_ratio = min(self.max_intervals / max(len(self.regions), 1), 1.0)
                global_accuracy = 1.0 - min(new_error * 1e6, 1.0)
                local_accuracy = 1.0 - min(max(local_errors, default=0) * 1e6, 1.0)

                terminal_bonus = 10.0 * accuracy_ratio * (
                    global_weight * global_accuracy +
                    local_weight * local_accuracy
                )
                total_reward += terminal_bonus

        # Sort regions by position for more efficient lookup
        self.regions.sort(key=lambda x: (x[0], x[2]))

        old_features = self._get_region_features(x0, x1, y0, y1)
        new_features = [self._get_region_features(*region) for region in new_regions]

        reward = self._calculate_enhanced_reward(
            prev_error,
            new_error,
            evals_used,
            old_features,
            new_features
        )

        return self._get_observation(), reward, done, False, {
            "error": new_error,
            "approximation": new_approx,
            "evals": len(self.evals),
            "regions": len(self.regions),
            "efficiency": error_reduction / max(evals_used, 1) if evals_used > 0 else 0
        }

    def visualize_solution(self, num_points=500):
        """Enhanced visualization with PFEM particle distribution"""
        # Create figure with subplots
        plt.figure(figsize=(15, 10))

        # Plot function and regions
        plt.subplot(2, 2, 1)
        x = np.linspace(self.ax, self.bx, num_points)
        y = np.linspace(self.ay, self.by, num_points)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(self.f)(X, Y)

        plt.contourf(X, Y, Z, cmap='viridis')
        plt.colorbar(label='f(x, y)')

        # Plot integration regions
        for x0, x1, y0, y1 in self.regions:
            plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'r-', alpha=0.5)

        plt.title('Function and Integration Regions')
        plt.xlabel('x')
        plt.ylabel('y')

        # Plot error distribution
        plt.subplot(2, 2, 2)
        errors = [self._get_region_features(x0, x1, y0, y1)[14] for x0, x1, y0, y1 in self.regions]
        region_centers_x = [(x0 + x1) / 2 for x0, x1, y0, y1 in self.regions]
        region_centers_y = [(y0 + y1) / 2 for x0, x1, y0, y1 in self.regions]

        plt.scatter(region_centers_x, region_centers_y, c=errors, cmap='hot', s=100)
        plt.colorbar(label='Error Estimate')
        plt.title('Error Distribution')
        plt.xlabel('x')
        plt.ylabel('y')

        # Plot PFEM particles if used
        if self.use_pfem:
            plt.subplot(2, 2, 3)
            particle_x = [p.x for p in self.pfem_integrator.particles]
            particle_y = [p.y for p in self.pfem_integrator.particles]
            particle_errors = [p.error_estimate for p in self.pfem_integrator.particles]

            plt.scatter(particle_x, particle_y,
                       c=particle_errors,
                       cmap='hot',
                       alpha=0.8,
                       s=50)
            plt.colorbar(label='Particle Error')
            plt.title('PFEM Particle Distribution')
            plt.xlabel('x')
            plt.ylabel('y')

            # Plot particle connectivity
            for p in self.pfem_integrator.particles:
                for n in p.neighbors:
                    plt.plot([p.x, n.x], [p.y, n.y], 'b-', alpha=0.2)

        # Plot convergence history
        plt.subplot(2, 2, 4)
        if hasattr(self, 'region_history'):
            history_x = range(len(self.region_history))
            errors = [abs(r[0] - self.true_value) for r in self.region_history]
            plt.semilogy(history_x, errors, 'b-', label='Error')
            plt.grid(True)
            plt.title('Convergence History')
            plt.xlabel('Split Number')
            plt.ylabel('Error (log scale)')

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        approx = sum(self._adaptive_integrate(x0, x1, y0, y1)[0] for x0, x1, y0, y1 in self.regions)
        error = abs(approx - self.true_value)

        print("\nSummary Statistics:")
        print(f"True Value:           {self.true_value:.10e}")
        print(f"Approximation:        {approx:.10e}")
        print(f"Absolute Error:       {error:.10e}")
        print(f"Relative Error:       {error/abs(self.true_value):.10e}")
        print(f"Function Evaluations: {len(self.evals)}")
        print(f"Number of Regions:    {len(self.regions)}")
        if self.use_pfem:
            print(f"PFEM Particles:       {len(self.pfem_integrator.particles)}")