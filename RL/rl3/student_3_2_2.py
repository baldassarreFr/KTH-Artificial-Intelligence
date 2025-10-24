#!/usr/bin/env python3
# rewards: [golden_fish, jellyfish_1, jellyfish_2, ... , step]
rewards = [1000.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -2.0]

# Q learning learning rate
alpha = 0.5 # faster updates for quick convergence

# Q learning discount rate
gamma = 0.85 # focus more on near-term (shortest path) than very long-term

# Epsilon initial
epsilon_initial = 1.0

# Epsilon final
epsilon_final = 0.01

# Annealing timesteps
annealing_timesteps = 8000

# threshold
threshold = 1e-6
