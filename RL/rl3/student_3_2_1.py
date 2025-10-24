#!/usr/bin/env python3
# rewards: [golden_fish, jellyfish_1, jellyfish_2, ... , step]
rewards = [-1000.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -0.10]

# Q learning learning rate
alpha = 0.15

# Q learning discount rate
gamma = 0.99

# Epsilon initial
epsilon_initial = 1.0

# Epsilon final
epsilon_final = 0.05

# Annealing timesteps
annealing_timesteps = 10000

# threshold
threshold = 1e-6
