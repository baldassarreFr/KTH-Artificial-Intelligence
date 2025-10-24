#!/usr/bin/env python3
from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
from hmm import HMM

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        self.models = [HMM(2, N_EMISSIONS) for _ in range(N_SPECIES)]
        self.fish = [[] for _ in range(N_FISH)]
        self.guessed = set()

    def guess(self, step, observations):
        for i in range(N_FISH):
            self.fish[i].append(observations[i])
        if N_STEPS - N_FISH >= step:
            return None
        else:
            probs = [[model.sequnceProbabilities(self.fish[i]) for model in self.models]
                     if i not in self.guessed else [-1 for _ in self.models] for i in range(N_FISH)]
            f_probs = [max(m_prob) for m_prob in probs]
            m = max(f_probs)
            f_i = f_probs.index(m)
            m_i = probs[f_i].index(m)
            return f_i, m_i

    def reveal(self, correct, fish_id, true_type):
        if not correct:
            self.models[true_type].train(self.fish[fish_id], 10)
        self.guessed.add(fish_id)
