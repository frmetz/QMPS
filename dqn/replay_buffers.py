import time
import sys
import random, itertools
import numpy as np
from collections import deque

import jax
# import jax.config
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.append('../environment/')
import env as environment


class BasicBuffer:
    """
    DQN Replay Buffer

    Parameters:
        env:            SpinChainEnv object
                        RL environment
        max_size:       int
                        buffer (deque) size
        batch_size:     int
                        batch size used for QMPS training
    """

    def __init__(self, env, max_size, batch_size, goal=False):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.L = env.L
        self.env = env
        self.goal = goal

        if env.library == environment.Library.TN:
            self.states = [np.empty((batch_size, s.shape[0], s.shape[1], s.shape[2]), dtype=env.ctype) for s in env.initial_state]
            self.next_states = [np.empty((batch_size, s.shape[0], s.shape[1], s.shape[2]), dtype=env.ctype) for s in env.initial_state]

    # @profile
    def push(self, state, action, reward, next_state, done, goal):
        """ Pushes single transition onto deque """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    # @profile
    def sample(self, batch_size):
        """ Samples batch of transition from deque """
        state_batch = []
        time_batch = []
        action_batch = []
        action_idx_batch = []
        reward_batch = []
        next_state_batch = []
        next_time_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        # start_time = time.time()
        for i,experience in enumerate(batch):
            state, action, reward, next_state, done = experience
            if self.env.library == environment.Library.TN:
                if self.env.time_qudit:
                    state, times = state
                    next_state, next_times = next_state
                    time_batch.append(times)
                    next_time_batch.append(next_times)
                for l in range(self.L):
                    self.states[l][i] = state[l]
                    self.next_states[l][i] = next_state[l]
            else:
                state_batch.append(state)
                next_state_batch.append(next_state)

            if self.env.continuous:
                action_batch.append(action[1])
                action_idx_batch.append(action[0])
            else:
                action_batch.append(action)

            reward_batch.append(reward)
            done_batch.append(done)
        # print("BUFFER ",time.time() - start_time)

        reward_batch = np.array(reward_batch)
        state_batch = np.array(state_batch)
        next_state_batch = np.array(next_state_batch)
        action_batch = np.array(action_batch)
        action_idx_batch = np.array(action_idx_batch)
        done_batch = np.array(done_batch)
        time_batch = np.array(time_batch)
        next_time_batch = np.array(next_time_batch)

        if self.env.library == environment.Library.QuSpin:
            if self.env.continuous:
                return (state_batch, next_state_batch,reward_batch, action_batch, action_idx_batch, done_batch)
            else:
                return (state_batch, next_state_batch, reward_batch, action_batch, done_batch)
        if self.env.time_qudit:
            if self.env.continuous:
                return (self.states, self.next_states, time_batch, next_time_batch, reward_batch, action_batch, action_idx_batch, done_batch)
            else:
                return (self.states, self.next_states, time_batch, next_time_batch, reward_batch, action_batch, done_batch)
        else:
            if self.env.continuous:
                return (self.states, self.next_states, None, None, reward_batch, action_batch, action_idx_batch, done_batch)
            else:
                return (self.states, self.next_states, None, None, reward_batch, action_batch, done_batch, None)
