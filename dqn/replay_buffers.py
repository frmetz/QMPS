import time
import sys
import random, itertools
import numpy as np
from collections import deque, namedtuple
from functools import partial

import jax
# import jax.config
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

sys.path.append('../environment/')
import env as environment


class BasicBuffer_cpu: # Faster on CPU
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

    def __init__(self, env, max_size, batch_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.L = env.L
        self.env = env
        self.batch_size = batch_size

        if env.library == environment.Library.TN:
            self.states = [np.empty((batch_size, s.shape[0], s.shape[1], s.shape[2]), dtype=env.ctype) for s in env.initial_state]
            self.next_states = [np.empty((batch_size, s.shape[0], s.shape[1], s.shape[2]), dtype=env.ctype) for s in env.initial_state]

    # @profile
    def push(self, state, action, reward, next_state, done):
        """ Pushes single transition onto deque """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    # @profile
    def sample(self):
        """ Samples batch of transition from deque """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, self.batch_size)
        for i,experience in enumerate(batch):
            state, action, reward, next_state, done = experience
            if self.env.library == environment.Library.TN:
                for l in range(self.L):
                    self.states[l][i] = state[l]
                    self.next_states[l][i] = next_state[l]
            else:
                state_batch.append(state)
                next_state_batch.append(next_state)
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)
        return self.states, self.next_states, np.array(reward_batch, dtype=self.env.dtype), np.array(action_batch, dtype=np.int16), np.array(done_batch, dtype=bool)

# @jit
# def stack(x):
#     return jax.tree_map(lambda *xs: jnp.stack(xs), *x)

@partial(jax.jit, donate_argnums=0)
def inplace_store(buffer_tree, ptr, value_tree):
    return jax.tree_map(lambda buffer, value: buffer.at[ptr].set(value), buffer_tree, value_tree)

class BasicBuffer_gpu: # Faster on GPU
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

    def __init__(self, env, max_size, batch_size, seed=1):
        self.max_size = max_size
        self.batch_size = batch_size
        self.env = env
        self.counter = 0
        self.rng = jax.random.PRNGKey(seed)

        example_tree = [env.initial_state, env.initial_state, jnp.zeros((), dtype=np.float), jnp.zeros((), dtype=np.int), jnp.zeros((), dtype=np.bool)]
        self.buffer = jax.tree_map(lambda example: jnp.zeros_like(example, shape=(max_size, *example.shape)), example_tree)

    # @profile
    def push(self, state, action, reward, next_state, done):
        """ Pushes single transition onto deque """
        value = [state, next_state, reward, action, done]
        self.buffer = inplace_store(self.buffer, self.counter, value)
        self.counter = (self.counter + 1) % self.max_size

    @partial(jax.jit, static_argnums=(0,), donate_argnums=2)
    def _sample(self, buffer_tree, key):
        key, subkey= jax.random.split(key)
        indices = jax.random.choice(subkey, self.max_size, shape=(self.batch_size,), replace=False)
        return jax.tree_map(lambda t: t[indices], buffer_tree), key

    # @profile
    def sample(self):
        """ Samples batch of transition from deque """
        batch, self.rng = self._sample(self.buffer, self.rng)
        return batch
