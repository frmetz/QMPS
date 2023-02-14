import time
import itertools
from functools import partial
import random
import copy
import numpy as np

import jax
# import jax.config
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, tree_map, value_and_grad
from jax.example_libraries import optimizers

from replay_buffers import *
from models import *


def block_until_ready(pytree):
  return tree_map(lambda x: x.block_until_ready(), pytree)

class DQNAgent:
    """
    Deep Q-Learning agent

    Parameters:
        env:            SpinChainEnv object
                        RL environment
        double:         bool
                        Double Q-learning
        learning_rate:  float
                        learning rate
        gamma:          float
                        discount factor
        buffer_size:    int
                        size of replay buffer
        hidden_dim:     int
                        number of neurons in all hidden layers
        batch_size:     int
                        batch size for update
        uniform:        bool
                        whether QMPS bond dimension is fixed uniformly or not (exponentially growing towards the center)
        nn:             bool
                        whether to append neural network to QMPS output
        tn:             str
                        determines tn ansatz: either 'mps' or 'mpo'
        n_feat:         int
                        feature vector dimension (QMPS output dimension if NN is used as well)
        std:            float
                        Gaussian standard deviation when initializing tensors
        D:              int
                        QMPS bond dimension if NN is used as well
        initial_params  list(ndarray)
                        initial paramters for QMPS network (otherwise it is initialized randomly)
        seed:           int
                        random seed
    """
    def __init__(self,
                env,
                double: bool=True,
                learning_rate: float=1e-4,
                gamma: float=1.0,
                buffer_size: int=8000,
                batch_size: int=64,
                hidden_dim: int=100,
                uniform: bool=True,
                nn: bool=True,
                tn='mps',
                n_feat: int=32,
                std: float=0.5,
                D=16,
                initial_params=None,
                seed: int=123,
                factor: float=1.0,
                offset: float=0,
                scale: float=1.0,
                sign: float=1.0,
                share=True,
                fixed=False,
                dtype: str="float",
                profiling: bool=False
                ):

        self.env = env
        self.double = double
        self.gamma = gamma
        self.D = D
        self.profiling = profiling
        self.nn = nn
        print("Use NN: ", nn)
        print("Default platform: ", jax.devices()[0].platform)

        Buffer = BasicBuffer_cpu if jax.devices()[0].platform == "cpu" else BasicBuffer_gpu
        self.replay_buffer = Buffer(env, max_size=buffer_size, batch_size=batch_size)

        np.random.seed(seed)
        self.rng = jax.random.PRNGKey(seed)

        layer_sizes = [n_feat, hidden_dim, hidden_dim, self.env.n_actions]
        print("NN layer dimensions: ", layer_sizes)

        d = 2 if tn == 'mps' else 4
        D = d**(env.L//2) if env.L < 20 else d**10
        D = self.D if D > self.D else D

        n_feat = n_feat if nn else self.env.n_actions
        self.share = share

        TN = QMPS if tn == 'mps' else QMPO
        self.model = TN.eye(env.L, 2, D, n_feat, batch_size, share=share, uniform=uniform, std=std, norm_factor=factor, nn=nn, layer_sizes=layer_sizes, nn_scale=0.1)

        # fixed = False
        self.fixed = fixed
        # self.opt_init, self.opt_update, self.get_params = optimizers.sgd(learning_rate)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(self.model.params) if not fixed else self.opt_init(self.model.params[1])
        self.params = self.get_params(self.opt_state)
        self.itercount = itertools.count()

        self.tensors = [self.model.get_tensors(self.model.params[0])] + [self.model.params[1]]
        self.target_tensors = self.tensors.copy()

        print("QMPS tensor dimensions: ", [t.shape for t in self.tensors[0]])


    @partial(jit, static_argnums=(0,), donate_argnums=1)
    def random_action(self, key):
        key, subkey= jax.random.split(key)
        action = jax.random.choice(subkey, self.env.n_actions)
        return action, key

    @partial(jit, static_argnums=(0,))
    def greedy_action(self, tensors, state):
        qvals = self.model.predict_single(tensors, state)
        return jnp.argmax(qvals)

    # @profile
    def get_action(self, state, eps=0.0):
        """ epsilon greedy action selection (acts greedily by default) """
        if(np.random.rand() < eps):
            action, self.rng = self.random_action(self.rng)
        else:
            action = self.greedy_action(self.tensors, state)
        return action

    @partial(jit, static_argnums=(0,))
    def loss(self, params, states, labels, actions):
        """ DQN regression loss """
        preds = self.model.predict2(params, states)
        preds_select = jnp.take_along_axis(preds, jnp.expand_dims(actions, axis=1), axis=1)
        return jnp.mean(0.5 * (preds_select.squeeze() - labels)**2)

    @partial(jit, static_argnums=(0,))
    def loss_fixed_batch(self, params, tensors, states, labels, actions):
        """ DQN regression loss """
        preds = self.model.predict_fixed_batch(params, tensors, states)
        preds_select = jnp.take_along_axis(preds, jnp.expand_dims(actions, axis=1), axis=1)
        return jnp.mean(0.5 * (preds_select.squeeze() - labels)**2)

    @partial(jit, static_argnums=(0,))
    def calculate_target(self, params, next_states, rewards, dones, target_params):
        """ DQN regression target """
        if self.double: # Hasselt 2015
            max_actions = jnp.argmax(self.model.predict(params, next_states), 1) # take argmax action according to model
            max_next_Q = jnp.take_along_axis(self.model.predict(target_params, next_states), jnp.expand_dims(max_actions, axis=1), axis=1)# use target for evaluation
            max_next_Q = max_next_Q.squeeze(1)
        else:
            next_Q = self.model.predict(target_params, next_states)
            max_next_Q = jnp.max(next_Q, 1)

        labels = jax.lax.stop_gradient(rewards + (1 - dones) * self.gamma * max_next_Q)
        return labels

    # @profile
    def update(self, batch_size):
        """ Updates model on single mini batch """
        transitions = self.replay_buffer.sample()
        states, next_states, rewards, actions, dones = transitions

        labels = self.calculate_target(self.tensors, next_states, rewards, dones, self.target_tensors)
        if self.profiling: labels.block_until_ready()

        if self.fixed:
            if self.share:
                raise NotImplementedError()
            else:
                loss, gradients = value_and_grad(self.loss_fixed_batch)(self.params, self.tensors[0], states, labels, actions)
        else:
            loss, gradients = self.model.value_and_grad(self.params, self.tensors, states, labels, actions)


        self.opt_state = jit(self.opt_update)(next(self.itercount), gradients, self.opt_state)
        if self.profiling: block_until_ready(self.opt_state)

        self.params = jit(self.get_params)(self.opt_state)
        if self.fixed:
            self.tensors[1] = self.params
        else:
            self.tensors = [self.model.get_tensors(self.params[0])] + [self.params[1]]

        return loss
