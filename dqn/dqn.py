import time
import sys
import itertools
import random
import math
from functools import partial
import pickle as pkl
import numpy as np

import jax
# import jax.config
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, tree_map, value_and_grad
from jax.experimental import optimizers

from replay_buffers import BasicBuffer
from models import *


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
        n_feat:         int
                        feature vector dimension (QMPS output dimension if NN is used as well)
        factor:         float
                        factor by which initial QMPS tensor parameters are scaled
        offset:         float
                        constant to add as offset to QMPS output
        scale:          float
                        constant by which QMPS output is scaled
        sign:           float
                        which sign should QMPS output have (after log is applied), only important when training without NN
        chiq:           int
                        QMPS bond dimension if NN is used as well
        polyak:         bool
                        polyakov averaging for weights in target network
        tau:            float
                        polyakov averaging parameter
        goal:           bool
                        true for multi-task RL
        initial_params  list(ndarray)
                        initial paramters for QMPS network (otherwise it is initialized randomly)
        dtype:          string
                        sets precision/type of numpy arrays
        profiling:      bool
                        when profiling code (needed due to jax's asynchronous dispatch)
        seed:           int
                        random seed
    """
    def __init__(self,
                env,
                double: bool=True,
                learning_rate: float=3e-4,
                gamma: float=1.0,
                buffer_size: int=10000,
                batch_size: int=128,
                hidden_dim: int=32,
                uniform: bool=True,
                nn: bool=True,
                n_feat: int=32,
                factor: float=4.0,
                offset: float=0,
                scale: float=1.0,
                sign: float=1.0,
                chiq: int=16,
                polyak: bool=False,
                tau: float=0.01,
                goal: bool=False,
                initial_params=None,
                dtype: str="float",
                profiling: bool=False,
                seed: int=123
                ):

        self.env = env
        self.double = double
        self.polyak = polyak
        self.tau = tau
        self.gamma = gamma
        self.dtype = dtype
        self.profiling = profiling
        self.nn = nn
        print("Use NN: ", nn)

        self.itercount = itertools.count()
        np.random.seed(seed)

        self.n_states = env.L
        self.n_actions = env.n_actions

        # if self.env.time_qudit:
        #     MPS = QMPS_Time
        # else:
        #     MPS = QMPS

        self.replay_buffer = BasicBuffer(env, max_size=buffer_size, batch_size=batch_size, goal=goal)

        # NN layers
        if goal:
            layer_sizes = [n_feat+1, hidden_dim, hidden_dim, self.n_actions]
        else:
            layer_sizes = [n_feat, hidden_dim, hidden_dim, self.n_actions]
        mps_output_size = n_feat if nn else self.n_actions

        self.model = QMPS(self.env.L, scale=scale, sign=sign, offset=offset, nn=self.nn, goal=goal)
        self.value_and_grad = self.model.value_and_grad_nn_goals if goal else self.model.value_and_grad_nn
        self.predict_single = self.model.predict_single
        self.predict = self.model.predict

        d = 2
        std = 0.2
        norm_factor = factor
        if nn:
            hidden_dim = 2**(env.L//2)
            if hidden_dim > chiq:
                hidden_dim = chiq

        init_params_list = [self.model.init_random_params(mps_output_size, d, hidden_dim, env.n_time_steps, std=std, norm_factor=norm_factor, uniform=uniform) for _ in range(2)]
        init_params = [jnp.array([p1, p2]) for p1,p2 in zip(init_params_list[0], init_params_list[1])]#, scale*np.ones(env.n_actions), offset*np.ones(env.n_actions)]
        print("QMPS tensor dimensions: ", [p.shape for p in init_params])

        # print([np.min(np.abs(p)) for p in init_params])
        # print([np.max(np.abs(p)) for p in init_params])

        nn_scale = 0.1
        init_params = [init_params, initialize_nn(nn_scale, layer_sizes)] if nn else [init_params]
        print("NN layer dimensions: ", layer_sizes)

        if initial_params != None:
            print("Use pre-trained parameters!")
            with open(initial_params+".pkl", 'rb') as handle:
                init_params = pkl.load(handle)

        self.target_params = init_params.copy()

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(init_params)
        self.params = init_params


    def norm(self):
        """ Norm of QMPS """
        params = self.get_params(self.opt_state)
        return self.model.norm(params)[0]

    # @profile
    def get_action(self, state, goal, eps=0.0):
        """ epsilon greedy action selection (acts greedily by default) """
        if(np.random.rand() < eps):
            return np.random.randint(self.n_actions)
        else:
            params = self.get_params(self.opt_state)
            qvals = self.predict_single(params, state, goal=goal)
            if self.profiling: qvals.block_until_ready()
            action = jnp.argmax(qvals)
            # print(action)
            if self.profiling: action.block_until_ready()
        return action

    @partial(jit, static_argnums=(0,))
    def loss(self, params, states, labels, actions, goals=None):
        """ DQN regression loss """
        preds = self.predict(params, states, goals=goals)
        preds_select = jnp.take_along_axis(preds, jnp.expand_dims(actions, axis=1), axis=1)
        return jnp.mean(0.5 * (preds_select.squeeze() - labels)**2)

    @partial(jit, static_argnums=(0,))
    def calculate_target(self, params, next_states, rewards, dones, target_params, goals=None):
        """ DQN regression target """
        if self.double: # Hasselt 2015
            max_actions = jnp.argmax(self.predict(params, next_states, goals=goals), 1) # take argmax action according to model
            # print(max_actions)
            max_next_Q = jnp.take_along_axis(self.predict(target_params, next_states, goals=goals), jnp.expand_dims(max_actions, axis=1), axis=1)# use target for evaluation
            max_next_Q = max_next_Q.squeeze(1)
        else:
            next_Q = self.predict(target_params, next_states, goals=goals)
            max_next_Q = jnp.max(next_Q, 1)

        labels = jax.lax.stop_gradient(rewards + (1 - dones) * self.gamma * max_next_Q)
        return labels


    # @profile
    def update(self, batch_size):
        """ updates model (and target if polyak averaging) on single mini batch """
        transitions = self.replay_buffer.sample(batch_size)
        states, next_states, time_batch, next_time_batch, rewards, actions, dones, goals = transitions

        if self.env.time_qudit:
            states = (states, time_batch)
            next_states = (next_states, next_time_batch)

        params = self.get_params(self.opt_state)
        labels = self.calculate_target(params, next_states, rewards, dones, self.target_params, goals=goals)
        if self.profiling: labels.block_until_ready()


        # start_time = time.time()
        loss, gradients = self.model.value_and_grad_nn(params, states, labels, actions, goals=goals)
        if self.profiling: [g.block_until_ready() for g in gradients[0]]
        # print("CUS3: ", time.time() - start_time)


        # start_time = time.time()
        self.opt_state = jit(self.opt_update)(next(self.itercount), gradients, self.opt_state)
        if self.profiling: jax.tree_util.tree_flatten(self.opt_state)[0][0].block_until_ready()
        # print(time.time() - start_time)


        # target network update
        if self.double and self.polyak:
            params = self.get_params(self.opt_state)
            for i, (target_param, param) in enumerate(zip(self.target_params, params)):
                self.target_params[i] = (self.tau * param + (1 - self.tau) * target_param).copy()

        return loss
