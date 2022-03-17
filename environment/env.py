import sys
import os
import copy
import random
import time
import pickle
import numpy as np
from enum import Enum, auto
from typing import List, Union, Text, Optional, Any, Type

import jax
# import jax.config
# jax.config.update("jax_enable_x64", True)

# if os.environ.get('CONDA_DEFAULT_ENV') == 'quspin':
#     print("Using QuSpin!")
#     from tlfi_quspin import *
#     from tlfi_tn import *
# else:
#     from tlfi_tn import *

from tlfi_tn import *


def spin_state(theta: float, phi: float, ctype=np.complex64):
    """ Single-particle bloch sphere state parameterized by angles theta and phi """
    return np.array([np.cos(theta/2.), np.exp(1.j*phi) * np.sin(theta/2.)], dtype=ctype)

def ghz_state(L: int, dtype=np.float32):
    """ MPS representation of GHZ state """
    tensor = np.zeros((2,2,2), dtype=dtype)
    tensor[0,0,0] = 1.
    tensor[1,1,1] = 1.
    t_boundary = np.eye(2, dtype=dtype)
    return [t_boundary.reshape(1,2,2)] + [tensor for _ in range(L-2)] + [t_boundary.reshape(2,2,1)]

def run_first_agent(continue_train):
    """ Runs one QMPS agent when doing multi-stage training """
    model, params, env = continue_train
    state = env.reset()
    done = False
    while not done:
        preds = model.predict_single_complex(params, state)
        a = np.argmax(preds).item()
        state, reward, done, trunc_err, ent  = env.step(a)
    return state.copy()


class RewardFunction(Enum):
    Fidelity = auto()
    LogFidelity = auto()
    EnergyDensity = auto()

class InitialState(Enum):
    ProductState = auto()
    GroundState = auto()
    RandomState = auto()

class Library(Enum):
    TN = "tn"
    TeNPy = "tenpy"
    QuSpin = "quspin"

    def __str__(self):
        return self.value


class SpinChainEnv():
    """
    Gym style environment for controlling 1d spin chains

    Parameters:
        L:                  int
                            Number of spins (length of the chain)
        n_time_steps:       int
                            (Maximum) number of agent time steps per episode
        delta_t             float
                            time step duration is: pi/(2*delta_t)
        library:            Enum(Library)
                            Library used for simulation, Library.TN, Library.TeNPy or Library.QuSpin
        D:                  int
                            quantum state bond dimension
        threshold:          float
                            single-particle fidelity threshold, e.g. 0.99
        random_init_state:  Enum(InitialState)
                            Class of initial states: InitialState.ProductState, InitialState.GroundState, InitialState.RandomState
        initial_state_params: dict(tuple)
                            initial ground state parameters: (J, gx, gz)
        final_state_params: dict(tuple)
                            target ground state parameters: (J, gx, gz)
        sample_states:      Optional(list)
                            Pregenerated list of initial states to sample from
        init_ents:          Optional(list)
                            To sample_states corresponding entropies
        random_complex:     bool
                            whether initial state is complex or real-valued (only when using InitialState.RandomState)
        continuous:         bool
                            whether action space is coninuous or discrete (only works with discrete at the moment)
        time_qudit:         bool
                            whether to include the episode time information in the RL state
        random_initial_state: bool
                            whether each episode starts from a differnt (random) initial state or a fixed one
        fixed_length_episode: bool
                            whether an episode runs for a fixed length or can be terminated before n_time_steps is reached
        reward_at_end:      bool
                            whether there is only one nonzero reward at the end of an episode or also at intermediate time steps
        reward_function:    Enum(RewardFunction)
                            Reward function used: RewardFunction.Fidelity, RewardFunction.LogFidelity (default), RewardFunction.EnergyDensity
        profiling:          bool
                            when profiling code (needed due to jax's asynchronous dispatch)
        continue_train:     tuple
                            (model, model_params, env) of trained QMPS agent for multi-state training
        seed:               int
                            random seed
        dtype:              str
                            datatype for real-valued arrays
        ctype:              str
                            datatype for complex-valued arrays
    """

    def __init__(self,
            L: int,
            n_time_steps: int,
            n_actions: int,
            delta_t: float = 10,
            library: Library = Library.TN,
            D: int = 1,
            threshold: float = 0.01,
            random_init_state: InitialState = InitialState.GroundState,
            initial_state_params: tuple = dict(J=0.0, gx=0.0, gz=1.0),
            final_state_params: tuple = dict(J=0.0, gx=1.0, gz=0.0),
            sample_states: List = None,
            init_ents: List = None,
            random_complex: bool = False,
            continuous: bool = False,
            time_qudit: bool = True,
            random_initial_state: bool = False,
            fixed_length_episode: bool = True,
            reward_at_end: bool = True,
            reward_function: RewardFunction = RewardFunction.LogFidelity,
            profiling: bool = False,
            continue_train: tuple = None,
            seed: int = 123,
            dtype: str = "float32",
            ctype: str = "complex64"):

        print("Default device: ", jax.devices()[0])
        self.L = L
        self.n_time_steps = n_time_steps
        offset = 2.5
        self.step_duration = [0.5 * np.pi/delta_t, 0.5 * np.pi/(delta_t + offset)] # step size for positive and negative ops respectively
        # self.step_duration = [0.5 * np.pi/delta_t, 0.5 * np.pi/delta_t]
        self.D = D
        self.library = library
        self.continuous = continuous
        self.time_qudit = time_qudit
        self.random_initial_state = random_initial_state
        self.threshold = np.log(1-threshold)
        self.reward_function = reward_function
        self.fixed_length_episode = fixed_length_episode
        self.reward_at_end = reward_at_end
        self.profiling = profiling
        self.random_init_state = random_init_state
        self.random_complex = random_complex
        self.continue_train = continue_train
        self.dtype = dtype
        self.ctype = ctype

        random.seed(seed)
        np.random.seed(seed)
        np.random.RandomState(seed)
        rng = jax.random.PRNGKey(seed)

        if sample_states != None:
            self.sample_states = []
            for s in sample_states:
                tmp = []
                for t in s:
                    tmp.append(jnp.array(t))
                self.sample_states.append(tmp)
            # self.sample_states2 = self.sample_states[:2000]
        else:
            self.sample_states = sample_states
        self.init_ents = init_ents

        self.number_samples = 0 if self.sample_states==None else len(sample_states)

        if L == 4 and self.random_init_state == InitialState.RandomState:
            self.h_space_size = 10 #6
        elif self.random_init_state == InitialState.RandomState:
            assert False, "Change Hilbert space dimension when training on random states with symmetries"


        if library == Library.QuSpin:
            raise NotImplementedError()
            self.Hamiltonian = QuspinH
            # self.nn_state = np.zeros((6, 2), dtype=dtype) # state given to neural net agent (has to be real) 2**L
            self.nn_state = np.zeros((2**L, 2), dtype=dtype)
        elif library == Library.TeNPy:
            raise NotImplementedError()
            self.Hamiltonian = TenpyH
        elif library == Library.TN:
            self.Hamiltonian = TNH

        self.goal = 1.0 # goal state for multi-task RL

        self.H_init = self.Hamiltonian(L=self.L, D=self.D, **initial_state_params)
        self.initial_state = self.H_init.ground_state()

        self.H_target = self.Hamiltonian(L=self.L, D=self.D, **final_state_params)
        self.target_state = self.H_target.ground_state()

        self.state = copy.deepcopy(self.initial_state)

        h_max = 1.0
        Jx, Jy, Jz, gx, gy, gz = {'Jx': h_max}, {'Jy': h_max}, {'Jz': h_max}, {'gx': h_max}, {'gy': h_max}, {'gz': h_max}
        nJx, nJy, nJz, ngx, ngy, ngz = {'Jx': -h_max}, {'Jy': -h_max}, {'Jz': -h_max}, {'gx': -h_max}, {'gy': -h_max}, {'gz': -h_max}
        if continuous:
            if n_actions == 3: self.action_dicts = [Jz, gx, gz]
            elif n_actions == 4: self.action_dicts = [Jz, gx, gz, gy]
            elif n_actions == 6: self.action_dicts = [Jz, Jx, Jy, gx, gz, gy]
            else: assert False, "Number of actions not supported."
        else:
            if n_actions == 6: self.action_dicts = [gx, ngx, gy, ngy, gz, ngz]
            # elif n_actions == 6: self.action_dicts = [Jz, nJz, gx, ngx, gz, ngz]
            # elif n_actions == 8: self.action_dicts = [Jz, nJz, gx, ngx, gz, ngz, gy, ngy]
            elif n_actions == 12: self.action_dicts = [gx, ngx, gy, ngy, gz, ngz, Jx, nJx, Jy, nJy, Jz, nJz]
            elif n_actions == 7: self.action_dicts = [gy, gz, ngz, Jx, nJx, Jy, nJy] # CS B
            # elif n_actions == 8: self.action_dicts = [gx, ngx, gy, ngz, Jy, nJy, Jz, nJz] # AFM
            # elif n_actions == 10: self.action_dicts = [gx, ngx, gy, ngy, Jx, nJx, Jy, nJy, Jz, nJz] # excited
            else: assert False, "Number of actions not supported."

        self.n_actions = len(self.action_dicts)  # + 1 # "do nothing" action

        self.counter = 1  # counts number of steps per episode
        self.previous_reward = 0.
        # self.initial_overlap = self.compute_reward()
        # self.previous_reward = self.initial_overlap
        # print("Overlap between initial and target state = {:.8f}".format(self.initial_overlap))

        # self.reset()

    def compute_reward(self):
        """
        Different reward functions:
                        Fidelity
                        Log fidelity (divided by system size)
                        Energy density
        Returns:        float
                        reward
        """
        if self.reward_function == RewardFunction.Fidelity:
            return self.H_target.fidelity_jax(self.state, self.target_state)
        elif self.reward_function == RewardFunction.LogFidelity:
            return self.H_target.log_fidelity_jax(self.state, self.target_state) / self.L
        elif self.reward_function == RewardFunction.EnergyDensity:
            return -self.H_target.energy_density_difference_jax(self.state)  # .item()

    # @profile
    def step(self, action):
        """
        Interface between environment and agent. Performs one step in the environemnt.

        Parameters:
            action: int
                    the index of the respective action
        Returns:
            output: (MPS tensor / ndarray, float, bool, float, float, float)
                    information provided by the environment about its current state:
                    (state, reward, done, truncation error, entropy, goal state)
        """
        # if self.profiling: action.block_until_ready()
        if self.continuous: # continuous action
            action_idx, duration = action
            self.state, trunc_err, entropies = self.H_init.time_evolve_jax(
                self.state, duration, **self.action_dicts[int(action_idx)])
        else:  # discrete action
            # duration = self.step_duration + np.random.normal(0.0, 0.15)
            duration = self.step_duration[0] if action%2 == 0 else self.step_duration[1]
            self.state, trunc_err, entropies = self.H_init.time_evolve_jax(
                self.state, duration, **self.action_dicts[action])
        if self.profiling: [t.block_until_ready() for t in self.state]


        if self.fixed_length_episode:
            done = self.counter == self.n_time_steps
            if self.reward_at_end:
                if done:
                    reward = self.compute_reward()
                    if self.profiling: reward.block_until_ready()
                else:
                    reward = 0.
            else:
                next_reward = self.compute_reward()
                if self.profiling: next_reward.block_until_ready()
                reward = next_reward - self.previous_reward
                self.previous_reward = next_reward
        else:
            reward = self.compute_reward()
            if self.profiling: reward.block_until_ready()
            done = reward > self.threshold
            if not done: done = self.counter == self.n_time_steps

        self.counter = 1 if done else self.counter + 1

        if self.library == Library.TN:
            if self.time_qudit:
                self.time_state = np.zeros(self.n_time_steps, dtype=np.int16)  # reset time
                self.time_state[(self.counter - 1)] = 1  # new time
                return (self.state.copy(), self.time_state.copy()), reward, done, trunc_err, entropies
            else:
                return self.state.copy(), reward, done, trunc_err, entropies, self.goal
        else:
            self.nn_state[:, 0] = self.state.real
            self.nn_state[:, 1] = self.state.imag
            nn_state = self.nn_state.reshape((-1,))

            if self.time_qudit:
                self.time_state = np.zeros(self.n_time_steps, dtype=self.dtype) # reset time
                self.time_state[(self.counter - 1)] = 1.0 # new time
                nn_state = np.concatenate((self.time_state, nn_state))

            return np.copy(nn_state), reward, done, trunc_err, entropies, self.goal
            # return self.state, reward, done, trunc_err, entropies, self.goal

    # @profile
    def reset(self, theta: float=0., phi: float=0., gx: float=0., gz: float=0., idx: int=0, testing: bool=False):
        """
        Resets the environment to its initial values.

        Parameters:
            theta:  float
                    Bloch sphere angle
            phi:    float
                    Bloch sphere angle
            gx:     float
                    transverse field
            gz:     float
                    longitudinal field
            idx:    int
                    for indexing pregenerated initial state list
            testing: bool
                    whether initial state is randomly sampled or assigned
        Returns:
            output: (MPS tensor / ndarray, float)
                    the initial state of the environment and goal state
        """
        self.H_init.half_sites_entropy = 0.
        self.counter = 1
        self.previous_reward = 0.
        # self.previous_reward = self.initial_overlap

        if self.random_initial_state:
            if self.random_init_state == InitialState.ProductState:
                if not testing:
                    # theta = random.uniform(0., np.pi)
                    phi = random.uniform(0., 2*np.pi)
                    theta = np.arccos(1-2*random.uniform(0.0,1.))

                if self.library == Library.TN:
                    self.state = FiniteMPS([spin_state(theta, phi, self.ctype).reshape((1,2,1)) for _ in range(self.L)], center_position=None, canonicalize=True, backend='numpy').tensors
                elif self.library == Library.QuSpin:
                    state = spin_state(theta, phi, self.ctype)
                    for _ in range(self.L-1):
                        state = np.outer(state, spin_state(theta1, phi1, self.ctype))
                    self.state = state.reshape(-1)

            elif self.random_init_state == InitialState.RandomState:
                if self.sample_states == None:
                    if np.random.rand() < 0.25: # take product state with prob 0.25
                        theta = np.arccos(1-2*random.uniform(0.0,1.))
                        phi = random.uniform(0., 2*np.pi)
                        state = spin_state(theta, phi, self.ctype)
                        for _ in range(self.L-1):
                            state = np.outer(state, spin_state(theta1, phi1, self.ctype))
                        self.state = state.reshape(-1)
                    else:
                        if self.random_complex:
                            # phi = np.random.uniform(0., 2*np.pi, size=self.h_space_size)
                            # state = np.random.uniform(0., 1., size=self.h_space_size) * np.exp(1.j*phi)
                            state = np.random.normal(loc=0.0, scale=1.0, size=self.h_space_size) + 1.j * np.random.normal(loc=0.0, scale=1.0, size=self.h_space_size)
                        else:
                            # state = np.random.uniform(0., 1., size=self.h_space_size)
                            state = np.random.normal(loc=0.0, scale=1.0, size=self.h_space_size)
                        state /= np.sqrt(np.abs(state.conj() @ state))
                        # state = spin_basis_1d(L=self.L, pblock=1, zblock=1).project_from(state, sparse=False)
                        state = spin_basis_1d(L=self.L, pblock=1).project_from(state, sparse=False)
                        self.state = state / np.sqrt(np.abs(state.conj() @ state))

                        if self.library == Library.TN:
                            # self.state = FiniteMPS.random([2]*self.L, [2]*(self.L-1), dtype=self.ctype, canonicalize=True, backend='numpy').tensors
                            # state = self.H_init.wavefunction(self.state)
                            # state = self.H_init.state_to_tensor(state)
                            # self.state = FiniteMPS(state, center_position=None, canonicalize=True, backend='numpy').tensors

                            state = self.H_init.state_to_tensor(self.state)
                            self.state = FiniteMPS(state, center_position=None, canonicalize=True, backend='numpy').tensors
                            # self.H_init.half_sites_entropy = self.H_init.entanglement_entropy(self.state, self.L//2)

            elif self.random_init_state == InitialState.GroundState:
                if self.sample_states == None:
                    if not testing:
                        gx = random.uniform(1.0, 1.2)
                        # gz = random.uniform(0.0, 0.5)
                        gz = 0.
                    # gz = 5e-3 if gx < 1.0 else 0.
                    initial_state_params = dict(J=1.0, gx=gx, gz=gz)
                    if self.library == Library.TN:
                        H_init_random = self.Hamiltonian(L=self.L, D=self.D, backend="numpy", **initial_state_params)
                        self.state = H_init_random.ground_state()
                        # self.H_init.half_sites_entropy = self.H_init.entanglement_entropy(self.state, self.L//2)
                    elif self.library == Library.QuSpin:
                        H_init_random = QuspinH(L=self.L, **initial_state_params)
                        self.state = H_init_random.ground_state()
                else:
                    if testing:
                        self.state = self.sample_states[idx]
                    else:
                        idx = random.randint(0, self.number_samples)
                        self.state = self.sample_states[idx]#.tolist()
                        if self.init_ents != None:
                            self.H_init.half_sites_entropy = self.init_ents[0][idx]
                    # self.H_init.half_sites_entropy = self.H_init.entanglement_entropy(self.state, self.L//2)

        else:
            self.state = self.initial_state.copy()

        if self.library == Library.TN:
            if self.time_qudit:
                self.time_state = np.zeros(self.n_time_steps, dtype=np.int16)
                self.time_state[0] = 1
                return (self.state.copy(), self.time_state.copy()), self.goal
            else:
                return self.state.copy(), self.goal
        else:
            self.nn_state[:, 0] = self.state.real
            self.nn_state[:, 1] = self.state.imag
            nn_state = self.nn_state.reshape((-1,))

            if self.time_qudit:
                self.time_state = np.zeros(self.n_time_steps, dtype=self.dtype)
                self.time_state[0] = 1.0
                nn_state = np.concatenate((self.time_state, nn_state))
            return np.copy(nn_state), self.goal
            # return self.state, self.goal

    def render(self):
        pass
