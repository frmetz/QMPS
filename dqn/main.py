'''
Runs a single QMPS training instance
'''
import os
import sys
import time
import random
import pickle
import itertools
import shutil
import numpy as np

import dqn_utils
import dqn
sys.path.append('../environment/')
import env as environment


# directory where all results are saved
save_name = "test/"

# fix random seed
seed = 30
np.random.seed(seed)
random.seed(seed)

#########################################
# set environment parameters
#########################################

# important parameters (determines name of folder results are saved to)
env_params = dict(
    L = 4, # system size
    n_time_steps = 50, # maximum number of allowed protocol steps
    n_actions = 12, # action set size
    delta_t = 6, # time step size is: 0.5 * np.pi/delta_t, 0.5 * np.pi/(delta_t + 2.5)
    threshold = 0.005 # reward threshold
)

# additional environment parameters
D = 4 # quantum state bond dim
full_env_params = dict(
    initial_state_params = dict(J=1.0, gx=1.0, gz=0.0), # initial ground state parameters
    final_state_params = dict(J=0.0, gx=0.0, gz=1.0), # target ground state parameters
    seed=seed,
    D=D,
    random_initial_state=True, # whether to start from fixed or randomly sampled initial states
    sample_states=None, # precomputed initial states
    init_ents=None, # precomputed initial state entropies
    continue_train = None, # multi-stage training
    reward_function=environment.RewardFunction.LogFidelity,
    random_init_state = environment.InitialState.GroundState,
)
full_env_params.update(env_params)
env = environment.SpinChainEnv(**full_env_params)

########################################################################
# two-stage learning
########################################################################
# env_params2 = {
#     'L': 4,
#     'n_time_steps': 100,
#     'n_actions': 12,
#     'delta_t': 4,
#     'threshold': 0.04,
# }
# full_env_params.update(env_params2)
# full_env_params['sample_states'] = None
# env1 = environment.SpinEnv(**full_env_params)
########################################################################


#########################################
# set agent parameters
#########################################

batch_size = 64 # batch size
agent_params = dict(
    tn = 'mps', # tn architecture: mps or mpo
    nn = True, # whether to also use a neural network
    learning_rate = 1e-4,
    gamma = 0.98, # RL discount factor
    buffer_size = 8000, # Replay buffer size
    n_feat = 32, # MPS output dimension (feature dimension)
    hidden_dim = 100, # NN hidden layer dimension
    uniform = False, # whether MPS bond dimension is uniform or exponentially increasing
    D = 4, # MPS bond dimension
    seed = seed,
)

initial_params = None
agent = dqn.DQNAgent(env, batch_size=batch_size, std=0.5, scale = 1.0, factor = 1.0, **agent_params)


########################################################################
# two-stage learning
########################################################################
# model_name = "model_n_episodes80000_batch_size64_eps_init1.0_eps_final0.01_eps_decay1.0_target_update10_update_frequency1_D4_learning_rate0.0001_gamma0.98_buffer_size8000_nnTrue_n_feat32_hidden_dim100_uniformTrue_scale4.0_factor4.0_seed35_.pkl"
# with open(dir_name+model_name, 'rb') as handle:
#     params1 = pickle.load(handle)
# env.continue_train = (agent.model, params1, env1)
########################################################################


#########################################
# set training parameters
#########################################

training_params = dict(
n_episodes = 4000, # number of training episodes
batch_size = batch_size,
eps_init = 1.0, # initial RL epsilon (action selection)
eps_final = 0.01, # final RL epsilon parameter
eps_decay = 1.0, # determines slope of exponentially decaying curve
target_update = 10, # target network update frequency
update_frequency = 1, # optimization frequency (compared to environment steps)
)


#########################################
# create directory and file names for saving results
#########################################

# environment parameters determine dir name
dir_name = "../results/" + save_name
dir_name = dir_name + "".join("{}_".format(val) for key, val in env_params.items()) + "/"

# save script for reproducability
dir_name2 = dir_name+"_scripts"
os.makedirs(dir_name2, exist_ok=True)
print("Results saved to: ", dir_name)
shutil.copy2('main.py', dir_name2)

# agent and training parameters determine file names
str1 = "".join("{}{}_".format(key, val) for key, val in training_params.items()) + "D%s_" %(D)
str2 = "".join("{}{}_".format(key, val) for key, val in agent_params.items())
save_model = dir_name + "model_" + str1 + str2
print("\nRunning: ", str1 + str2)


#########################################
# Training
#########################################

start_time = time.time()
training_params['eps_decay'] = training_params['n_episodes'] / \
    (training_params['eps_decay'] * 8)
episode_rewards, epsilon, losses, trunc_errs, entropies, episode_steps, returns = dqn_utils.train(
    env, agent, **training_params, save_model=save_model)
# `train` function returns:
# list of final episode rewards
# list of (decaying) epsilon parameter (used for action selection)
# list of DQN regression loss values during training
# list of quantum state MPS bond truncation error during training
# list of quantum state MPS half-chain von Neumann entropies during training
# list of number of episode steps
# list of returns (summed up rewards)
exec_time = time.time() - start_time
print("DONE: --- {:.4f} seconds ---".format(exec_time))

# plot and save training curves
np.save(dir_name + "rewardlist_" + str1 + str2 + ".npy", episode_rewards)
np.save(dir_name + "entropies_" + str1 + str2 + ".npy", entropies)
np.save(dir_name + "truncerr_" + str1 + str2 + ".npy", trunc_errs)
np.save(dir_name + "steps_" + str1 + str2 + ".npy", episode_steps)

# final episode rewards
dqn_utils.plot_rewards(episode_rewards, epsilon, str1 + str2, dir=dir_name)
# episode returns (summed up rewards)
dqn_utils.plot_returns(returns, epsilon, str1 + str2, dir=dir_name)
# Number of episode steps during training
dqn_utils.plot_steps(episode_steps, str1 + str2, dir=dir_name)
# DQN regression loss during training
dqn_utils.plot_loss(losses, str1 + str2, dir=dir_name)
# Quantum state bond truncation error during training
dqn_utils.plot_truncerr(trunc_errs, str1 + str2, dir=dir_name)
