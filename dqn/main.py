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
save_name = "/test/"

# RL and training hyperparameters
profiling = False
seed = 33
np.random.seed(seed)
random.seed(seed)

env_params = dict(
    L = 4,
    n_time_steps = 50,
    n_actions = 12,
    delta_t = 6,
    threshold = 0.01,
    library = environment.Library.TN,
)

# dir_name = "../environment/data/rl_data_ising_param/"
# sample_states = np.load(dir_name+f"__state_list32_16_4000_1.0-1.2.npy", allow_pickle=True)#[:5000]
# init_ents = [np.load(dir_name+f"__ent_list32_16_4000_1.0-1.2.npy", allow_pickle=True)]#[:5000]]
sample_states = None
init_ents = None

D = 16 # quantum state bond dim
full_env_params = dict(
    initial_state_params = dict(J=0.0, gx=0.0, gz=1.0),
    final_state_params = dict(J=1.0, gx=1.0, gz=0.0),
    seed=seed,
    D=D,
    continuous=False,
    time_qudit=False,
    random_initial_state=True,
    reward_at_end=False,
    fixed_length_episode=False,
    profiling=profiling,
    sample_states=sample_states,
    init_ents=init_ents,
    random_complex=True,
    continue_train = None,
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

batch_size = 64
agent_params = dict(
    learning_rate = 1e-4,
    gamma = 0.98,
    buffer_size = 8000,
    n_feat = 16,
    hidden_dim = 100,
    uniform = False,
    scale = 4.0,
    factor = 4.0,
    chiq = 16,
    seed = seed
)
# dir_name = "pretrained/"
# string_name = "model_n_episodes8000_batch_size64_eps_init1.0_eps_final0.01_eps_decay1.0_target_update10_update_frequency1_D16_learning_rate0.0001_gamma0.98_buffer_size8000_hidden_dim80_uniformFalse_scale4.0_factor4.0_seed35_"
# initial_params = dir_name+string_name
initial_params = None
agent = dqn.DQNAgent(env, nn=True, batch_size=batch_size, sign=1.0, offset=0., initial_params=initial_params, profiling=profiling, **agent_params)

########################################################################
# two-stage learning
########################################################################
# model_name = "model_n_episodes80000_batch_size64_eps_init1.0_eps_final0.01_eps_decay1.0_target_update10_update_frequency1_D4_learning_rate0.0001_gamma0.98_buffer_size8000_nnTrue_n_feat32_hidden_dim100_uniformTrue_scale4.0_factor4.0_seed35_.pkl"
# with open(dir_name+model_name, 'rb') as handle:
#     params1 = pickle.load(handle)
# env.continue_train = (agent.model, params1, env1)
########################################################################

training_params = dict(
n_episodes = 4000,
batch_size = batch_size,
eps_init = 1.0,
eps_final = 0.01,
eps_decay = 1.0,
target_update = 10,
update_frequency = 1,
)

# environment parameters determine dir name
dir_name = "../results/test/"
dir_name = dir_name + "".join("{}_".format(val) for key, val in env_params.items()) + save_name  # "/"

# save all modules for reproducability
dir_name2 = dir_name+"_scripts"
os.makedirs(dir_name2, exist_ok=True)
print("Results saved to: ", dir_name)

shutil.copy2('main.py', dir_name2)
shutil.copy2('dqn_utils.py', dir_name2)
shutil.copy2('dqn.py', dir_name2)
shutil.copy2('models.py', dir_name2)
shutil.copy2('../environment/env.py', dir_name2)
shutil.copy2('../environment/tlfi_tn.py', dir_name2)
shutil.copy2('../environment/tlfi_tn_model.py', dir_name2)

# agent and training parameters determine file names
str1 = "".join("{}{}_".format(key, val) for key, val in training_params.items()) + "D%s_" %(D)
str2 = "".join("{}{}_".format(key, val) for key, val in agent_params.items())
save_model = dir_name + "model_" + str1 + str2
print("\nRunning: ", str1 + str2)

# train
start_time = time.time()
training_params['eps_decay'] = training_params['n_episodes'] / \
    (training_params['eps_decay'] * 8)
episode_rewards, epsilon, losses, trunc_errs, norms, entropies, episode_steps, returns = dqn_utils.train(
    env, agent, **training_params, save_model=save_model)
exec_time = time.time() - start_time
print("DONE: --- {:.4f} seconds ---".format(exec_time))

# evaluate optimal policy
final_reward, _, _, _ = dqn_utils.run_env(env, agent, n_episodes=1)

# plot and save training curves
np.save(dir_name + "rewardlist_" + str1 + str2 + ".npy", episode_rewards)
np.save(dir_name + "entropies_" + str1 + str2 + ".npy", entropies)
np.save(dir_name + "truncerr_" + str1 + str2 + ".npy", trunc_errs)
np.save(dir_name + "steps_" + str1 + str2 + ".npy", episode_steps)
dqn_utils.plot_rewards(episode_rewards, epsilon, str1 + str2, dir=dir_name)
dqn_utils.plot_returns(returns, epsilon, str1 + str2, dir=dir_name)
dqn_utils.plot_loss(losses, str1 + str2, dir=dir_name)
dqn_utils.plot_truncerr(trunc_errs, str1 + str2, dir=dir_name)
dqn_utils.plot_steps(episode_steps, str1 + str2, dir=dir_name)
