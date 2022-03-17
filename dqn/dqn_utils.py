import sys
import os
import random
import time
import math
from itertools import count

# import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 100000
import pickle as pkl

import jax
# import jax.config
# jax.config.update("jax_enable_x64", True)

sys.path.append('../environment/')
import env as environment

# @profile
def memory_init(env, agent, eps=1.0):
    """ Fills replay buffer by taking random actions """
    state, goal = env.reset()
    for i in range(agent.replay_buffer.max_size):
        action = agent.get_action(state, goal, eps=eps)
        next_state, reward, done, _, _, goal = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done, goal)
        state = next_state
        if done:
            state, goal = env.reset()


# @profile
def train(env, agent, n_episodes, batch_size, eps_init, eps_final, eps_decay, target_update, update_frequency: int=1, save_model: str="model"):
    """
    Training of the QMPS agent

    Parameters:
        agent           DQN
                        DQN agent based on QMPS ansatz
        n_episodes:     int
                        number of training episodes
        batch_size:     int
                        number of agent-enviornment transitions used for a single update (sampled from replay buffer)
        eps_init:       float
                        initial value of exploration parameter epsilon
        eps_final:      float
                        final value of epsilon
        eps_decay:      float
                        exponential decay rate of epsilon
        target_update:  int
                        number of episodes after which target model parameters are updated with parameters of trained model
        update_frequency: int
                        How often to update network w.r.t. environment steps
        save_model:     string
                        path and filename for saving trained model
    Returns:
        output:         (episode_rewards, epsilon, losses, trunc_errs, norms, entropies, episode_steps, returns)
                        information saved during training
    """
    episode_rewards, episode_steps, returns = [], [], []
    epsilon = []
    losses = []
    trunc_errs = []
    norms = []
    entropies = []
    eps = eps_init

    if env.library == environment.Library.TN:
        print("\nInitial norm: ", agent.norm())

    max_reward = -10000
    start_time = time.time()
    memory_init(env, agent, eps=eps_init)
    # memory_init(env, agent, eps=1.0)
    print("\n\nDone filling replay buffer: --- {:.4f} seconds ---".format(time.time() - start_time))

    j, k = 0, 0 # counts total number of env transitions, total number of network update steps
    start_time = time.time()
    for episode in range(n_episodes):
        state, goal = env.reset()
        init_state = state
        actions = []
        episode_reward = 0

        eps = eps_final + (eps_init - eps_final) * math.exp(-1. * episode / eps_decay)

        i = 0 # counts number of steps per episode
        for step in count():
            j += 1

            action = agent.get_action(state, goal, eps=eps)
            actions.append(action)

            if j % update_frequency == 0:
                loss = agent.update(batch_size)
                losses.append(loss)
                k += 1

                # target network update
                if not agent.polyak:
                    if k % target_update == 0:
                        if env.library == environment.Library.TN:
                            agent.target_params = agent.get_params(agent.opt_state).copy()
                        # else:
                        #     agent.target_model.load_state_dict(agent.model.state_dict())

            next_state, reward, done, trunc_err, entropy, goal = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done, goal)
            episode_reward += reward

            trunc_errs.append(trunc_err)
            i += 1

            if done:
                entropies.append(entropy)
                episode_rewards.append(reward)
                returns.append(episode_reward)
                episode_steps.append(i)
                i = 0
                epsilon.append(eps)
                if reward > max_reward:
                    max_reward = reward
                    best_policy = actions

                if episode % 1 == 0:
                    print("\nFinal reward {}: {:.6f}".format(episode, reward))
                    print("Elapsed time: {:.6f}".format(time.time() - start_time))
                    # if env.library == environment.Library.TN: norms.append(agent.norm())
                    start_time = time.time()
                break

            state = next_state

    if env.library == environment.Library.TN:
        with open(save_model+".pkl", 'wb') as handle:
            pkl.dump(agent.get_params(agent.opt_state), handle)

    # print("Max reward: ", max_reward)
    # print("Best policy: ", np.array(best_policy))

    np.save(save_model+".npy", best_policy)

    episode_rewards.append(max_reward)
    returns.append(max_reward)

    return episode_rewards, epsilon, losses, trunc_errs, norms, entropies, episode_steps, returns

def run_env(env, agent, n_episodes=1, verbose=True, theta=0, phi=0, testing=True):
    """ Runs environment for n_episodes with greedy action selection """
    episode_rewards = []
    actions = []

    for episode in range(n_episodes):
        state, goal = env.reset(theta=theta, phi=phi, testing=testing)
        if verbose: print("\n -----New episode")
        episode_reward = 0
        for step in count():
            action = agent.get_action(state, goal)
            actions.append(action)
            if verbose: print(f"Action {step}: {action}")
            next_state, reward, done, _, _, _ = env.step(action)
            episode_reward += reward
            # print(reward)

            if done:
                episode_rewards.append(episode_reward)
                if verbose: print("Episode {}: {:.4f}".format(episode, episode_reward))
                break
            state = next_state

    return episode_rewards[0], reward, step+1, actions


def plot_rewards(reward, epsilon, string, dir="plots/"):
    """ Final reward curve (and epsilon decay) """
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('episode #')
    ax1.set_ylabel('Reward per episode', color=color)
    ax1.plot(reward[:-1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel(r'$\epsilon$', color=color)
    ax2.plot(epsilon, color=color, linestyle="--")
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_yscale('log')

    plt.title("Final reward = {:.5f}, Best reward = {:.5f}".format(reward[-2], reward[-1]))
    fig.tight_layout()
    plt.savefig(dir+'reward_'+string+'.png', dpi=300)
    #plt.savefig(dir+'reward_'+string+'.pdf')
    plt.close()

def plot_returns(reward, epsilon, string, dir="plots/"):
    """ Return curve (and epsilon decay) """
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('episode #')
    ax1.set_ylabel('Return', color=color)
    ax1.plot(reward[:-1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel(r'$\epsilon$', color=color)
    ax2.plot(epsilon, color=color, linestyle="--")
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_yscale('log')

    plt.title("Final reward = {:.5f}, Best reward = {:.5f}".format(reward[-2], reward[-1]))
    fig.tight_layout()
    plt.savefig(dir+'return_'+string+'.png', dpi=300)
    #plt.savefig(dir+'reward_'+string+'.pdf')
    plt.close()

def plot_loss(loss, string, dir="plots/"):
    """ DQN regression loss curve """
    fig = plt.figure()
    plt.plot(loss)
    plt.yscale('log')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Loss')
    plt.savefig(dir+'loss_'+string+'.png', dpi=300)
    #plt.savefig(dir+'loss_'+string+'.pdf')
    plt.close()

def plot_steps(steps, string, dir="plots/"):
    """ Number of steps per episode """
    fig = plt.figure()
    plt.plot(steps, 'x', markeredgewidth=0.5)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Steps')
    plt.savefig(dir+'steps_'+string+'.png', dpi=300)
    #plt.savefig(dir+'steps_'+string+'.pdf')
    plt.close()

    skip = 40 if len(steps) > 10010 else 20
    steps = np.mean(np.array(steps).reshape(-1, skip), axis=1)
    fig = plt.figure()
    plt.plot(steps, 'x')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Steps')
    plt.savefig(dir+'steps_'+string+'2.png', dpi=300)
    #plt.savefig(dir+'steps_'+string+'2.pdf')
    plt.close()

def plot_truncerr(err, string, dir="plots/"):
    """ Truncation error for each env transition """
    fig = plt.figure()
    plt.plot(err)
    plt.yscale('symlog', linthresh=1e-6)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Truncation error')
    fig.tight_layout()
    plt.savefig(dir+'trunc_'+string+'.png', dpi=300)
    #plt.savefig(dir+'trunc_'+string+'.pdf')
    plt.close()

def plot_norm(norms, string, dir="plots/"):
    """ QMPS norm """
    fig = plt.figure()
    plt.plot(norms)
    plt.yscale('log')
    plt.xlabel('Number of Episodes')
    plt.ylabel('QMPS norm')
    plt.savefig(dir+'norm_'+string+'.png', dpi=300)
    #plt.savefig(dir+'norm_'+string+'.pdf')
    plt.close()
