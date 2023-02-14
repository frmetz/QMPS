import math
import sys
import os
import random
import time
from itertools import count
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 100000

sys.path.append('../environment/')
import env as environment

# @profile
def memory_init(env, agent, eps=1.0):
    """ Fills replay buffer by taking random actions"""
    state, _ = env.reset()
    for i in range(agent.replay_buffer.max_size):
        action = agent.get_action(state, eps=eps)
        next_state, reward, done, _, _, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()

# @profile
def train(env, agent, n_episodes, batch_size, eps_init, eps_final, eps_decay, target_update, update_frequency=1, save_model="model"):
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
        output:         (episode_rewards, epsilon, losses, trunc_errs, entropies, episode_steps, returns)
                        information saved during training
    """
    trunc_errs, entropies, rewards, returns, episode_steps, epsilon, losses = [], [], [], [], [], [], []

    start_time = time.time()
    memory_init(env, agent, eps=eps_init)
    print("\n\nDone filling replay buffer: --- {:.4f} seconds ---".format(time.time() - start_time))

    n_steps, n_updates = 0, 0
    start_time = time.time()
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0
        eps = eps_final + (eps_init - eps_final) * math.exp(-1. * episode / eps_decay)

        for step in count():
            n_steps += 1
            action = agent.get_action(state, eps=eps)

            if n_steps % update_frequency == 0:
                loss = agent.update(batch_size)
                losses.append(loss)
                n_updates += 1

                # target network update
                if n_updates % target_update == 0:
                    if env.library == environment.Library.TN:
                        agent.target_tensors = agent.tensors.copy()

            next_state, reward, done, trunc_err, entropy, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_return += reward

            trunc_errs.append(trunc_err)
            if done:
                entropies.append(entropy)
                rewards.append(reward)
                returns.append(episode_return)
                episode_steps.append(step+1)
                epsilon.append(eps)

                if episode % 1 == 0:
                    print("\nFinal reward {}: {:.6f}".format(episode, reward))
                    print("Number steps {}: {}".format(episode, step))
                    print("Elapsed time: {:.4f}".format(time.time() - start_time))
                    start_time = time.time()

                break
            state = next_state

    if env.library == environment.Library.TN:
        with open(save_model+".pkl", 'wb') as handle:
            if agent.fixed:
                pkl.dump([agent.model.params[0]] + [agent.get_params(agent.opt_state)], handle)
            else:
                pkl.dump(agent.get_params(agent.opt_state), handle)

    return rewards, epsilon, losses, trunc_errs, entropies, episode_steps, returns

def run_env(env, agent, n_episodes=1, verbose=True, theta=0, phi=0, testing=True):
    """ Runs environment for n_episodes with greedy action selection """
    for episode in range(n_episodes):
        state, _ = env.reset(theta=theta, phi=phi, testing=testing)
        if verbose: print("\n -----New episode")
        episode_return = 0
        for step in count():
            action, _ = agent.get_action(state)
            actions.append(action)
            if verbose: print(f"Action {step}: {action}")
            next_state, reward, done, _, _, _ = env.step(action)
            episode_return += reward

            if done:
                if verbose: print("Episode {}: {:.4f}".format(episode, reward))
                break
            state = next_state

    return episode_return, reward, step+1, actions


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
