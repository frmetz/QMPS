# Self-Correcting Quantum Many-Body Control using Reinforcement Learning with Tensor Networks

This repository contains the code and data for the corresponding preprint article [arXiv:2201.11790](https://arxiv.org/abs/2201.11790).

## Summary

We present a novel Q-learning framework (QMPS) specifically designed for controlling 1d spin chains in which the RL agent is represented by a combination of a matrix product state (MPS) and a neural network (NN). The algorithm can be used to find optimal control protocols that prepare a target (ground) state starting from a set of initial states, and as an example we implement the paradigmatic mixed-field Ising model. To reach system sizes which lie beyond exact simulation techniques, we employ matrix product states as a representation for the quantum state and as a trainable machine learning ansatz. The hybrid MPS+NN architecture is then optimized via backpropagation and conventional gradient descent.

## Content

__RL agent__
* [dqn/main.py](dqn/main.py): Script that performs one full instance of training for specified (hyper)parameters and plots/saves the results.
* [dqn/dqn.py](dqn/dqn.py): QMPS agent (essentially a DQN agent where the ansatz is composed of a hybrid NN+MPS network).
* [dqn/dqn_utils.py](dqn/dqn_utils.py): Functions for training & evaluating a QMPS agent.
* [dqn/models.py](dqn/models.py): QMPS ansatz with forward and backward passes.
* [dqn/replay_buffers.py](dqn/replay_buffers.py): DQN replay buffer.

__RL environment__
* [environment/env.py](environment/env.py): Gym-style RL environment for controlling a spin chain
* [environment/tlfi_tn.py](environment/tlfi_tn.py): Functions for simulating a spin chain. In this case the transverse longitudinal field Ising (TLFI) model.
* [environment/tlfi_tn_model.py](environment/tlfi_tn_model.py): MPO implementation of the TLFI Hamiltonian

__Tutorials__
* [dqn/main.ipynb](dqn/main.ipynb): Tutorial demonstrating how to train a QMPS agent.
* [dqn/test_trained_agent.ipynb](dqn/test_trained_agent.ipynb): Tutorial demonstrating how to test a successfully trained agent on different initial states.

[trained_models/](trained_models/) contains the corresponding data of the control studies presented in the paper (including the optimized QMPS parameters).

[figures/](figures/) contains the data used for generating the main plots in the paper and python scripts for reproducing them.

## Requirements
The code is written in Python and apart from the usual libraries (numpy [tested on v1.22.3], scipy [v1.8.0], matplotlib [v3.5.1]) you need to have the following packages installed:

* [JAX](https://github.com/google/jax): For performance enhancenment via just-in-time compilation. (tested on jax/jaxlib v0.3.24 and v0.4.3)
* [TensorNetwork](https://github.com/google/TensorNetwork): For the spin chain simulations. (tested on v0.4.6)

## Run the code
Simply download/clone this repo and run `python main.py` from within the [dqn/](dqn/) folder. This will create a folder `results/` where all results of the training are stored (learning curves as plots and as `.npy` files, trained model parameters as a `.pkl` file).

The run time varies depending on the parameters used. As an example, one full episode of training (including 50 environment and optimization steps) for N=16 spins, a QMPS bond diminesion of 32, a quantum state bond dimension of 16, a feature vector dimension of 32, and a batch size of 64 took 6.5 sec on a Intel Xeon Gold 6230 CPU and 1.8 sec on a NVIDIA Tesla P100 SXM2 GPU.

The demo training script [dqn/main.py](dqn/main.py) shouldn't take longer than ~20min to run.

## Citation

If you use our code/models for your research, consider citing our paper:
```
@misc{metz2022,
      title={Self-Correcting Quantum Many-Body Control using Reinforcement Learning with Tensor Networks}, 
      author={Friederike Metz and Marin Bukov},
      year={2022},
      eprint={2201.11790},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
