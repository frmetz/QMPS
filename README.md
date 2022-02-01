# Self-Correcting Quantum Many-Body Control using Reinforcement Learning with Tensor Networks

This repository contains the code and results presented in the preprint article [arXiv:2201.11790](https://arxiv.org/abs/2201.11790).

## Summary

We present a novel tensor-network-based Q-learning framework (QMPS) for controlling 1d spin chains. As an exemplary task we consider the problem of ground state preparation and prepare different states of the paradigmatic mixed-field Ising model. In order to reach large system sizes, we simulate the spin chain using matrix product states (MPS). The control problem is solved using reinforcement learning, specifically deep Q-learning. As an ansatz, we employ a hybrid MPS+NN network that is especially designed for controlling a large number of spins/qubits. 

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

[trained_models/](trained_models/) contains the corresponding data of the control studies presented in the paper (including the optimized QMPS parameters).

## Requirements
The code is written in Python and apart from the usual libraries (numpy, scipy, matplotlib) you need to have the following packages installed:

* [JAX](https://github.com/google/jax): For performance enhancenment via just-in-time compilation.
* [TensorNetwork](https://github.com/google/TensorNetwork): For the spin chain simulations.

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
