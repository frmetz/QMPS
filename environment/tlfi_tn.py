'''
Some code has been adapted from: https://github.com/google/TensorNetwork/tree/master/tensornetwork/matrixproductstates
'''

import time
import copy
from functools import partial
import random
import pickle as pkl
import numpy as np
import scipy
from scipy.linalg import expm

import jax
# from jax.config import config
# config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

import tensornetwork as tn
from tensornetwork.network_components import Node
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
from tensornetwork.matrixproductstates.dmrg import FiniteDMRG
from tensornetwork.linalg.node_linalg import conj
import tensornetwork.ncon_interface as ncon

from tlfi_tn_model import FiniteTLFI

tn.set_default_backend("numpy")

class TNH():
    """
    TensorNetwork based spin chain simulation class

    Parameters:
        L:                  int
                            Number of spins (length of the chain)
        D:                  int
                            quantum state bond dimension
        J:                  float
                            spin exchange interaction (zz)
        gx:                 float
                            transverse field
        gz:                 float
                            longitudinal field
        dtype:              str
    """
    def __init__(self, L: int=2, D: int=1, J: float=0.0, gx: float=0.0, gz: float=1.0, dtype: str="complex64", backend: str="jax"):

        self.L = L
        self.D = D
        self.backend = backend
        self.dtype = dtype
        self.J = J
        self.gx = gx
        self.gz = gz

        self.H = FiniteTLFI(L=L, J=J, gx=gx, gz=gz, dtype=self.dtype, backend=self.backend)
        self.H_numpy = FiniteTLFI(L=L, J=J, gx=gx, gz=gz, dtype=self.dtype, backend="numpy")

        self.ground_state_energy = 0
        self.half_sites_entropy = 0.

        self.sz = jnp.array([[1.+0.j,0],[0,-1]], dtype=dtype)
        self.sx = jnp.array([[0.+0.j,1],[1,0]], dtype=dtype)
        self.sy = jnp.array([[0.,-1j],[1j,0]], dtype=dtype)


    def gs(self, init_state=None, verbose=False):
        """
        Performs DMRG to obtain ground state

        Parameters:
            init_state      ndarray
                            initial state for DMRG
        Returns:
            ground state:   FiniteMPS
        """
        if init_state == None:
            mps = FiniteMPS.random([2] * self.L, [self.D] * (self.L - 1), dtype=self.dtype, backend="numpy")
        else:
            mps = FiniteMPS(init_state, canonicalize=True, backend="numpy")
        dmrg = FiniteDMRG(mps, self.H_numpy)
        E = dmrg.run_two_site(max_bond_dim=self.D, num_sweeps=200, precision=1E-10, num_krylov_vecs=10, verbose=verbose, delta=1E-10, tol=1E-10, ndiag=10)
        self.ground_state_energy = np.real(E)
        print("\n")
        return mps

    def ground_state(self):
        """
        Performs DMRG to obtain ground state
        Returns:
            ground state:   list(ndarray)
        """
        return self.gs().tensors

    @partial(jit, static_argnums=(0,))
    def state_to_tensor_jax(self, v):
        """
        Converts wavefunction to MPS

        Parameters:
            v       ndarray
                    wavefunction
        Returns:
            mps:    list(ndarray)
        """
        tensors = []
        d = 2
        Dl = 1
        for i in range(self.L):
            psi = v.reshape(d*Dl,-1)
            u, s, v = jnp.linalg.svd(psi, full_matrices=False)
            # truncation
            s_trunc = s[self.D:]
            s = s[0:self.D]

            s /= jnp.linalg.norm(s)
            u = u[:, 0:len(s)]
            v = v[0:len(s), :]
            tensors.append(jnp.reshape(u,(Dl, d, len(s))))
            v = jnp.dot(jnp.diag(s), v)
            Dl, _ = v.shape
        return tensors

    def reduced_density_matrix(self, mps, pos: int, sites: int=1):
        """
        Computes reduced density matrix from MPS at a given position

        Parameters:
            mps             FiniteMPS
            pos             int
                            position at which to calculate rdm
            sites           int
                            whether to calculate rdm of a single or two spins
        Returns:
            rdm:            ndarray
        """
        # mps = FiniteMPS(state, canonicalize=True, backend=self.backend)
        if pos != mps.center_position:
            psi1 = self.position(mps, pos, D=self.D).tensors
        else:
            psi1 = mps.tensors
        # print(np.abs(psi1.check_orthonormality('l', pos-1)))
        # print(np.abs(psi1.check_orthonormality('r', pos+2)))
        if sites==1:
            res = tn.ncon([psi1[pos], jnp.conj(psi1[pos])],[[1,-1,2],[1,-2,2]], backend="jax")
        elif sites==2:
            res = tn.ncon([psi1[pos], psi1[pos+1]],[[-1,-2,1],[1,-3,-4]], backend="jax") # Dl, d, d, Dr
            res = tn.ncon([res, jnp.conj(res)],[[1,-1,-2,2],[1,-3,-4,2]], backend="jax")
        else: assert False, "Only single or two-site rdms are supported"

        return res

    def position(self, mps, site: int, normalize: bool=True, D: int=None, max_truncation_err = None):
        """Shifts center_position to new site

        Parameters:
            mps         FiniteMPS
            site        int
                        new center_position site
            D:          int
                        quantum state bond dimension
            normalize:  bool
                        whether to normalize matrices during SVD
        Returns:
          mps:          FiniteMPS
        """

        #nothing to do
        if site == mps.center_position:
            Z = mps.norm(mps.tensors[mps.center_position])
            if normalize:
                mps.tensors[mps.center_position] /= Z
            return Z

        #shift center_position to the right using QR or SV decomposition
        if site > mps.center_position:
            n = mps.center_position
            for n in range(mps.center_position, site):

                isometry, S, V, _ = mps.svd(mps.tensors[n], 2, D,
                                           max_truncation_err)
                rest = ncon.ncon([mps.backend.diagflat(S), V], [[-1, 1], [1, -2]],
                               backend=mps.backend)

                mps.tensors[n] = isometry
                mps.tensors[n + 1] = ncon.ncon([rest, mps.tensors[n + 1]],
                                                [[-1, 1], [1, -2, -3]],
                                                backend=mps.backend.name)
                Z = mps.norm(mps.tensors[n + 1])
                # for an mps with > O(10) sites one needs to normalize to avoid
                # over or underflow errors; this takes care of the normalization
                if normalize:
                  mps.tensors[n + 1] /= Z

            mps.center_position = site

        #shift center_position to the left using RQ or SV decomposition
        else:
            for n in reversed(range(site + 1, mps.center_position + 1)):
                U, S, isometry, _ = mps.svd(mps.tensors[n], 1, D,
                                           max_truncation_err)
                rest = ncon.ncon([U, mps.backend.diagflat(S)], [[-1, 1], [1, -2]],
                               backend=mps.backend)

                mps.tensors[n] = isometry  #a right-isometric tensor of rank 3
                mps.tensors[n - 1] = ncon.ncon([mps.tensors[n - 1], rest],
                                            [[-1, -2, 1], [1, -3]],
                                            backend=mps.backend.name)
                Z = mps.norm(mps.tensors[n - 1])
                # for an mps with > O(10) sites one needs to normalize to avoid
                # over or underflow errors; this takes care of the normalization
                if normalize:
                  mps.tensors[n - 1] /= Z

            mps.center_position = site
        #return the norm of the last R tensor (useful for checks)
        return mps

    def wavefunction(self, psi):
        """ Converts MPS into wavefunction with basis defined as in QuSpin
        Parameters:
            psi         list(ndarray)
                        MPS to be converted
        Returns:
            wavefunction:  ndarray
        """
        nodes = {}
        for site in range(self.L):
            # nodes[site] = Node(psi.tensors[site], backend=self.backend)
            nodes[site] = Node(psi[site], backend=self.backend)
        for site in range(1, self.L):
            nodes[site][0] ^ nodes[site - 1][2]

        wf = nodes[0] @ nodes[1]
        for i in range(2, self.L):
            wf = wf @ nodes[i]
        return wf.tensor.reshape((-1,))

    @partial(jit, static_argnums=(0,))
    def fidelity_jax(self, psi1, psi2):
        """
        Returns overlap between two MPS states

        Parameters:
            psi1, psi2:     list(ndarray)
                            two MPS
        Returns:
            fidelity:       float
        """
        res = jnp.squeeze(tn.ncon([psi1[0], jnp.conj(psi2[0])],[[-1,1,-3],[-2,1,-4]], backend="jax"),axis=(0,1)) # Du, Dm
        for n, tensor in enumerate(psi1[1:]):
            res = tn.ncon([res, tensor, jnp.conj(psi2[n+1])],[[1,2],[1,3,-1],[2,3,-2]], backend="jax")
        return jnp.abs(jnp.squeeze(res)) ** 2

    @partial(jit, static_argnums=(0,))
    def log_fidelity_jax(self, psi1, psi2):
        """
        Returns log of fidelity between two MPS states

        Parameters:
            psi1, psi2:     list(ndarray)
                            two MPS
        Returns:
            fidelity:       float
        """
        fid = self.fidelity_jax(psi1, psi2)
        return jnp.log(fid+1e-20)

    @partial(jit, static_argnums=(0,))
    def energy_jax(self, psi):
        """
        Returns energy (expectation value of Hamiltonian) of given MPS

        Parameters:
            psi:        list(ndarray)
        Returns:
            energy:     float
        """
        res = jnp.squeeze(tn.ncon([self.H.tensors[0], psi[0], jnp.conj(psi[0])],[[-3,-5,1,2],[-1,1,-4],[-2,2,-6]], backend="jax"),axis=(0,1,2)) # Du, Dm, Dl
        for n, tensor in enumerate(self.H.tensors[1:]):
            res = tn.ncon([res, tensor, psi[n+1], jnp.conj(psi[n+1])],[[1,2,4],[2,-2,3,5],[1,3,-1],[4,5,-3]], backend="jax")
        return jnp.real(jnp.squeeze(res))

    @partial(jit, static_argnums=(0,))
    def energy_density_difference_jax(self, psi):
        """
        Returns energy difference with ground state energy normalized to system size

        Parameters:
            psi:        list(ndarray)
        Returns:
            energy diff:  float
        """
        return (self.energy_jax(psi) - self.ground_state_energy) / jnp.abs(self.ground_state_energy)

    # @partial(jit, static_argnums=(0,))
    def entanglement_entropy(self, tensors, site):
        """
        Entanglement entropy between two subsystems split at specific site

        Parameters:
            tensors:      list(ndarray)
                          MPS
            site:         int
                          size of subsystem
        Returns:
            entropy:      float
        """
        for j in range(site):
            result = jnp.einsum('abi,icd->abcd', tensors[j], tensors[j+1]) # (d, d, d, d) * (Dl, d, D) * (D, d, Dr) -> (Dl, d, d, Dr)
            Dl, d, _, Dr = result.shape
            u, s, v = jnp.linalg.svd(result.reshape((Dl*d, Dr*d)), full_matrices=False)

            # truncation
            s_trunc = s[self.D:]
            s = s[0:self.D]
            trunc_err = jnp.linalg.norm(s_trunc) if len(s_trunc)>0 else 1e-10 # truncation error (set to some value in case of no truncation)

            s /= jnp.linalg.norm(s)
            u = u[:, 0:len(s)]
            v = v[0:len(s), :]
            tensors[j] = jnp.reshape(u,(Dl, d, len(s)))
            tensors[j+1] = jnp.reshape(jnp.dot(jnp.diag(s), v), (len(s), d, Dr))
        s2 = s * s
        ent_entropy = -jnp.sum(jax.scipy.special.xlogy(s2, s2))
        return ent_entropy

    def magnetization_expectation(self, psi, direction, staggered=False):
        """
        Returns expectation values of sigma_x, sigma_y, sigma_z

        Parameters:
            psi:            FiniteMPS
            direction:      string
                            either "x", "y", or "z"
            staggered:      bool
                            if true then one computes staggered magnetization instead
        Returns:
            entropy:      float
        """
        if direction == "z":
            op = np.array(self.sz)
        elif direction == "x":
            op = np.array(self.sx)
        elif direction == "y":
            op = np.array(self.sy)
        if staggered:
            ops = [op, -op]*(self.L//2)
        else:
            ops = [op]*self.L
        return np.mean(psi.measure_local_operator(ops, list(range(self.L))))

    def time_evolve_jax(self, tensors, duration: float, Jz: float=0, Jy: float=0, Jx: float=0, gz: float=0, gy: float=0, gx: float=0):
        """
        Time evolution

        Parameters:
            tensors:        list(ndarray)
                            MPS to be time evolved
            duration:       float
                            time step duration delta_t
            gx,gy,gz,Jx,Jy,Jz:   float
                            strength and direction of corresponding "Hamiltonian" to time evolve with
                            Note that only one of those is allowed to be nonzero
        Returns:
            state:          list(ndarray)
                            time evolved state
            err             float
                            truncation error
            entropy         float
                            half-site von Neumann entanglement entropy
        """
        if Jz == Jy == Jx == 0:
            if gx != 0:
                op = -gx * self.sx
            elif gz != 0:
                op = -gz * self.sz
            elif gy != 0:
                op = -gy * self.sy
            state = self.apply_one_site_op_jax(tensors, duration, op)
            # state[0].block_until_ready()
            return state, 0., self.half_sites_entropy
        else:
            if Jz != 0:
                op = -Jz * jnp.kron(self.sz, self.sz)
            elif Jx != 0:
                op = -Jx * jnp.kron(self.sx, self.sx)
            elif Jy != 0:
                op = -Jy * jnp.kron(self.sy, self.sy)
            state, err, entropies = self.apply_two_site_op_jax(tensors, duration, op)
            self.half_sites_entropy = entropies[self.L//2-1]
            # state[0].block_until_ready()
            return state, err, self.half_sites_entropy

    @partial(jit, static_argnums=(0,))
    def apply_one_site_op_jax(self, tensors, duration, op):
        """
        Applies single site gate uniformly to all spins

        Parameters:
            tensors:        list(ndarray)
                            MPS to be time evolved
            duration:       float
                            time step duration delta_t
                            either "x", "y", or "z"
            op:             ndarray
                            single-site operator (Hamiltonian)
        Returns:
            tensors:        list(ndarray)
                            time evolved state
        """
        # print("one site")
        gate = jax.scipy.linalg.expm(-1.j * duration * op)
        for site in range(self.L):
            tensors[site] = tn.ncon([gate, tensors[site]],
                                [[-2, 1], [-1, 1, -3]], backend="jax")
        return tensors

    @partial(jit, static_argnums=(0,))
    def apply_two_site_op_jax(self, tensors, duration, op):
        """
        Applies two-site gate uniformly to all spins

        Parameters:
            tensors:        list(ndarray)
                            MPS to be time evolved
            duration:       float
                            time step duration delta_t
                            either "x", "y", or "z"
            op:             ndarray
                            two-site operator (Hamiltonian)
        Returns:
            tensors:        list(ndarray)
                            time evolved state
        """
        print("two site")
        gate = jax.scipy.linalg.expm(-1.j * duration * op).reshape((2,2,2,2))
        err = 1.
        entropies2 = []
        for j in range(self.L-1):
            result = jnp.einsum('bcjk,aji,ikd->abcd', gate, tensors[j], tensors[j+1]) # (d, d, d, d) * (Dl, d, D) * (D, d, Dr) -> (Dl, d, d, Dr)
            Dl, d, _, Dr = result.shape
            u, s, v = jnp.linalg.svd(result.reshape((Dl*d, Dr*d)), full_matrices=False)

            # truncation
            s_trunc = s[self.D:]
            s = s[0:self.D]
            trunc_err = jnp.sum(s_trunc ** 2) if len(s_trunc)>0 else 0. # truncation error (set to some value in case of no truncation)
            err = err * (1. - 2.*trunc_err)

            s /= jnp.linalg.norm(s)
            u = u[:, 0:len(s)]
            v = v[0:len(s), :]
            tensors[j] = jnp.reshape(u,(Dl, d, len(s)))
            tensors[j+1] = jnp.reshape(jnp.dot(jnp.diag(s), v), (len(s), d, Dr))

            s2 = s * s
            entropies2.append(-jnp.sum(jax.scipy.special.xlogy(s2, s2)))
        # MPS is now in left canonical form

        # bring MPS back into right canonical form
        for n in reversed(range(1, self.L)):
            Dl, d, Dr = tensors[n].shape
            Q, R = jnp.linalg.qr(tensors[n].reshape((Dl, d*Dr)).transpose(), mode='reduced') #technically we need RQ decomposition
            tensors[n] = Q.transpose().reshape((Dl,d,Dr))
            tensors[n-1] = jnp.einsum('abi,ci->abc', tensors[n-1], R)
            Z = jnp.linalg.norm(tensors[n-1])
            tensors[n-1] /= Z

        return tensors, 1.-err, entropies2
