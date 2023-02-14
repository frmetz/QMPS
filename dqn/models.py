from functools import partial
import numpy as np

import jax
# import jax.config
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, value_and_grad

from models_utils import *


class QMPO(NN):
    @classmethod
    def eye(cls, sites, d=2, D=4, feature_dim=12, batch_size=64, share=True, uniform=False, std=0, norm_factor=1., nn=True, layer_sizes=[1], nn_scale=0.1, dtype="complex64", scale=1.0):
        """ Trainable QMPO ansatz (MPO + NN) """
        # label position is at center not at zero
        half_sites = sites // 2
        tensors_r = init_random_params(feature_dim, 2, D, center=half_sites, d=4, std=std, norm_factor=norm_factor, uniform=uniform, complex=False)
        tensors_c = init_random_params(feature_dim, 2, D, center=half_sites, d=4, std=std, norm_factor=norm_factor, uniform=uniform, complex=True)

        tensors = [[p1, p2] for p1, p2 in zip(tensors_r, tensors_c)]
        tensors = normalize(tensors)

        tensors = [tensors, initialize_nn(nn_scale, layer_sizes)]
        n_labels = layer_sizes[-1]

        return cls(params=tensors, sites=sites, nn=nn, d=d, D=D, feature_dim=feature_dim, batch_size=batch_size, n_labels=n_labels, share=share, scale=scale)

    @partial(jit, static_argnums=(0,))
    def get_tensors(self, params_mps):
        A = []
        for i,t in enumerate(params_mps):
            if i == self.half_sites:
                A.append(t[0])
                continue
            X = t[0] + 1.j * t[1]
            A.append(jnp.einsum('abid,acid->abcd', X, jnp.conj(X))) # Xl, d, d, Xr
        return A

    @partial(jit, static_argnums=(0,))
    def predict_single_share(self, params, state_input):
        """ Contracts the input and parameter MPS <psi|phi> """
        A = params[0]

        right = jnp.einsum('aijd,bie,cjf->abcdef', A[-1], state_input[-1], jnp.conj(state_input[-1])).squeeze(axis=(3,4,5))
        for n, a in enumerate(reversed(A[self.half_sites+1:-1])):
            right = jnp.einsum('ijk, almi, blj, cmk->abc', right, a, state_input[-(n+2)], jnp.conj(state_input[-(n+2)]), optimize='optimal')

        left = jnp.einsum('aijd,bie,cjf->abcdef', A[0], state_input[0], jnp.conj(state_input[0])).squeeze(axis=(0,1,2))
        for n, a in enumerate(A[1:self.half_sites]):
            left = jnp.einsum('ijk, ilma, jlb, kmc->abc', left, a, state_input[n+1], jnp.conj(state_input[n+1]), optimize='optimal')

        center = jnp.einsum('bjk, cjk->bc', left, right) # Nt, Xr, Xl
        z = jnp.einsum('il, ibl->b', center, A[self.half_sites])
        out = jnp.real(z)
        if self.use_nn:
            return self.nn(params[1], out)
        else:
            return self.scale * out

    # @profile
    @partial(jit, static_argnums=(0,))
    def predict_share(self, params, state_input):
        """ Contracts the input and parameter MPS <psi|phi> over a batch of inputs """
        Nt, _, d, _ = state_input[0].shape
        A = params[0]

        right = jnp.einsum('aijd,gbie,gcjf->abcdefg', A[-1], state_input[-1], jnp.conj(state_input[-1]), optimize='optimal').squeeze(axis=(3,4,5))
        for n, a in enumerate(reversed(A[self.half_sites+1:-1])):
            right = jnp.einsum('ijkd, almi, dblj, dcmk->abcd', right, a, state_input[-(n+2)], jnp.conj(state_input[-(n+2)]), optimize='optimal')

        left = jnp.einsum('aijd,gbie,gcjf->abcdefg', A[0], state_input[0], jnp.conj(state_input[0]), optimize='optimal').squeeze(axis=(0,1,2))
        for n, a in enumerate(A[1:self.half_sites]):
            left = jnp.einsum('ijkd, ilma, djlb, dkmc->abcd', left, a, state_input[n+1], jnp.conj(state_input[n+1]), optimize='optimal')

        center = jnp.einsum('bjka, cjka->abc', left, right) # Nt, Xr, Xl
        z = jnp.einsum('ail, ibl->ab', center, A[self.half_sites], optimize='optimal')
        out = jnp.real(z)

        if self.use_nn:
            return self.nn(params[1], out)
        else:
            return self.scale * out

    # @partial(jit, static_argnums=(0,))
    def calc_grad(self, g, B):
        g = jnp.mean(g, axis=0)
        g = jnp.einsum('bice, bide->bcde', g, B)
        g_real = 2*jnp.real(g)
        g_imag = 2*jnp.imag(g)
        return [g_real, g_imag]

    # @profile
    # @partial(jit, static_argnums=(0,))
    def compute_envs(self, A, state_input):
        right_envs = []
        right = jnp.einsum('aijd,gbie,gcjf->abcdefg', A[-1], state_input[-1], jnp.conj(state_input[-1]), optimize='optimal').squeeze(axis=(3,4,5))
        right_envs.append(right) # (Xl, Dl, Dl*, Nt)
        for n, a in enumerate(reversed(A[self.half_sites+1:-1])):
            right = jnp.einsum('ijkd, almi, dblj, dcmk->abcd', right, a, state_input[-(n+2)], jnp.conj(state_input[-(n+2)]), optimize='optimal')
            right_envs.append(right) # (Xl, Dl, Dl*, Nt)

        left_envs = []
        left = jnp.einsum('aijd,gbie,gcjf->abcdefg', A[0], state_input[0], jnp.conj(state_input[0]), optimize='optimal').squeeze(axis=(0,1,2))
        left_envs.append(left) # (Nt, X, D)
        for n, a in enumerate(A[1:self.half_sites]):
            left = jnp.einsum('ijkd, ilma, djlb, dkmc->abcd', left, a, state_input[n+1], jnp.conj(state_input[n+1]), optimize='optimal')
            left_envs.append(left) # (Xr, Dr, Dr*, Nt)

        return left_envs, right_envs

    # @profile
    # @partial(jit, static_argnums=(0,6,7,8), donate_argnums=4)
    def mpo_grad_step(self, A, params, state_input, env, env2, mode, mode2=None, mode3=None):
        if mode == 'l':
            if mode2 == 'center':
                env = jnp.einsum('icda, abi->abcd', env, A, optimize='optimal') # Nt, D, X, X*
            else:
                env = jnp.einsum('akcidj, bijk->abcd', env, A, optimize='optimal') # Nt, D, X, X*
            if mode3 != 'end':
                env = jnp.einsum('abij, acdi, aefj->abcdef', env, state_input, jnp.conj(state_input), optimize='optimal')
                g = jnp.einsum('bija, aeicjd->abcde', env2, env, optimize='optimal')
            else:
                g = jnp.einsum('afij, abdi, acej->abcdef', env, state_input, jnp.conj(state_input), optimize='optimal').squeeze(1)
        elif mode == 'r':
            if mode2 == 'center':
                env = jnp.einsum('icda, aib->abcd', env, A, optimize='optimal') # Nt, D, X, X*
            else:
                env = jnp.einsum('akcidj, kijb->abcd', env, A, optimize='optimal') # Nt, D, X, X*
            if mode3 != 'end':
                env = jnp.einsum('abij, aidc, ajfe->abcdef', env, state_input, jnp.conj(state_input), optimize='optimal')
                g = jnp.einsum('eija, abicjd->abcde', env2, env, optimize='optimal')
            else:
                g = jnp.einsum('abij, aidc, ajfe->acbdfe', env, state_input, jnp.conj(state_input), optimize='optimal').squeeze(1)

        B = params[0] + 1.j * params[1]
        return self.calc_grad(g, B), env


    # @profile
    # @partial(jit, static_argnums=(0,))
    def mpo_grad(self, dloss, center, params_mps, tensors, state_input, left_envs, right_envs):
        grad = []
        D, _, _ = params_mps[self.half_sites][0].shape

        g = jnp.matmul(center.reshape((self.Nt, D*D, 1)), dloss.reshape((self.Nt, 1, -1))).reshape((self.Nt, D, D, -1)).transpose(0,1,3,2)
        g2 = jnp.mean(g, axis=0)
        g_real = jnp.real(g2)
        g_imag = g_real
        grad.append([g_real, g_imag])

        ddloss = jnp.dot(dloss.reshape((self.Nt, 1, -1)), params_mps[self.half_sites][0]).squeeze(1) # (Nt, D, D)

        # optimize half_sites-1
        gg, rightt = self.mpo_grad_step(ddloss, params_mps[self.half_sites-1], state_input[self.half_sites-1], right_envs[-1], left_envs[-2], 'l', mode2='center')
        grad.append(gg)

        # optimize beginning - half_sites-1
        for n, tensor in enumerate(reversed(tensors[2:self.half_sites])):
            gg, rightt = self.mpo_grad_step(tensor, params_mps[self.half_sites-n-2], state_input[self.half_sites-n-2], rightt, left_envs[-(n+3)], 'l')
            grad.append(gg)

        # optimize beginning
        gg, rightt = self.mpo_grad_step(tensors[1], params_mps[0], state_input[0], rightt, left_envs[0], 'l', mode3='end')
        grad.append(gg)

        grad = grad[::-1]

        # optimize half_sites+1
        gg, leftt = self.mpo_grad_step(ddloss, params_mps[self.half_sites+1], state_input[self.half_sites], left_envs[-1], right_envs[-2], 'r', mode2='center')
        grad.append(gg)

        # optimize half_sites+2 - end
        for n, tensor in enumerate(tensors[self.half_sites+1:-2]):
            gg, leftt = self.mpo_grad_step(tensor, params_mps[self.half_sites+n+2], state_input[self.half_sites+1+n], leftt, right_envs[-(n+3)], 'r')
            grad.append(gg)

        # optimize end
        gg, leftt = self.mpo_grad_step(tensors[-2], params_mps[-1], state_input[-1], leftt, left_envs[-1], 'r', mode3='end')
        grad.append(gg)

        return grad

    @partial(jit, static_argnums=(0,))
    def loss_nn(self, params, input, labels, actions):
        preds = self.nn(params, input).reshape((1,-1))
        preds_select = jnp.take_along_axis(preds, jnp.expand_dims(actions.reshape(1), axis=1), axis=1)
        return jnp.mean(0.5 * (preds_select.squeeze() - labels)**2)

    # @profile
    @partial(jit, static_argnums=(0,))
    def value_and_grad(self, params, tensors, state_input, labels, actions):
        """ Computes loss function and gradient """
        Nt, _, d, _ = state_input[0].shape
        params_mps = params[0]
        nfeat = params_mps[self.half_sites][0].shape[1]
        n_labels = self.n_labels

        left_envs, right_envs = self.compute_envs(tensors[0], state_input)

        center = jnp.einsum('bjka, cjka->abc', left_envs[-1], right_envs[-1], optimize='optimal') # Nt, Xr, Xl
        z = jnp.einsum('ail, ibl->ab', center, params_mps[self.half_sites][0], optimize='optimal')
        z = jnp.real(z)

        if self.use_nn:
            activations, outputs = self.nn_forward(z, params[1], actions)
            out = outputs[-1]
        else:
            out = self.scale * z
            out = jnp.take_along_axis(out, jnp.expand_dims(actions, axis=(1)).reshape((-1,1)), axis=1).squeeze(axis=1)

        f = (out - labels)

        delta = jnp.zeros((Nt, n_labels), dtype=np.int32)
        delta_new = jax.ops.index_update(delta, (np.arange(Nt), actions), 1)
        g = jnp.matmul(f.reshape((Nt, 1, 1)), delta_new.reshape((Nt, 1, n_labels))).reshape((Nt, n_labels))

        if self.use_nn:
            grad_nn, g = self.nn_backward(g, params[1], activations, outputs)
            g = (jnp.dot(g, params[1][0][0].transpose()))
        else:
            g = self.scale * g
            grad_nn = []

        grad = self.mpo_grad(g, center, params_mps, tensors[0], state_input, left_envs, right_envs)

        return 0.5*jnp.mean(f**2), [grad, grad_nn]

class QMPS(NN):
    @classmethod
    def eye(cls, sites, d=2, D=4, feature_dim=32, batch_size=64, share=True, uniform=False, std=0, norm_factor=1., nn=True, layer_sizes=[1], nn_scale=0.1, dtype="complex64", scale=1.0):
        """ Trainable QMPS ansatz (MPS + NN) """
        # label position is at center not at zero
        half_sites = sites // 2
        if share:
            tensors_r = init_random_params(feature_dim, 2, D, center=half_sites, d=2, std=std, norm_factor=norm_factor, uniform=uniform, complex=False)
            tensors_c = init_random_params(feature_dim, 2, D, center=half_sites, d=2, std=std, norm_factor=norm_factor, uniform=uniform, complex=True)
        else:
            tensors_r = init_random_params_batch(feature_dim, 2, D, center=half_sites, d=2, std=std, norm_factor=norm_factor, uniform=uniform, complex=False)
            tensors_c = init_random_params_batch(feature_dim, 2, D, center=half_sites, d=2, std=std, norm_factor=norm_factor, uniform=uniform, complex=True)

        tensors = [[p1, p2] for p1, p2 in zip(tensors_r, tensors_c)]

        tensors = [tensors, initialize_nn(nn_scale, layer_sizes)]
        n_labels = layer_sizes[-1]

        return cls(params=tensors, sites=sites, nn=nn, d=d, D=D, feature_dim=feature_dim, batch_size=batch_size, n_labels=n_labels, share=share, scale=scale)

    @partial(jit, static_argnums=(0,))
    def get_tensors(self, params_mps):
        return [t[0] + 1.j * t[1] for t in params_mps]

    @partial(jit, static_argnums=(0,))
    def norm(self, params):
        params_mps = params[0]
        left = jnp.einsum('bid,aic->abcd', params_mps[0][0] + 1.j * params_mps[0][1], params_mps[0][0] - 1.j * params_mps[0][1])
        for n, tensor in enumerate(params_mps[1:]):
            left = jnp.einsum('abji, ikd, jkc->abcd', left, tensor[0] + 1.j * tensor[1], tensor[0] - 1.j * tensor[1])
        return jnp.abs(left.reshape(-1))

    def normalize(self, params_mps):
        params = FiniteMPS(params_mps, center_position=0, canonicalize=True)
        params.position(self.half_sites)
        return params.tensors

    # @partial(jit, static_argnums=(0,), inline=True)
    def predict_share(self, params, state_input):
        X = params[0]

        left = jnp.einsum('jid,zjic->zdc', X[0], state_input[0]) # D, X
        for n, tensor in enumerate(X[1:self.half_sites]):
            left = jnp.einsum('zij, ikd, zjkc->zdc', left, tensor, state_input[n+1], optimize='optimal')

        right = jnp.einsum('dij,zcij->zdc', X[-1], state_input[-1]) # D, X
        for n, tensor in enumerate(reversed(X[self.half_sites+1:-1])):
            right = jnp.einsum('zij, dki, zckj->zdc', right, tensor, state_input[-(n+2)], optimize='optimal')

        center = jnp.einsum('zai, zbi->zab', left, right) # Nt, Dr, Dl
        z = jnp.einsum('zij, ibj->zb', center, X[self.half_sites], optimize='optimal')
        out = self.scale*2.*jnp.log(jnp.abs(z) + self.eps) / self.sites + self.offset

        if self.use_nn:
            return self.nn(params[1], out)
        else:
            return out

    # @partial(jit, static_argnums=(0,), inline=True)
    def predict_batch(self, params, state_input):
        X = params[0]

        left = jnp.einsum('ajid,zjic->zadc', X[0], state_input[0]) # D, X
        for n, tensor in enumerate(X[1:self.half_sites]):
            left = jnp.einsum('zaij, aikd, zjkc->zadc', left, tensor, state_input[n+1], optimize='optimal')

        right = jnp.einsum('adij,zcij->zadc', X[-1], state_input[-1]) # D, X
        for n, tensor in enumerate(reversed(X[self.half_sites:-1])):
            right = jnp.einsum('zaij, adki, zckj->zadc', right, tensor, state_input[-(n+2)], optimize='optimal')

        center = jnp.einsum('zaji, zaji->za', left, right) # Nt, Dr, Dl
        out = self.scale*2.*jnp.log(jnp.abs(center) + self.eps) / self.sites + self.offset

        if self.use_nn:
            return self.nn(params[1], out)
        else:
            return out

    # @partial(jit, static_argnums=(0,), inline=True)
    def predict2_share(self, params, state_input):
        params_mps = params[0]

        X = params_mps[0][0] + 1.j * params_mps[0][1]
        left = jnp.einsum('rid,zsic->zdcrs', X, state_input[0]).squeeze(axis=(3,4)) # D, X
        for n, tensor in enumerate(params_mps[1:self.half_sites]):
            X = tensor[0] + 1.j * tensor[1]
            left = jnp.einsum('zij, ikd, zjkc->zdc', left, X, state_input[n+1], optimize='optimal')

        X = params_mps[-1][0] + 1.j * params_mps[-1][1]
        right = jnp.einsum('dir,zcis->zdcrs', X, state_input[-1]).squeeze(axis=(3,4)) # D, X
        for n, tensor in enumerate(reversed(params_mps[self.half_sites+1:-1])):
            X = tensor[0] + 1.j * tensor[1]
            right = jnp.einsum('zij, dki, zckj->zdc', right, X, state_input[-(n+2)], optimize='optimal')

        center = jnp.einsum('zai, zbi->zab', left, right) # Nt, Dr, Dl
        X = params_mps[self.half_sites][0] + 1.j * params_mps[self.half_sites][1]
        z = jnp.einsum('zij, ibj->zb', center, X, optimize='optimal')
        out = self.scale*jnp.log(jnp.abs(z) ** 2 + self.eps) / self.sites + self.offset
        if self.use_nn:
            return self.nn(params[1], out)
        else:
            return out

    # @partial(jit, static_argnums=(0,), inline=True)
    def predict2_batch(self, params, state_input):
        params_mps = params[0]

        X = params_mps[0][0] + 1.j * params_mps[0][1]
        left = jnp.einsum('arid,zsic->zadcrs', X, state_input[0]).squeeze(axis=(4,5)) # D, X
        for n, tensor in enumerate(params_mps[1:self.half_sites]):
            X = tensor[0] + 1.j * tensor[1]
            left = jnp.einsum('zaij, aikd, zjkc->zadc', left, X, state_input[n+1], optimize='optimal')

        X = params_mps[-1][0] + 1.j * params_mps[-1][1]
        right = jnp.einsum('adir,zcis->zadcrs', X, state_input[-1]).squeeze(axis=(4,5)) # D, X
        for n, tensor in enumerate(reversed(params_mps[self.half_sites:-1])):
            X = tensor[0] + 1.j * tensor[1]
            right = jnp.einsum('zaij, adki, zckj->zadc', right, X, state_input[-(n+2)], optimize='optimal')

        center = jnp.einsum('zaji, zaji->za', left, right) # Nt, Dr, Dl
        out = self.scale*jnp.log(jnp.abs(center) ** 2 + self.eps) / self.sites + self.offset
        if self.use_nn:
            return self.nn(params[1], out)
        else:
            return out

    # @partial(jit, static_argnums=(0,), inline=True)
    def predict_fixed_batch(self, params, tensors, state_input):
        X = tensors

        left = jnp.einsum('ajid,zjic->zadc', X[0], state_input[0]) # D, X
        for n, tensor in enumerate(X[1:self.half_sites]):
            left = jnp.einsum('zaij, aikd, zjkc->zadc', left, tensor, state_input[n+1], optimize='optimal')

        right = jnp.einsum('adij,zcij->zadc', X[-1], state_input[-1]) # D, X
        for n, tensor in enumerate(reversed(X[self.half_sites:-1])):
            right = jnp.einsum('zaij, adki, zckj->zadc', right, tensor, state_input[-(n+2)], optimize='optimal')

        center = jnp.einsum('zaji, zaji->za', left, right) # Nt, Dr, Dl
        out = self.scale*2.*jnp.log(jnp.abs(center) + self.eps) / self.sites + self.offset
        if self.use_nn:
            return self.nn(params, out)
        else:
            return out

    @partial(jit, static_argnums=(0,), inline=True)
    def predict_single_share(self, params, state_input):
        """ Contracts the input and parameter MPS <psi|phi> """
        X  = params[0]

        left = jnp.einsum('jid,jic->dc', X[0], state_input[0]) # D, X
        for n, tensor in enumerate(X[1:self.half_sites]):
            left = jnp.einsum('ij, ikd, jkc->dc', left, tensor, state_input[n+1])

        right = jnp.einsum('dij,cij->dc', X[-1], state_input[-1]) # D, X
        for n, tensor in enumerate(reversed(X[self.half_sites+1:-1])):
            right = jnp.einsum('ij, dki, ckj->dc', right, tensor, state_input[-(n+2)], optimize='optimal')

        center = jnp.einsum('ai, bi->ab', left, right) # Nt, Dr, Dl
        z = jnp.einsum('ij, ibj->b', center, X[self.half_sites], optimize='optimal')
        out = self.scale*2.*jnp.log(jnp.abs(z) + self.eps) / self.sites + self.offset

        if self.use_nn:
            return self.nn(params[1], out)
        else:
            return out

    # @partial(jit, static_argnums=(0,), inline=True)
    def predict_single_batch(self, params, state_input):
        """ Contracts the input and parameter MPS <psi|phi> """
        X  = params[0]

        left = jnp.einsum('ajid,jic->adc', X[0], state_input[0]) # D, X
        for n, tensor in enumerate(X[1:self.half_sites]):
            left = jnp.einsum('aij, aikd, jkc->adc', left, tensor, state_input[n+1])

        right = jnp.einsum('adij,cij->adc', X[-1], state_input[-1]) # D, X
        for n, tensor in enumerate(reversed(X[self.half_sites:-1])):
            right = jnp.einsum('aij, adki, ckj->adc', right, tensor, state_input[-(n+2)], optimize='optimal')

        center = jnp.einsum('aji, aji->a', left, right) # Nt, Dr, Dl
        out = self.scale*2.*jnp.log(jnp.abs(center) + self.eps) / self.sites + self.offset

        if self.use_nn:
            return self.nn(params[1], out)
        else:
            return out

    # @profile
    # @partial(jit, static_argnums=(0,))
    def compute_envs(self, X, state_input):
        left_envs, right_envs = [], []
        left = jnp.einsum('jid,zjic->zdc', X[0], state_input[0]) # D, X
        left_envs.append(left)
        for n, tensor in enumerate(X[1:self.half_sites]):
            left = jnp.einsum('zij, ikd, zjkc->zdc', left, tensor, state_input[n+1], optimize='optimal')
            left_envs.append(left)

        right = jnp.einsum('dij,zcij->zdc', X[-1], state_input[-1]) # D, X
        right_envs.append(right)
        for n, tensor in enumerate(reversed(X[self.half_sites+1:-1])):
            right = jnp.einsum('zij, dki, zckj->zdc', right, tensor, state_input[-(n+2)], optimize='optimal')
            right_envs.append(right)

        return left_envs, right_envs

    # @partial(jit, static_argnums=(0,), inline=True)
    def calc_grad(self, g):
        g = 2 * jnp.mean(g, axis=0)
        g_real = jnp.real(g)
        g_imag = -jnp.imag(g)
        return [g_real, g_imag]

    # @profile
    # @partial(jit, static_argnums=(0,6,7,8), donate_argnums=4)
    def mps_grad_step(self, A, state_input, env, env2, mode, mode2=None, mode3=None):
        if mode == 'l':
            if mode3 != 'end':
                if mode2 == 'center':
                    env = jnp.einsum('zij, zci, zabj->zabc', env, A, state_input, optimize='optimal') # Nt, X, d, D
                else:
                    env = jnp.einsum('zijk, cjk, zabi->zabc', env, A, state_input, optimize='optimal')
                g = jnp.einsum('zai, zibc->zabc', env2, env, optimize='optimal')
            else:
                g = jnp.einsum('zijk, cjk, zabi->zabc', env, A, state_input, optimize='optimal')

        elif mode == 'r':
            if mode3 != 'end':
                if mode2 == 'center':
                    env = jnp.einsum('zij, zic, zjba->zabc', env, A, state_input, optimize='optimal') # Nt, X, d, D
                else:
                    env = jnp.einsum('zijk, kjc, ziba->zabc', env, A, state_input, optimize='optimal')
                g = jnp.einsum('zci, ziba->zabc', env2, env, optimize='optimal')
            else:
                g = jnp.einsum('zijk, kjc, ziba->zcba', env, A, state_input, optimize='optimal')
        return self.calc_grad(g), env

    # @profile
    # @partial(jit, static_argnums=(0,))
    def mps_grad(self, dloss, center, params_mps, state_input, left_envs, right_envs):
        grad = []

        g = jnp.matmul(center.reshape((self.Nt, self.D*self.D, 1)), dloss).reshape((self.Nt, self.D, self.D, -1)).transpose(0,1,3,2)
        grad.append(self.calc_grad(g))

        ddloss = jnp.dot(dloss, params_mps[self.half_sites]).squeeze(1) # (Nt, D, D)

        # optimize half_sites-1
        gg, rightt = self.mps_grad_step(ddloss, state_input[self.half_sites-1], right_envs[-1], left_envs[-2], 'l', mode2='center')
        grad.append(gg)

        # optimize beginning - half_sites-1
        for n, tensor in enumerate(reversed(params_mps[2:self.half_sites])):
            gg, rightt = self.mps_grad_step(tensor, state_input[self.half_sites-n-2], rightt, left_envs[-(n+3)], 'l')
            grad.append(gg)

        # optimize beginning
        gg, rightt = self.mps_grad_step(params_mps[1], state_input[0], rightt, left_envs[0], 'l', mode3='end')
        grad.append(gg)

        grad = grad[::-1]

        # optimize half_sites+1
        gg, leftt = self.mps_grad_step(ddloss, state_input[self.half_sites], left_envs[-1], right_envs[-2], 'r', mode2='center')
        grad.append(gg)

        # optimize half_sites+2 - end
        for n, tensor in enumerate(params_mps[self.half_sites+1:-2]):
            gg, leftt = self.mps_grad_step(tensor, state_input[self.half_sites+1+n], leftt, right_envs[-(n+3)], 'r')
            grad.append(gg)

        # optimize end
        gg, leftt = self.mps_grad_step(params_mps[-2], state_input[-1], leftt, left_envs[-1], 'r', mode3='end')
        grad.append(gg)

        return grad

    # @profile
    @partial(jit, static_argnums=(0,))
    def value_and_grad(self, params, tensors, state_input, labels, actions):
        """ Computes loss function and gradient """
        params_mps = tensors[0]

        left_envs, right_envs = self.compute_envs(params_mps, state_input) # Nt, D, X

        center = jnp.einsum('zai, zbi->zab', left_envs[-1], right_envs[-1], optimize='optimal') # Nt, Dr, Dl
        z = jnp.einsum('zij, ibj->zb', center, params_mps[self.half_sites], optimize='optimal')
        y = jnp.real(z * jnp.conjugate(z))
        value = self.scale * jnp.log(y+self.eps) / self.sites + self.offset # change if different activation

        if self.use_nn:
            activations, outputs = self.nn_forward(value, params[1], actions)
            out = outputs[-1]
        else:
            out = value
            out = jnp.take_along_axis(out, jnp.expand_dims(actions, axis=(1)).reshape((-1,1)), axis=1).squeeze(axis=1)

        f = (out - labels)

        delta = jnp.zeros((self.Nt, self.n_labels), dtype=np.int32)
        delta_new = delta.at[np.arange(self.Nt), actions].set(1)

        g = jnp.matmul(f.reshape((self.Nt, 1, 1)), delta_new.reshape((self.Nt, 1, self.n_labels))).reshape((self.Nt, self.n_labels))

        if self.use_nn:
            grad_nn, g = self.nn_backward(g, params[1], activations, outputs)
            g = (jnp.dot(g, params[1][0][0].transpose()))
        else:
            grad_nn = []

        dvalue = self.scale / (self.sites * (y+self.eps)) # change if different activation
        dloss = ((g * dvalue) * jnp.conjugate(z)).reshape((self.Nt, 1, -1))

        grad = self.mps_grad(dloss, center, params_mps, state_input, left_envs, right_envs)

        return 0.5*jnp.mean(f**2), [grad, grad_nn]
