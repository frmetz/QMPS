from functools import partial
import numpy as np

import jax
# import jax.config
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

def initialize_mps_tensor(d_phys, d_bond1, d_bond2, std=1e-3, boundary=False, norm_factor=1.0, dtype=np.float32):
    """ Randomly initialize single MPS tensor """
    if boundary:
        x = np.zeros((1, d_phys, d_bond1), dtype=dtype)
        x[:, :, 0] = 1.
    else:
        x = np.array(d_phys * [np.eye(d_bond1, d_bond2)], dtype=dtype)
    x += np.random.normal(0.0, std, size=x.shape)
    return jnp.array(x) / norm_factor

def initialize_mpo_tensor(d_phys1, d_phys2, d_bond, std=1e-3, boundary=False, norm_factor=1.0, dtype=np.float32):
    """ Randomly initialize single MPO tensor """
    if boundary:
        x = np.zeros((1, d_phys1, d_phys2, d_bond), dtype=dtype)
        x[:, :, :, 0] = 1
    else:
        x = np.array(d_phys1 * [d_phys2 * [np.eye(d_bond)]], dtype=dtype)
    x += np.random.normal(0.0, std, size=x.shape)
    return jnp.array(x) / norm_factor

def initialize_nn(scale, layer_sizes):
    """ Randomly initialize NN """
    return [(scale * np.random.randn(m, n), scale * np.random.randn(n)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]


class QMPS():
    """
    Trainable QMPS ansatz (MPS + NN)

    Parameters:
        sites:          int
                        length of spin chain
        offset:         float
                        constant to add as offset to QMPS output
        scale:          float
                        constant by which QMPS output is scaled
        sign:           float
                        which sign should QMPS output have (after log is applied), only important when training without NN
        nn:             bool
                        whether to append neural network to QMPS output
        goal:           bool
                        true for multi-task RL
    """
    def __init__(self, sites, scale=1.0, sign=1.0, offset=0.0, nn=True, goal = False):
        self.sites = sites
        self.half_sites = sites // 2
        self.scale = scale
        self.sign = sign
        self.offset = offset
        self.use_nn = nn
        self.goal = goal
        self.eps = 1e-9

    def init_random_params(self, n_labels, d_phys, d_bond, n_times, std=1e-3, norm_factor=1.0, uniform=True):
        """ Initialize parameter tensor """
        if uniform:
            params = [initialize_mps_tensor(d_phys, d_bond, d_bond, std=std, boundary=True, norm_factor=norm_factor)]
            params += [initialize_mps_tensor(d_phys, d_bond, d_bond, std=std, norm_factor=norm_factor).transpose((1, 0, 2)) for _ in range(self.half_sites-1)]
            params += [initialize_mps_tensor(n_labels, d_bond, d_bond, std=std, norm_factor=norm_factor).transpose((1, 0, 2))]
            params += [initialize_mps_tensor(d_phys, d_bond, d_bond, std=std, norm_factor=norm_factor).transpose((1, 0, 2)) for _ in range(self.half_sites-1)]
            params += [initialize_mps_tensor(d_phys, d_bond, d_bond, std=std, boundary=True, norm_factor=norm_factor).transpose((2, 1, 0))]
        else:
            D = 2 if 2 <= d_bond else 1
            params = [initialize_mps_tensor(d_phys, D, D, std=std, boundary=True, norm_factor=norm_factor)]

            D2 = D

            for l in range(self.half_sites-2):
                D1 = 2**(l+1) if 2**(l+1) <= d_bond else d_bond
                D2 = 2**(l+2) if 2**(l+2) <= d_bond else d_bond
                params += [initialize_mps_tensor(d_phys, D1, D2, std=std, norm_factor=norm_factor).transpose((1, 0, 2))]

            params += [initialize_mps_tensor(d_phys, D2, d_bond, std=std, norm_factor=norm_factor).transpose((1, 0, 2))]
            params += [initialize_mps_tensor(n_labels, d_bond, d_bond, std=std, norm_factor=norm_factor).transpose((1, 0, 2))]
            params += [initialize_mps_tensor(d_phys, d_bond, D2, std=std, norm_factor=norm_factor).transpose((1, 0, 2))]

            for l in reversed(range(self.half_sites-2)):
                D1 = 2**(l+1) if 2**(l+1) <= d_bond else d_bond
                D2 = 2**(l+2) if 2**(l+2) <= d_bond else d_bond
                params += [initialize_mps_tensor(d_phys, D2, D1, std=std, norm_factor=norm_factor).transpose((1, 0, 2))]

            params += [initialize_mps_tensor(d_phys, D, D, std=std, boundary=True, norm_factor=norm_factor).transpose((2, 1, 0))]
        return params

    @partial(jit, static_argnums=(0,))
    def norm(self, params):
        """ Norm of QMPS """
        # params_mps = params#[0]
        params_mps = params[0]
        left = jnp.einsum('bid,aic->abcd', params_mps[0][0] + 1.j * params_mps[0][1], params_mps[0][0] - 1.j * params_mps[0][1])
        for n, tensor in enumerate(params_mps[1:]):
            left = jnp.einsum('abji, ikd, jkc->abcd', left, tensor[0] + 1.j * tensor[1], tensor[0] - 1.j * tensor[1])
        return jnp.abs(left.reshape(-1))

    @partial(jit, static_argnums=(0,))
    def nn(self, params, inputs):
        """ NN forward pass """
        activations = inputs
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            # activations = outputs
            activations = jnp.tanh(outputs)

        final_w, final_b = params[-1]
        result = jnp.dot(activations, final_w) + final_b
        return result

    @partial(jit, static_argnums=(0,))
    def predict(self, params, state_input, goals=None):
        """
        QMPS forward pass on a batch of input:
        Involves contraction of quantum state MPS and QMPS plus the subsequent NN pass
        """
        Nt, _, d, _ = state_input[0].shape
        params_mps = params[0]
        # nfeat = params_mps[self.half_sites].shape[2]

        right = jnp.dot(state_input[-1].squeeze(3), params_mps[-1][0] + 1.j * params_mps[-1][1]).squeeze(3) #(Nt, X, d) (D, d, 1)-> Nt, X, D
        for n, tensor in enumerate(reversed(params_mps[self.half_sites+1:-1])):
            # (Nt, X, D) (D, d, D) (Nt, X, d, X)
            Dr = tensor[0].shape[0]
            Xr = state_input[-(n+2)].shape[1]
            right = jnp.dot(right, (tensor[0] + 1.j * tensor[1]).transpose(0,2,1)) #(Nt, X, D) (D, d, D) -> (Nt, X, D, d)
            right = jnp.matmul(state_input[-(n+2)].reshape((Nt, d*Xr, -1)), right.reshape((Nt, -1, d*Dr))).reshape((Nt, Xr, d, Dr, d)) # (Nt, X, D, d) (Nt, X, d, X)
            right = jnp.trace(right, axis1=2, axis2=4) # (Nt, X, D)

        left = jnp.dot(state_input[0].squeeze(1).transpose(0,2,1), params_mps[0][0] + 1.j * params_mps[0][1]).squeeze(2) #(Nt, X, d) (1, d, D)-> Nt, X, D
        for n, tensor in enumerate(params_mps[1:self.half_sites]):
            left = jnp.dot(left, (tensor[0] + 1.j * tensor[1]).transpose(1,0,2)) # (Nt, X, d, D)
            _, Xl, _, Dr = left.shape
            left = jnp.matmul(state_input[n+1].transpose(0,3,2,1).reshape((Nt, -1, Xl)), left.reshape((Nt, Xl, d*Dr))).reshape((Nt, -1, d, d, Dr)) # (Nt, X, D, d) (Nt, X, d, X)
            left = jnp.trace(left, axis1=2, axis2=3) # (Nt, X, D)

        center = jnp.matmul(left.transpose(0,2,1), right) # (Nt, D, D)
        z = jnp.dot(center, (params_mps[self.half_sites][0] + 1.j * params_mps[self.half_sites][1]).transpose(0,2,1)) # (Nt, D, D, nl)
        z = jnp.trace(z, axis1=1, axis2=2)


        if self.use_nn:
            out = self.scale*jnp.log(jnp.abs(z)**2 + self.eps) / self.sites + self.offset
            if self.goal:
                return self.nn(params[1], jnp.concatenate((out, goals.reshape((-1,1))), axis=1))
            else:
                return self.nn(params[1], out)
        else:
            # return self.sign * self.scale * jnp.abs(left.reshape(n_samples, -1)) + self.offset
            return self.scale*jnp.log(jnp.abs(z)**2 + self.eps) / self.sites + self.offset
            # return self.scale*2.*jnp.log(jnp.abs(z)) / self.sites + self.offset
            # return self.sign * (scale**2) * jnp.abs(left.reshape(n_samples, -1)) + bias


    @partial(jit, static_argnums=(0,))
    def predict_single(self, params, state_input, goal=None):
        """
        QMPS forward pass on a single input example:
        Involves contraction of quantum state MPS and QMPS plus the subsequent NN pass
        """
        params_mps = params[0]

        left = jnp.einsum('bid,aic->abcd', params_mps[0][0] + 1.j * params_mps[0][1], state_input[0])
        for n, tensor in enumerate(params_mps[1:self.half_sites]):
            left = jnp.einsum('abji, ikd, jkc->abcd', left, tensor[0] + 1.j * tensor[1], state_input[n+1])

        left = jnp.einsum('bcdi, iae->abcde', left, params_mps[self.half_sites][0] + 1.j * params_mps[self.half_sites][1])
        for n, tensor in enumerate(params_mps[self.half_sites+1:]):
            left = jnp.einsum('abcji, ike, jkd->abcde', left, tensor[0] + 1.j * tensor[1], state_input[n+self.half_sites])

        if self.use_nn:
            out = self.scale*jnp.log(jnp.abs(left.reshape(-1))**2 + self.eps) / self.sites + self.offset
            # out = self.scale*2.*jnp.log(jnp.abs(left.reshape(-1))) / self.sites + self.offset
            if self.goal:
                return self.nn(params[1], jnp.append(out, goal))
            else:
                return self.nn(params[1], out)
        else:
            # return self.sign * self.scale * jnp.abs(left.reshape(-1)) + self.offset
            return self.scale*jnp.log(jnp.abs(left.reshape(-1))**2 + self.eps) / self.sites + self.offset
            # return self.scale*2.*jnp.log(jnp.abs(left.reshape(-1))) / self.sites + self.offset
            # return self.sign * (scale**2) * jnp.abs(left.reshape(-1)) + bias

    @partial(jit, static_argnums=(0,))
    def calc_grad(self, g):
        g = 2 * jnp.mean(g, axis=0)
        g_real = jnp.real(g)
        g_imag = -jnp.imag(g)
        return jnp.array([g_real, g_imag])

    @partial(jit, static_argnums=(0,))
    def dtanh(self, x):
        return 1 / jnp.cosh(x)**2

    # dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    # (n,k),(k,m)->(n,m): (Nt, -1, :) * (Nt, :, -1)

    @partial(jit, static_argnums=(0,))
    def value_and_grad_nn(self, params, state_input, labels, actions, goals=None):
        """
        QMPS forward & backward pass on a batch of training data:
        Involves contraction of quantum state MPS and QMPS plus the subsequent NN pass

        Returns:
            (QMPS output, gradients)
        """
        Nt, _, d, _ = state_input[0].shape
        params_mps = params[0]
        nfeat = params_mps[self.half_sites].shape[2]

        right_envs = []
        right = jnp.dot(state_input[-1].squeeze(3), params_mps[-1][0] + 1.j * params_mps[-1][1]).squeeze(3) #(Nt, X, d) (D, d, 1)-> Nt, X, D
        right_envs.append(right)
        for n, tensor in enumerate(reversed(params_mps[self.half_sites+1:-1])):
            # (Nt, X, D) (D, d, D) (Nt, X, d, X)
            Dr = tensor[0].shape[0]
            Xr = state_input[-(n+2)].shape[1]
            right = jnp.dot(right, (tensor[0] + 1.j * tensor[1]).transpose(0,2,1)) #(Nt, X, D) (D, d, D) -> (Nt, X, D, d)
            right = jnp.matmul(state_input[-(n+2)].reshape((Nt, d*Xr, -1)), right.reshape((Nt, -1, d*Dr))).reshape((Nt, Xr, d, Dr, d)) # (Nt, X, D, d) (Nt, X, d, X)
            right = jnp.trace(right, axis1=2, axis2=4) # (Nt, X, D)
            right_envs.append(right)

        left_envs = []
        left = jnp.dot(state_input[0].squeeze(1).transpose(0,2,1), params_mps[0][0] + 1.j * params_mps[0][1]).squeeze(2) #(Nt, X, d) (1, d, D)-> Nt, X, D
        left_envs.append(left) # (Nt, X, D)
        for n, tensor in enumerate(params_mps[1:self.half_sites]):
            left = jnp.dot(left, (tensor[0] + 1.j * tensor[1]).transpose(1,0,2)) # (Nt, X, d, D)
            _, Xl, d, Dr = left.shape
            left = jnp.matmul(state_input[n+1].transpose(0,3,2,1).reshape((Nt, -1, Xl)), left.reshape((Nt, Xl, d*Dr))).reshape((Nt, -1, d, d, Dr)) # (Nt, X, D, d) (Nt, X, d, X)
            left = jnp.trace(left, axis1=2, axis2=3) # (Nt, X, D)
            left_envs.append(left)

        center = jnp.matmul(left.transpose(0,2,1), right) # (Nt, D, D)
        z = jnp.dot(center, (params_mps[self.half_sites][0] + 1.j * params_mps[self.half_sites][1]).transpose(0,2,1)) # (Nt, D, D, nl)
        z = jnp.trace(z, axis1=1, axis2=2)

        y = jnp.real(z * jnp.conjugate(z))
        value = self.scale * jnp.log(y+self.eps) / self.sites + self.offset # change if different activation


        ## NN
        grad_nn = []
        activations, outputs = [], []
        activations.append(value)
        for w, b in params[1][:-1]:
            outputs.append(jnp.dot(activations[-1], w) + b)
            activations.append(jnp.tanh(outputs[-1]))

        final_w, final_b = params[1][-1]
        out = jnp.dot(activations[-1], final_w) + final_b
        n_labels = out.shape[1]
        out = jnp.take_along_axis(out, jnp.expand_dims(actions, axis=(1)).reshape((-1,1)), axis=1).squeeze(axis=1)
        outputs.append(out)

        f = (out - labels)

        # gradients
        # last layer
        delta = jnp.zeros((Nt, n_labels), dtype=np.int32)
        delta_new = jax.ops.index_update(delta, (np.arange(Nt), actions), 1)

        g = jnp.matmul(f.reshape((Nt, 1, 1)), delta_new.reshape((Nt, 1, n_labels))).reshape((Nt, n_labels))
        grad_b = jnp.mean(g, axis=0)
        gg = jnp.matmul(activations[-1].reshape((Nt, -1, 1)), g.reshape((Nt, 1, -1)))
        grad_w = jnp.mean(gg , axis=0)
        grad_nn.append((grad_w, grad_b))

        # middle layer
        for i, output in enumerate(reversed(outputs[:-1])):
            g = jnp.dot(g, params[1][-(i+1)][0].transpose()) * self.dtanh(output)
            grad_b = jnp.mean(g, axis=0)
            gg = jnp.matmul(activations[-(i+2)].reshape((Nt, -1, 1)), g.reshape((Nt, 1, -1)))
            grad_w = jnp.mean(gg, axis=0)
            grad_nn = [(grad_w, grad_b)] + grad_nn

        dvalue = self.scale / (self.sites * (y+self.eps)) # change if different activation
        g = (jnp.dot(g, params[1][0][0].transpose()))
        dloss = ((g * dvalue) * jnp.conjugate(z)).reshape((Nt,1,1,nfeat,1))
        grad = []

        D, _, _ = params_mps[self.half_sites][0].shape
        g = jnp.matmul(center.reshape((Nt, D*D, 1)), dloss.reshape((Nt, 1, -1))).reshape((Nt, D, D, nfeat)).transpose(0,1,3,2)
        grad.append(self.calc_grad(g))

        ddloss = jnp.dot(dloss.reshape((Nt, 1, nfeat)), params_mps[self.half_sites][0] + 1.j * params_mps[self.half_sites][1]).squeeze(1) # (Nt, D, D)

        # optimize half_sites-1
        _, Xl, _, Xr = state_input[self.half_sites-1].shape
        rightt = jnp.matmul(right, ddloss.transpose(0,2,1)) # (Nt, X, D)
        rightt = jnp.matmul(state_input[self.half_sites-1].reshape((Nt, Xl*d, Xr)), rightt).reshape((Nt, Xl, d, -1)) # (Nt, X, d, D)
        g = jnp.matmul(left_envs[-2].transpose(0,2,1), rightt.reshape((Nt, Xl, -1))).reshape((Nt, -1, d, D))
        grad.append(self.calc_grad(g))

        # optimize beginning - half_sites-1
        for n, tensor in enumerate(reversed(params_mps[2:self.half_sites])):
            _, Xl, _, Xr = state_input[self.half_sites-n-2].shape
            rightt = jnp.dot(rightt.reshape((Nt, Xr*d, -1)), (tensor[0] + 1.j * tensor[1]).transpose(0,2,1)).reshape((Nt, Xr, d, -1, d))
            rightt = jnp.trace(rightt, axis1=2, axis2=4) # (Nt, Xl, D)
            rightt = jnp.matmul(state_input[self.half_sites-n-2].reshape((Nt, Xl*d, Xr)), rightt).reshape((Nt, Xl, d, -1)) # Nt, X, d, D
            Dr = rightt.shape[3]
            g = jnp.matmul(left_envs[-(n+3)].transpose(0,2,1), rightt.reshape((Nt, Xl, d*Dr))).reshape((Nt, -1, d, Dr))
            grad.append(self.calc_grad(g))

        # optimize beginning
        Xr = state_input[0].shape[3]
        rightt = jnp.dot(rightt.reshape((Nt, Xr*d, -1)), (params_mps[1][0] + 1.j * params_mps[1][1]).transpose(0,2,1)).reshape((Nt, Xr, d, -1, d))
        rightt = jnp.trace(rightt, axis1=2, axis2=4) # (Nt, Xl, D)
        g = jnp.matmul(state_input[0].reshape((Nt, d, Xr)), rightt).reshape((Nt, 1, d, -1)) # Nt, X, d, D
        grad.append(self.calc_grad(g))

        grad = grad[::-1]

        # optimize half_sites+1
        _, Xl, _, Xr = state_input[self.half_sites].shape
        leftt = jnp.matmul(left, ddloss) # (Nt, X, D)
        leftt = jnp.matmul(leftt.transpose(0,2,1), state_input[self.half_sites].reshape((Nt, Xl, d*Xr))).reshape((Nt, -1, d, Xr)) # (Nt, D, d, X)
        g = jnp.matmul(leftt.reshape((Nt, -1, Xr)), right_envs[-2]).reshape((Nt, D, d, -1))#.transpose(0,1,3,2,4)
        grad.append(self.calc_grad(g))

        # optimize half_sites+2 - end
        for n, tensor in enumerate(params_mps[self.half_sites+1:-2]):
            _, Xl, _, Xr = state_input[self.half_sites+1+n].shape
            leftt = jnp.dot(leftt.transpose(0,2,3,1).reshape((Nt, d*Xl, -1)), (tensor[0] + 1.j * tensor[1]).transpose(1,0,2)).reshape((Nt, d, Xl, d, -1))
            leftt = jnp.trace(leftt, axis1=1, axis2=3) # (Nt, Xl, D)
            leftt = jnp.matmul(leftt.transpose(0,2,1), state_input[self.half_sites+1+n].reshape((Nt, Xl, d*Xr))).reshape((Nt, -1, d, Xr)) # Nt, D, d, X
            Dr = leftt.shape[1]
            g = jnp.matmul(leftt.reshape((Nt, -1, Xr)), right_envs[-(n+3)]).reshape((Nt, Dr, d, -1))#.transpose(0,1,3,2,4)
            grad.append(self.calc_grad(g))

        # optimize end
        Xl = state_input[-1].shape[1]
        leftt = jnp.dot(leftt.transpose(0,2,3,1).reshape((Nt, d*Xl, -1)), (params_mps[-2][0] + 1.j * params_mps[-2][1]).transpose(1,0,2)).reshape((Nt, d, Xl, d, -1))
        leftt = jnp.trace(leftt, axis1=1, axis2=3) # (Nt, Xl, D)
        g = jnp.matmul(leftt.transpose(0,2,1), state_input[-1].reshape((Nt, Xl, d))).reshape((Nt, -1, d, 1)) # Nt, D, d, X
        grad.append(self.calc_grad(g))

        return 0.5*jnp.mean(f**2), [grad, grad_nn]

    @partial(jit, static_argnums=(0,))
    def value_and_grad_nn_goals(self, params, state_input, labels, actions, goals=None):
        Nt, _, d, _ = state_input[0].shape
        params_mps = params[0]
        nfeat = params_mps[self.half_sites].shape[2]

        right_envs = []
        right = jnp.dot(state_input[-1].squeeze(3), params_mps[-1][0] + 1.j * params_mps[-1][1]).squeeze(3) #(Nt, X, d) (D, d, 1)-> Nt, X, D
        right_envs.append(right)
        for n, tensor in enumerate(reversed(params_mps[self.half_sites+1:-1])):
            # (Nt, X, D) (D, d, D) (Nt, X, d, X)
            Dr = tensor[0].shape[0]
            Xr = state_input[-(n+2)].shape[1]
            right = jnp.dot(right, (tensor[0] + 1.j * tensor[1]).transpose(0,2,1)) #(Nt, X, D) (D, d, D) -> (Nt, X, D, d)
            right = jnp.matmul(state_input[-(n+2)].reshape((Nt, d*Xr, -1)), right.reshape((Nt, -1, d*Dr))).reshape((Nt, Xr, d, Dr, d)) # (Nt, X, D, d) (Nt, X, d, X)
            right = jnp.trace(right, axis1=2, axis2=4) # (Nt, X, D)
            right_envs.append(right)

        left_envs = []
        left = jnp.dot(state_input[0].squeeze(1).transpose(0,2,1), params_mps[0][0] + 1.j * params_mps[0][1]).squeeze(2) #(Nt, X, d) (1, d, D)-> Nt, X, D
        left_envs.append(left) # (Nt, X, D)
        for n, tensor in enumerate(params_mps[1:self.half_sites]):
            left = jnp.dot(left, (tensor[0] + 1.j * tensor[1]).transpose(1,0,2)) # (Nt, X, d, D)
            _, Xl, d, Dr = left.shape
            left = jnp.matmul(state_input[n+1].transpose(0,3,2,1).reshape((Nt, -1, Xl)), left.reshape((Nt, Xl, d*Dr))).reshape((Nt, -1, d, d, Dr)) # (Nt, X, D, d) (Nt, X, d, X)
            left = jnp.trace(left, axis1=2, axis2=3) # (Nt, X, D)
            left_envs.append(left)

        center = jnp.matmul(left.transpose(0,2,1), right) # (Nt, D, D)
        z = jnp.dot(center, (params_mps[self.half_sites][0] + 1.j * params_mps[self.half_sites][1]).transpose(0,2,1)) # (Nt, D, D, nl)
        z = jnp.trace(z, axis1=1, axis2=2)

        y = jnp.real(z * jnp.conjugate(z))
        value = self.scale * jnp.log(y+self.eps) / self.sites + self.offset # change if different activation

        value2 = jnp.concatenate((value, goals.reshape((-1,1))), axis=1)

        ## NN
        grad_nn = []
        activations, outputs = [], []
        activations.append(value2)
        for w, b in params[1][:-1]:
            outputs.append(jnp.dot(activations[-1], w) + b)
            activations.append(jnp.tanh(outputs[-1]))

        final_w, final_b = params[1][-1]
        out = jnp.dot(activations[-1], final_w) + final_b
        n_labels = out.shape[1]
        out = jnp.take_along_axis(out, jnp.expand_dims(actions, axis=(1)).reshape((-1,1)), axis=1).squeeze(axis=1)
        outputs.append(out)

        f = (out - labels)

        # gradients
        # last layer
        delta = jnp.zeros((Nt, n_labels), dtype=np.int32)
        delta_new = jax.ops.index_update(delta, (np.arange(Nt), actions), 1)

        g = jnp.matmul(f.reshape((Nt, 1, 1)), delta_new.reshape((Nt, 1, n_labels))).reshape((Nt, n_labels))
        grad_b = jnp.mean(g, axis=0)
        gg = jnp.matmul(activations[-1].reshape((Nt, -1, 1)), g.reshape((Nt, 1, -1)))
        grad_w = jnp.mean(gg , axis=0)
        grad_nn.append((grad_w, grad_b))

        # middle layer
        for i, output in enumerate(reversed(outputs[:-1])):
            g = jnp.dot(g, params[1][-(i+1)][0].transpose()) * self.dtanh(output)
            grad_b = jnp.mean(g, axis=0)
            gg = jnp.matmul(activations[-(i+2)].reshape((Nt, -1, 1)), g.reshape((Nt, 1, -1)))
            grad_w = jnp.mean(gg, axis=0)
            grad_nn = [(grad_w, grad_b)] + grad_nn


        dvalue = self.scale / (self.sites * (y+self.eps)) # change if different activation
        g = (jnp.dot(g, params[1][0][0][:-1,:].transpose()))
        dloss = ((g * dvalue) * jnp.conjugate(z)).reshape((Nt,1,1,nfeat,1))
        grad = []

        D, _, _ = params_mps[self.half_sites][0].shape
        g = jnp.matmul(center.reshape((Nt, D*D, 1)), dloss.reshape((Nt, 1, -1))).reshape((Nt, D, D, nfeat)).transpose(0,1,3,2)
        grad.append(self.calc_grad(g))

        ddloss = jnp.dot(dloss.reshape((Nt, 1, nfeat)), params_mps[self.half_sites][0] + 1.j * params_mps[self.half_sites][1]).squeeze(1) # (Nt, D, D)

        # optimize half_sites-1
        _, Xl, _, Xr = state_input[self.half_sites-1].shape
        rightt = jnp.matmul(right, ddloss.transpose(0,2,1)) # (Nt, X, D)
        rightt = jnp.matmul(state_input[self.half_sites-1].reshape((Nt, Xl*d, Xr)), rightt).reshape((Nt, Xl, d, -1)) # (Nt, X, d, D)
        g = jnp.matmul(left_envs[-2].transpose(0,2,1), rightt.reshape((Nt, Xl, -1))).reshape((Nt, -1, d, D))
        grad.append(self.calc_grad(g))

        # optimize beginning - half_sites-1
        for n, tensor in enumerate(reversed(params_mps[2:self.half_sites])):
            _, Xl, _, Xr = state_input[self.half_sites-n-2].shape
            rightt = jnp.dot(rightt.reshape((Nt, Xr*d, -1)), (tensor[0] + 1.j * tensor[1]).transpose(0,2,1)).reshape((Nt, Xr, d, -1, d))
            rightt = jnp.trace(rightt, axis1=2, axis2=4) # (Nt, Xl, D)
            rightt = jnp.matmul(state_input[self.half_sites-n-2].reshape((Nt, Xl*d, Xr)), rightt).reshape((Nt, Xl, d, -1)) # Nt, X, d, D
            Dr = rightt.shape[3]
            g = jnp.matmul(left_envs[-(n+3)].transpose(0,2,1), rightt.reshape((Nt, Xl, d*Dr))).reshape((Nt, -1, d, Dr))
            grad.append(self.calc_grad(g))

        # optimize beginning
        Xr = state_input[0].shape[3]
        rightt = jnp.dot(rightt.reshape((Nt, Xr*d, -1)), (params_mps[1][0] + 1.j * params_mps[1][1]).transpose(0,2,1)).reshape((Nt, Xr, d, -1, d))
        rightt = jnp.trace(rightt, axis1=2, axis2=4) # (Nt, Xl, D)
        g = jnp.matmul(state_input[0].reshape((Nt, d, Xr)), rightt).reshape((Nt, 1, d, -1)) # Nt, X, d, D
        grad.append(self.calc_grad(g))

        grad = grad[::-1]

        # optimize half_sites+1
        _, Xl, _, Xr = state_input[self.half_sites].shape
        leftt = jnp.matmul(left, ddloss) # (Nt, X, D)
        leftt = jnp.matmul(leftt.transpose(0,2,1), state_input[self.half_sites].reshape((Nt, Xl, d*Xr))).reshape((Nt, -1, d, Xr)) # (Nt, D, d, X)
        g = jnp.matmul(leftt.reshape((Nt, -1, Xr)), right_envs[-2]).reshape((Nt, D, d, -1))#.transpose(0,1,3,2,4)
        grad.append(self.calc_grad(g))

        # optimize half_sites+2 - end
        for n, tensor in enumerate(params_mps[self.half_sites+1:-2]):
            _, Xl, _, Xr = state_input[self.half_sites+1+n].shape
            leftt = jnp.dot(leftt.transpose(0,2,3,1).reshape((Nt, d*Xl, -1)), (tensor[0] + 1.j * tensor[1]).transpose(1,0,2)).reshape((Nt, d, Xl, d, -1))
            leftt = jnp.trace(leftt, axis1=1, axis2=3) # (Nt, Xl, D)
            leftt = jnp.matmul(leftt.transpose(0,2,1), state_input[self.half_sites+1+n].reshape((Nt, Xl, d*Xr))).reshape((Nt, -1, d, Xr)) # Nt, D, d, X
            Dr = leftt.shape[1]
            g = jnp.matmul(leftt.reshape((Nt, -1, Xr)), right_envs[-(n+3)]).reshape((Nt, Dr, d, -1))#.transpose(0,1,3,2,4)
            grad.append(self.calc_grad(g))

        # optimize end
        Xl = state_input[-1].shape[1]
        leftt = jnp.dot(leftt.transpose(0,2,3,1).reshape((Nt, d*Xl, -1)), (params_mps[-2][0] + 1.j * params_mps[-2][1]).transpose(1,0,2)).reshape((Nt, d, Xl, d, -1))
        leftt = jnp.trace(leftt, axis1=1, axis2=3) # (Nt, Xl, D)
        g = jnp.matmul(leftt.transpose(0,2,1), state_input[-1].reshape((Nt, Xl, d))).reshape((Nt, -1, d, 1)) # Nt, D, d, X
        grad.append(self.calc_grad(g))

        return 0.5*jnp.mean(f**2), [grad, grad_nn]
