from functools import partial
import numpy as np
import jax
# import jax.config
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

class NN(object):
    def __init__(self, params, sites, nn, d, D, feature_dim, batch_size, n_labels, share, scale):
        """
        Common QMPS/QMPO class
        """
        self.params = params if nn else [params[0], []]
        self.use_nn = nn
        self.d = d
        self.D = D
        self.nfeat = feature_dim
        self.Nt = batch_size
        self.n_labels = n_labels
        self.sites = sites
        self.half_sites = sites // 2
        self.eps = 1e-9
        self.scale = scale
        self.offset = 0.

        if share:
            self.predict = self.predict_share
            self.predict_single = self.predict_single_share
            self.predict2 = self.predict2_share
        else:
            self.predict = self.predict_batch
            self.predict_single = self.predict_single_batch
            self.predict2 = self.predict2_batch

    # @partial(jit, static_argnums=(0,), inline=True)
    def nn(self, params, inputs):
        """ NN forward pass """
        activations = inputs
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = jnp.tanh(outputs)
        final_w, final_b = params[-1]
        result = jnp.dot(activations, final_w) + final_b
        return result

    # @profile
    # @partial(jit, static_argnums=(0,), inline=True)
    def nn_forward(self, z, params, actions):
        """ NN forward pass """
        activations, outputs = [], []
        activations.append(z)
        for w, b in params[:-1]:
            outputs.append(jnp.dot(activations[-1], w) + b)
            activations.append(jnp.tanh(outputs[-1]))
        final_w, final_b = params[-1]
        out = jnp.dot(activations[-1], final_w) + final_b
        out = jnp.take_along_axis(out, jnp.expand_dims(actions, axis=(1)).reshape((-1,1)), axis=1).squeeze(axis=1)
        outputs.append(out)
        return activations, outputs

    # @profile
    # @partial(jit, static_argnums=(0,), inline=True)
    def nn_backward(self, g, params, activations, outputs):
        """ NN backward pass """
        grad_nn = []
        grad_b = jnp.mean(g, axis=0)
        gg = jnp.matmul(activations[-1].reshape((self.Nt, -1, 1)), g.reshape((self.Nt, 1, -1)))
        grad_w = jnp.mean(gg , axis=0)
        grad_nn.append((grad_w, grad_b))
        # middle layer
        for i, output in enumerate(reversed(outputs[:-1])):
            g = jnp.dot(g, params[-(i+1)][0].transpose()) * self.dtanh(output)
            grad_b = jnp.mean(g, axis=0)
            gg = jnp.matmul(activations[-(i+2)].reshape((self.Nt, -1, 1)), g.reshape((self.Nt, 1, -1)))
            grad_w = jnp.mean(gg, axis=0)
            grad_nn = [(grad_w, grad_b)] + grad_nn
        return grad_nn, g

    # @partial(jit, static_argnums=(0,), inline=True)
    def dtanh(self, x):
        return 1 / jnp.cosh(x)**2

# @partial(jit, inline=True)
def fill_diagonal(a, val):
    # assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)

def initialize_mps_tensor(d_phys, d_bond1, d_bond2, std=1e-3, boundary=False, norm_factor=1.0, dtype=np.float32, complex=False):
    """ Randomly initialize single MPS tensor """
    if boundary:
        x = np.zeros((1, d_phys, d_bond1), dtype=dtype)
        if not complex: x[:, :, 0] = 1.
    else:
        x = np.array(d_phys * [np.eye(d_bond1, d_bond2)], dtype=dtype)
        if complex: x = np.array(d_phys * [np.zeros((d_bond1, d_bond2))], dtype=dtype)
    x += np.random.normal(0.0, std, size=x.shape)
    return jnp.array(x) / norm_factor

def initialize_mps_tensor_batch(n_feat, d_phys, d_bond1, d_bond2, std=1e-3, boundary=False, norm_factor=1.0, dtype=np.float32, complex=False):
    """ Randomly initialize single MPS tensor """
    if boundary:
        x = np.zeros((n_feat, 1, d_phys, d_bond1), dtype=dtype)
        if not complex: x[:, :, :, 0] = 1.
    else:
        x = np.array(n_feat * d_phys * [np.eye(d_bond1, d_bond2)], dtype=dtype).reshape((n_feat, d_phys, d_bond1, d_bond2))
        if complex: x = np.array(n_feat * d_phys * [np.zeros((d_bond1, d_bond2))], dtype=dtype).reshape((n_feat, d_phys, d_bond1, d_bond2))
    x += np.random.normal(0.0, std, size=x.shape)
    return jnp.array(x) / norm_factor

def initialize_mpo_tensor(d_phys, d_bond1, d_bond2, std=1e-3, boundary=False, norm_factor=1.0, dtype=np.float32, complex=False):
    """ Randomly initialize single MPO tensor """
    if boundary:
        x = np.zeros((1, d_phys, d_phys, d_bond1), dtype=dtype)
        if not complex: x[:, :, :, 0] = 1 / np.sqrt(2) #/ norm_factor
    else:
        x = np.array(d_phys * [d_phys * [np.eye(d_bond1, d_bond2)]], dtype=dtype) / np.sqrt(2) #/ norm_factor
        if complex: x = np.array(d_phys * [d_phys * [np.zeros((d_bond1, d_bond2))]], dtype=dtype)
    x += np.random.normal(0.0, std, size=x.shape) #/ norm_factor
    return jnp.array(x) / norm_factor

def initialize_nn(scale, layer_sizes):
    """ Randomly initialize NN """
    return [(scale * np.random.randn(m, n), scale * np.random.randn(n)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_random_params_batch(n_labels, d_phys, d_bond, center, d=2, std=1e-3, norm_factor=1.0, uniform=True, complex=False):
    """ Initialize parameter tensor """
    half_sites = center
    if d == 2:
        if uniform:
            params = [initialize_mps_tensor_batch(n_labels, d_phys, d_bond, d_bond, std=std, boundary=True, norm_factor=norm_factor, complex=complex)]
            params += [initialize_mps_tensor_batch(n_labels, d_phys, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((0, 2, 1, 3)) for _ in range(center-1)]
            params += [initialize_mps_tensor_batch(n_labels, d_phys, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((0, 2, 1, 3)) for _ in range(center-1)]
            params += [initialize_mps_tensor_batch(n_labels, d_phys, d_bond, d_bond, std=std, boundary=True, norm_factor=norm_factor, complex=complex).transpose((0, 3, 2, 1))]
        else:
            D = d if d <= d_bond else d_bond
            params = [initialize_mps_tensor_batch(n_labels, d_phys, D, D, std=std, boundary=True, norm_factor=norm_factor, complex=complex)]

            D2 = D
            for l in range(center-2):
                D1 = d**(l+1) if d**(l+1) <= d_bond else d_bond
                D2 = d**(l+2) if d**(l+2) <= d_bond else d_bond
                params += [initialize_mps_tensor_batch(n_labels, d_phys, D1, D2, std=std, norm_factor=norm_factor, complex=complex).transpose((0, 2, 1, 3))]

            params += [initialize_mps_tensor_batch(n_labels, d_phys, D2, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((0, 2, 1, 3))]
            # params += [initialize_mps_tensor(n_labels, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2))]
            params += [initialize_mps_tensor_batch(n_labels, d_phys, d_bond, D2, std=std, norm_factor=norm_factor, complex=complex).transpose((0, 2, 1, 3))]

            for l in reversed(range(center-2)):
                D1 = d**(l+1) if d**(l+1) <= d_bond else d_bond
                D2 = d**(l+2) if d**(l+2) <= d_bond else d_bond
                params += [initialize_mps_tensor_batch(n_labels, d_phys, D2, D1, std=std, norm_factor=norm_factor, complex=complex).transpose((0, 2, 1, 3))]

            params += [initialize_mps_tensor_batch(n_labels, d_phys, D, D, std=std, boundary=True, norm_factor=norm_factor, complex=complex).transpose((0, 3, 2, 1))]

    return params

def init_random_params(n_labels, d_phys, d_bond, center, d=2, std=1e-3, norm_factor=1.0, uniform=True, complex=False):
    """ Initialize parameter tensor """
    half_sites = center
    if d == 2:
        if uniform:
            params = [initialize_mps_tensor(d_phys, d_bond, d_bond, std=std, boundary=True, norm_factor=norm_factor, complex=complex)]
            params += [initialize_mps_tensor(d_phys, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2)) for _ in range(center-1)]
            params += [initialize_mps_tensor(n_labels, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2))]
            params += [initialize_mps_tensor(d_phys, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2)) for _ in range(center-1)]
            params += [initialize_mps_tensor(d_phys, d_bond, d_bond, std=std, boundary=True, norm_factor=norm_factor, complex=complex).transpose((2, 1, 0))]
        else:
            D = d if d <= d_bond else d_bond
            params = [initialize_mps_tensor(d_phys, D, D, std=std, boundary=True, norm_factor=norm_factor, complex=complex)]

            D2 = D
            for l in range(center-2):
                D1 = d**(l+1) if d**(l+1) <= d_bond else d_bond
                D2 = d**(l+2) if d**(l+2) <= d_bond else d_bond
                params += [initialize_mps_tensor(d_phys, D1, D2, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2))]

            params += [initialize_mps_tensor(d_phys, D2, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2))]
            params += [initialize_mps_tensor(n_labels, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2))]
            params += [initialize_mps_tensor(d_phys, d_bond, D2, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2))]

            for l in reversed(range(center-2)):
                D1 = d**(l+1) if d**(l+1) <= d_bond else d_bond
                D2 = d**(l+2) if d**(l+2) <= d_bond else d_bond
                params += [initialize_mps_tensor(d_phys, D2, D1, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2))]

            params += [initialize_mps_tensor(d_phys, D, D, std=std, boundary=True, norm_factor=norm_factor, complex=complex).transpose((2, 1, 0))]
    elif d == 4:
        if uniform:
            params = [initialize_mpo_tensor(d_phys, d_bond, d_bond, std=std, boundary=True, norm_factor=norm_factor, complex=complex)]
            params += [initialize_mpo_tensor(d_phys, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((2, 0, 1, 3)) for _ in range(half_sites-1)]
            params += [initialize_mps_tensor(n_labels, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2))]
            params += [initialize_mpo_tensor(d_phys, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((2, 0, 1, 3)) for _ in range(half_sites-1)]
            params += [initialize_mpo_tensor(d_phys, d_bond, d_bond, std=std, boundary=True, norm_factor=norm_factor, complex=complex).transpose((3, 2, 1, 0))]
        else:
            D = d if d <= d_bond else d_bond
            params = [initialize_mpo_tensor(d_phys, D, D, std=std, boundary=True, norm_factor=norm_factor, complex=complex)]

            D2 = D
            for l in range(half_sites-2):
                D1 = d**(l+1) if d**(l+1) <= d_bond else d_bond
                D2 = d**(l+2) if d**(l+2) <= d_bond else d_bond
                params += [initialize_mpo_tensor(d_phys, D1, D2, std=std, norm_factor=norm_factor, complex=complex).transpose((2, 0, 1, 3))]

            params += [initialize_mpo_tensor(d_phys, D2, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((2, 0, 1, 3))]
            params += [initialize_mps_tensor(n_labels, d_bond, d_bond, std=std, norm_factor=norm_factor, complex=complex).transpose((1, 0, 2))]
            params += [initialize_mpo_tensor(d_phys, d_bond, D2, std=std, norm_factor=norm_factor, complex=complex).transpose((2, 0, 1, 3))]

            for l in reversed(range(half_sites-2)):
                D1 = d**(l+1) if d**(l+1) <= d_bond else d_bond
                D2 = d**(l+2) if d**(l+2) <= d_bond else d_bond
                params += [initialize_mpo_tensor(d_phys, D2, D1, std=std, norm_factor=norm_factor, complex=complex).transpose((2, 0, 1, 3))]

            params += [initialize_mpo_tensor(d_phys, D, D, std=std, boundary=True, norm_factor=norm_factor, complex=complex).transpose((3, 2, 1, 0))]
    return params

def normalize(params):
    """ Normalize MPO tensor """
    params_mps = params
    new_params = []
    half_sites = len(params_mps) // 2
    for n,tensor in enumerate(params_mps):
        if n == half_sites:
            # new_params.append(tensor[0])
            new_params.append([tensor[0], tensor[1]])
            continue
        X = tensor[0] + 1.j * tensor[1]
        A = np.einsum('abid,acid->abcd', X, np.conj(X))

        Xl, d, d, Xr = X.shape
        psi = A.transpose(0,3,1,2)

        u, s, v = np.linalg.svd(psi, full_matrices=False)

        x = np.zeros((s.shape[0],s.shape[1],s.shape[2],s.shape[2]))
        x[:,:,0,0] = s[:,:,0]
        x[:,:,1,1] = s[:,:,1]

        mean, std = 1. / s.shape[1], 1.0 / s.shape[1]
        a, b = np.random.uniform(mean-std, mean+std, size=(s.shape[0],s.shape[1])), np.random.uniform(mean-std, mean+std, size=(s.shape[0],s.shape[1]))
        x[:,:,0,0] = a #0.25 #1./np.sqrt(2)
        x[:,:,1,1] = b #0.25 #1./np.sqrt(2)

        new = np.einsum('abcj,abji,abid->abcd', u, x, v)#.transpose(0,2,3,1)

        # when defining A = symm(Z)
        # new[:,:,0,0] = np.real(new[:,:,0,0])
        # new[:,:,1,1] = np.real(new[:,:,1,1])
        # new_params.append(new.transpose(0,2,3,1))

        # when defining A = Z Z^dagger
        new[np.abs(new) < 1e-7] = 0
        L = np.linalg.cholesky(new).transpose(0,2,3,1)
        new_params.append([jnp.real(L), jnp.imag(L)])
    return new_params
