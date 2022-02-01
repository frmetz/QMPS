'''
Code adapted from: https://github.com/google/TensorNetwork/blob/master/tensornetwork/matrixproductstates/mpo.py
'''

import numpy as np
import tensornetwork as tn
# from tensornetwork.network_components import Node, contract_between
# from tensornetwork.backends import backend_factory
# from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends.abstract_backend import AbstractBackend
from typing import List, Union, Text, Optional, Any, Type
from tensornetwork.matrixproductstates.mpo import FiniteMPO
# from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
# from tensornetwork.matrixproductstates.dmrg import FiniteDMRG


class FiniteTLFI(FiniteMPO):
  """
  Transverse and longitudinal field Ising model
  """

  def __init__(self,
               L = 2,
               J = 1.0,
               gx = 1.0,
               gz = 1.0,
               dtype: Type[np.number] = 'float64',
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = 'TLFI_MPO') -> None:
    """
    Returns the MPO of the finite TLFI model.
    Args:
      L :  Size of lattice
      J :  The Sz*Sz coupling strength between nearest neighbor lattice sites.
      gx:  Transverse magnetic field
      gz:  longitudinal magnetic field
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.
    """
    self.J = J
    self.gx = gx
    self.gz = gz
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([1, -1]).astype(dtype)
    mpo = []

    temp = np.zeros(shape=[1, 3, 2, 2], dtype=dtype)
    temp[0, 0, :, :] = -(self.gz * sigma_z + self.gx * sigma_x)
    temp[0, 1, :, :] = -self.J * sigma_z
    temp[0, 2, 0, 0] = 1.0
    temp[0, 2, 1, 1] = 1.0
    mpo.append(temp)


    for n in range(1, L - 1):
      temp = np.zeros(shape=[3, 3, 2, 2], dtype=dtype)
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      temp[1, 0, :, :] = sigma_z
      temp[2, 0, :, :] = -(self.gz * sigma_z + self.gx * sigma_x)
      temp[2, 1, :, :] = -self.J * sigma_z
      temp[2, 2, 0, 0] = 1.0
      temp[2, 2, 1, 1] = 1.0
      mpo.append(temp)

    temp = np.zeros([3, 1, 2, 2], dtype=dtype)
    temp[0, 0, 0, 0] = 1.0
    temp[0, 0, 1, 1] = 1.0
    temp[1, 0, :, :] = sigma_z
    temp[2, 0, :, :] = -(self.gz * sigma_z + self.gx * sigma_x)
    mpo.append(temp)

    super().__init__(tensors=mpo, backend=backend, name=name)
