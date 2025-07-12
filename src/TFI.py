"""Toy code implementing the transverse-field ising model."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import tree_util, jit

class TFIModel:
    """Class generating the Hamiltonian of the transverse-field Ising model.

    The Hamiltonian reads
    .. math ::
        H = - J \\sum_{i} \\sigma^x_i \\sigma^x_{i+1} - g \\sum_{i} \\sigma^z_i

    Parameters
    ----------
    L : int
        Number of sites.
    J, g : float
        Coupling parameters of the above defined Hamiltonian.

    Attributes
    ----------
    L : int
        Number of sites.
    d : int
        Local dimension (=2 for spin-1/2 of the transverse field ising model)
    sigmax, sigmay, sigmaz, id :
        Local operators, namely the Pauli matrices and identity.
    H_bonds : list of jnp.Array[ndim=4]
        The Hamiltonian written in terms of local 2-site operators, ``H = sum_i H_bonds[i]``.
        Each ``H_bonds[i]`` has (physical) legs (i out, (i+1) out, i in, (i+1) in),
        in short ``i j i* j*``.
    """

    def __init__(self, L, J, g):
        self.L, self.d = L, 2
        self.J, self.g = J, g
        self.sigmax = jnp.array([[0., 1.], [1., 0.]])
        self.sigmay = jnp.array([[0., -1j], [1j, 0.]])
        self.sigmaz = jnp.array([[1., 0.], [0., -1.]])
        self.id = jnp.eye(2)
        self._H_bonds = None
        self._H_mpo = None

    def _init_H_bonds(self):
        """Initialize `H_bonds` hamiltonian. Called by H_bonds."""
        sx, sz, id = self.sigmax, self.sigmaz, self.id
        d = self.d
        H_list = []
        for i in range(self.L - 1):
            gL = gR = 0.5 * self.g
            if i == 0: # first bond
                gL = self.g
            if i + 1 == self.L - 1: # last bond
                gR = self.g
            H_bond = -self.J * jnp.kron(sx, sx) - gL * jnp.kron(sz, id) - gR * jnp.kron(id, sz)
            # H_bond has legs ``i, j, i*, j*``
            H_list.append(jnp.reshape(H_bond, [d, d, d, d]))
        self._H_bonds = H_list

    def _init_H_mpo(self):
        """Initialize the MPO representation of the Hamiltonian."""
        W = jnp.zeros((3, 3, self.d, self.d))

        W = W.at[0, 0].set(self.id)
        W = W.at[0, 1].set(self.sigmax)
        W = W.at[0, 2].set(-self.g * self.sigmaz)
        W = W.at[1, 2].set(-self.J * self.sigmax)
        W = W.at[2, 2].set(self.id)

        self._H_mpo = [W.copy() for _ in range(self.L)]
    
    @property
    def H_bonds(self):
        """Return the Hamiltonian bonds."""
        if self._H_bonds is None:
            self._init_H_bonds()
        return self._H_bonds
    
    @property
    def H_mpo(self):
        """Return the MPO representation of the Hamiltonian."""
        if self._H_mpo is None:
            self._init_H_mpo()
        return self._H_mpo
    
    @jit
    def energy(self, psi):
        """Evaluate energy E = <psi|H|psi> for the given MPS."""
        assert psi.L == self.L
        return jnp.sum(psi.bond_expectation_value(self.H_bonds))
    
    @jit
    def energy_mpo(self, psi):
        """
        Compute the expectation value ⟨mps_bra|MPO|mps_ket⟩ for two MPS and an MPO.
        All should be lists of tensors of the same length.
        Args:
            mps_bra: MPS object (bra, conjugated)
            mps_ket: MPS object (ket)
            mpo: list of MPO tensors (one per site)
        Returns:
            Scalar energy expectation value
        """
        assert psi.L == self.L
        left_vec = jnp.array([1, 0, 0], dtype=psi.Bs[0].dtype)
        contr = left_vec.reshape(1, 3, 1)
        for n in range(self.L):
            M_bra = psi.Bs[n].conj() # vL*, i*, vR*
            M_ket = psi.Bs[n]        # vL , i , vR
            W = self.H_mpo[n]        # wL , wR, i,  i*

            contr = jnp.tensordot(contr, M_bra, axes=([0], [0]))        # wL,  vL, i*, vR*
            contr = jnp.tensordot(contr, W, axes=([0, 2], [0, 2]))      # vR*, wR, i*, i*
            contr = jnp.tensordot(contr, M_ket, axes=([0, 3], [0, 1]))  # vR*, vR

        assert contr.shape == (1, 3, 1)
        return contr[0, 2, 0] # right_vec = jnp.array([0, 0, 1], dtype=psi.Bs[0].dtype)

    def _tree_flatten(self):
        children = (self._H_bonds,)  # arrays / dynamic values
        aux_data = {
            "L": self.L,
            "J": self.J,
            "g": self.g,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls(aux_data["L"], aux_data["J"], aux_data["g"])
        obj._H_bonds, = children
        return obj

tree_util.register_pytree_node(TFIModel,
                               TFIModel._tree_flatten,
                               TFIModel._tree_unflatten)
