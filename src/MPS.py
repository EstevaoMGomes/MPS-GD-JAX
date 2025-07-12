"""Toy code implementing a matrix product state."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import tree_util, jit
from functools import partial

class MPS:
    """Class for a matrix product state.

    We index sites with `i` from 0 to L-1; bond `i` is left of site `i`.
    We *assume* that the state is in right-canonical form.

    Parameters
    ----------
    Bs, Ss:
        Same as attributes.

    Attributes
    ----------
    Bs : list of jnp.Array[ndim=3]
        The 'matrices' in right-canonical form, one for each physical site.
        Each `B[i]` has legs (virtual left, physical, virtual right), in short ``vL i vR``
    Ss : list of jnp.Array[ndim=1]
        The Schmidt values at each of the bonds, ``Ss[i]`` is left of ``Bs[i]``.
    L : int
        Number of sites.
    """

    def __init__(self, Bs, Ss):
        self.Bs = Bs
        self.Ss = Ss #stop_gradient(s) for s in Ss]
        self.L = len(Bs)

    @jit
    def copy(self):
        # return tree_util.tree_map(jnp.array, self)
        return MPS([jnp.array(B) for B in self.Bs], [jnp.array(S) for S in self.Ss])
    
    @jit
    def norm_squared(self):
        """Calculate the norm squared of the MPS, which is the overlap with itself."""
        return overlap(self, self).real
    
    @jit
    def normalize(self):
        center = self.L // 2
        psi = self.copy()
        psi.Bs[center] = psi.Bs[center] / jnp.sqrt(psi.norm_squared())
        return psi

    @jit
    def canonicalize(self):
        """
        Return a new MPS in right-canonical form by a left-to-right SVD sweep.
        Optionally truncate to chi_max and discard singular values < eps.
        Does not modify self.
        """
        psi = self.copy() # Create a copy to avoid modifying self
        chis = psi.get_chi()
        for i in range(psi.L - 1):
            chivC = chis[i]
            j = i + 1
            theta = psi.get_theta2(i)  # vL i j vR

            Ai, Sj, Bj = split_theta(theta, chivC)

            # put back into MPS
            Gi = jnp.tensordot(jnp.diag(psi.Ss[i]**(-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
            psi.Bs[i] = jnp.tensordot(Gi, jnp.diag(Sj), axes=[2, 0])  # vL i [vC], [vC] vC
            psi.Ss[j] = Sj #jax.lax.stop_gradient(Sj)  # vC
            psi.Bs[j] = Bj  # vC j vR
        return psi
    
    @partial(jit, static_argnames=["i"])
    def get_theta1(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs ``vL, i, vR`` (as one of the Bs)."""
        return jnp.tensordot(jnp.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    @partial(jit, static_argnames=["i"])
    def get_theta2(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``."""
        j = i + 1
        return jnp.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR

    def get_chi(self):
        """Return bond dimensions."""
        return [self.Bs[i].shape[2] for i in range(self.L - 1)]

    @jit
    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = jnp.tensordot(op, theta, axes=[1, 1])  # i [i*], vL [i] vR
            result.append(jnp.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return jnp.real(jnp.array(result))

    @jit
    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.L - 1):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = jnp.tensordot(op[i], theta, axes=[[2, 3], [1, 2]])
            # i j [i*] [j*], vL [i] [j] vR
            result.append(jnp.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return jnp.real(jnp.array(result))

    @jit
    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        result = []
        for i in range(1, self.L):
            S = jnp.array(self.Ss[i])
            S = S[S > 1e-30]  # 0*log(0) should give 0; avoid warnings or NaN by discarding small S
            S2 = S * S
            assert abs(jnp.linalg.norm(S) - 1.) < 1.e-14
            result.append(-jnp.sum(S2 * jnp.log(S2)))
        return jnp.array(result)
    
    def _tree_flatten(self):
        children = (self.Bs, self.Ss)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        Bs, Ss = children
        return cls(Bs, Ss)

tree_util.register_pytree_node(MPS,
                               MPS._tree_flatten,
                               MPS._tree_unflatten)

@jit
def overlap(mps_bra, mps_ket):
    """
    Compute the overlap ⟨mps_bra|mps_ket⟩ for two MPS in right-canonical form.
    Both should be lists of tensors of the same length.
    """
    L = len(mps_bra.Bs)
    contr = jnp.ones((1, 1), dtype=mps_bra.Bs[0].dtype)
    for n in range(L):
        # M_ket: (vL, i, vR)
        M_ket = mps_ket.Bs[n]
        contr = jnp.tensordot(contr, M_ket, axes=(1, 0)) # vR* [vR], [vL] j vR contract indices in []
        # M_bra: (vL, i, vR)
        M_bra = mps_bra.Bs[n].conj()
        contr = jnp.tensordot(M_bra, contr, axes=([0, 1], [0, 1])) # [vL*] [j*] vR*, [vR*] [j] vR
    assert contr.shape == (1, 1)
    return contr[0, 0]

def init_spinup_MPS(L: int, chi_max: int, noise: bool = False, eps: float = 1e-4, key=None) -> MPS:
    """
    Create an all-up spin MPS with maximum bond dimension chi_max.
    Optionally add small noise to each tensor.
    
    Args:
        L: int, number of sites
        chi_max: int, maximum bond dimension (at the center)
        key: jax.random.PRNGKey or None
        noise: bool, whether to add noise
        eps: float, noise amplitude (if noise=True)
    Returns:
        mps: list of jnp.ndarray, each of shape (left, 2, right)
    """
    # Compute the staircase bond dimensions
    chi = [min(2**min(i, L-i), chi_max) for i in range(L+1)]
    shapes = [(chi[i], 2, chi[i+1]) for i in range(L)]
    Bs = []
    for left, phys, right in shapes:
        tensor = jnp.zeros((left, phys, right), dtype=jnp.float64)
        tensor = tensor.at[0, 0, 0].set(1.0)
        if noise:
            import jax.random as jr
            if key is None:
                key = jr.PRNGKey(42)
            subkey, key = jr.split(key)
            tensor += eps * jr.normal(subkey, shape=(left, phys, right), dtype=jnp.float64)
        Bs.append(tensor)
    Ss = [jnp.pad(jnp.ones([1], jnp.float64), (0, chi[i]-1)) for i in range(L)]
    mps = MPS(Bs, Ss)
    mps = mps.canonicalize() # Canonicalize to ensure the noise is properly incorporated
    # mps = mps.normalize()  # Not needed, as canonicalization already normalizes the MPS
    return mps

def init_spinup_MPS_old(L):
    import numpy as np
    """Return a product state with all spins up as an MPS"""
    B = np.zeros([1, 2, 1], np.float64)
    B[0, 0, 0] = 1.
    S = np.ones([1], np.float64)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS(Bs, Ss)

@partial(jit, static_argnames=["chivC"])
def split_theta(theta, chivC):
    """Split a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Parameters
    ----------
    theta : jnp.Array[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chivC : int
        Maximum number of singular values to keep

    Returns
    -------
    A : jnp.Array[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : jnp.Array[ndim=1]
        Singular/Schmidt values.
    B : jnp.Array[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """
    chivL, dL, dR, chivR = theta.shape
    theta = jnp.reshape(theta, [chivL * dL, dR * chivR])

    X, Y, Z = jnp.linalg.svd(theta, full_matrices=False) # returns Y sorted in descending order

    # truncate
    X, Y, Z = X[:, :chivC], Y[:chivC], Z[:chivC, :]

    Y = jnp.maximum(Y, 1e-12)  # avoid division by zero

    # renormalize
    S = Y / jnp.linalg.norm(Y)  # == Y/sqrt(sum(Y**2))

    # split legs of X and Z
    A = jnp.reshape(X, [chivL, dL, chivC])
    B = jnp.reshape(Z, [chivC, dR, chivR])
    return A, S, B

def split_truncate_theta(theta, chi_max, eps):
    """Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    Parameters
    ----------
    theta : jnp.Array[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.

    Returns
    -------
    A : jnp.Array[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : jnp.Array[ndim=1]
        Singular/Schmidt values.
    B : jnp.Array[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """
    chivL, dL, dR, chivR = theta.shape
    theta = jnp.reshape(theta, [chivL * dL, dR * chivR])

    X, Y, Z = jnp.linalg.svd(theta, full_matrices=False) # returns Y sorted in descending order
    
    # truncate
    chivC = min(chi_max, jnp.sum(Y > eps))
    X, Y, Z = X[:, :chivC], Y[:chivC], Z[:chivC, :]

    # renormalize
    S = Y / jnp.linalg.norm(Y)  # == Y/sqrt(sum(Y**2))

    # split legs of X and Z
    A = jnp.reshape(X, [chivL, dL, chivC])
    B = jnp.reshape(Z, [chivC, dR, chivR])
    return A, S, B