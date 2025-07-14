"""Toy code implementing the time evolving block decimation (TEBD)."""

import numpy as np
from scipy.linalg import expm
from src.MPS import split_truncate_theta

class TEBD_engine:
    def __init__(self, psi, model, chi_max, eps, dt):
        self.psi = psi
        self.model = model
        self.chi_max = chi_max
        self.eps = eps
        self.dt = dt
        self._U_bonds_dt = None
        self._U_bonds_half_dt = None
    
    def _init_U_bonds(self, dt):
        """Given a model, calculate ``U_bonds[i] = expm(-dt*model.H_bonds[i])``.

        Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
        Note that no imaginary 'i' is included, thus real `dt` means imaginary time evolution!
        """
        H_bonds = self.model.H_bonds
        d = H_bonds[0].shape[0]
        U_bonds = []
        for H in H_bonds:
            H = np.reshape(H, [d * d, d * d])
            U = expm(-dt * H)
            U_bonds.append(np.reshape(U, [d, d, d, d]))
        return U_bonds
    
    @property
    def U_bonds_dt(self):
        """Return the U_bonds for the full time step."""
        if self._U_bonds_dt is None:
            self._U_bonds_dt = self._init_U_bonds(self.dt)
        return self._U_bonds_dt

    @property
    def U_bonds_half_dt(self):
        """Return the U_bonds for the half time step."""
        if self._U_bonds_half_dt is None:
            self._U_bonds_half_dt = self._init_U_bonds(self.dt / 2)
        return self._U_bonds_half_dt

    def update_bond(self, i, bond="dt"):
        """Apply `U_bond` acting on i,j=(i+1) to `psi`."""
        if bond == "dt":
            U_bond = self.U_bonds_dt[i]
        elif bond == "half_dt" or bond == "dt/2":
            U_bond = self.U_bonds_half_dt[i]
        else:
            raise ValueError("bond must be 'dt', 'half_dt' or 'dt/2'")
        j = i + 1
        # construct theta matrix
        theta = self.psi.get_theta2(i)  # vL i j vR
        # apply U
        Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i j [i*] [j*], vL [i] [j] vR
        Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
        # split and truncate
        Ai, Sj, Bj = split_truncate_theta(Utheta, self.chi_max, self.eps)
        # put back into MPS
        Gi = np.tensordot(np.diag(self.psi.Ss[i]**(-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
        self.psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])  # vL i [vC], [vC] vC
        self.psi.Ss[j] = Sj  # vC
        self.psi.Bs[j] = Bj  # vC j vR

    def run(self, steps, order=2):
        """Evolve the state `psi` for `N_steps` time steps with TEBD.
        The state psi is modified in place."""
        energy_TEBD = []
        Nbonds = self.psi.L - 1

        if order == 1:
            for n in range(steps):
                for k in [0, 1]:  # even, odd
                    for i_bond in range(k, Nbonds, 2):
                        self.update_bond(i_bond, "dt")
                energy_TEBD.append(self.model.energy(self.psi))
            return energy_TEBD

        elif order == 2:
            for i_bond in range(0, Nbonds, 2): # even bonds
                self.update_bond(i_bond, "half_dt")
            for i_bond in range(1, Nbonds, 2): # odd bonds
                self.update_bond(i_bond, "dt")
            energy_TEBD.append(self.model.energy(self.psi))
            for n in range(steps - 1):
                for k in [0, 1]:  # even and odd
                    for i_bond in range(k, Nbonds, 2):
                        self.update_bond(i_bond, "dt")
                energy_TEBD.append(self.model.energy(self.psi))
            for i_bond in range(0, Nbonds, 2): # even bonds
                self.update_bond(i_bond, "half_dt")
            energy_TEBD.pop()
            energy_TEBD.append(self.model.energy(self.psi))
            return energy_TEBD
        
        else:
            raise ValueError("order must be 1 or 2")
    