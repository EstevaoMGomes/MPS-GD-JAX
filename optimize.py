import os
from time import perf_counter as timer

import jax.numpy as jnp
from jax import jit, grad, config
config.update("jax_enable_x64", True)
import optax

from src.MPS import init_spinup_MPS
from src.TFI import TFIModel
from src.TEBD import TEBD_engine
from src.DMRG import DMRGEngine

import matplotlib.pyplot as plt

images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

L = 14
J = 1.0
g = 1.5

print(f"Transverse-field Ising model with L={L}, J={J}, g={g}")

model = TFIModel(L, J, g) # True ground energy = -23.2229594341174 

sigma_z = model.sigmaz
sigma_x = model.sigmax


steps = 50
psi = init_spinup_MPS(L, 30, noise=True, eps=1e-5)

print("Chi:", psi.get_chi())
print(f"<psi|sigma_z|psi> =  {psi.site_expectation_value(sigma_z).sum():.8f}")
print(f"<psi|sigma_x|psi> =   {psi.site_expectation_value(sigma_x).sum():.8f}")
print(f"<psi|psi>         =   {psi.norm_squared():.8f}")
print(f"<psi|H|psi>       = {model.energy(psi):.8f}")
print(f"<psi|H|psi>       = {model.energy_mpo(psi):.8f}")


###########################################################################
#                                DMRG                                     #
###########################################################################

print("\nRunning DMRG...")

energy_DMRG = [model.energy(psi)]
dmrg = DMRGEngine(psi.copy(), model, chi_max=30, eps=1e-12)
for sweep in range(4):
    dmrg.sweep()
    energy = model.energy(dmrg.psi)
    energy_DMRG.append(energy)

print(f"DMRG ground state energy: {energy_DMRG[-1]:.5f}")

energy_DMRG = jnp.array(energy_DMRG)

###########################################################################
#                                TEBD                                     #
###########################################################################

print("\nRunning TEBD...")
psi_TEBD = init_spinup_MPS(L, 2, noise=False)
energy_TEBD = [model.energy(psi_TEBD)]
scheduler = optax.schedules.exponential_decay(1.e-1, 20, 0.1, staircase=False)
for step in range(steps):
    tebd = TEBD_engine(psi_TEBD, model, chi_max=30, eps=1e-10, dt=scheduler(step))
    energy = tebd.run(1, order=2)
    energy_TEBD += energy
    print(f"TEBD ground state energy: {energy[-1]:.5f}")

energy_TEBD = jnp.array(energy_TEBD)

###########################################################################
#                                 GD                                      #
###########################################################################

@jit
def loss(psi):
    """Loss function to minimize, which is the energy expectation value."""
    return model.energy_mpo(psi) / psi.norm_squared()

# optimizer = optax.scale_by_lbfgs()
scheduler = optax.schedules.exponential_decay(2.e-2, 20, 0.5, staircase=False)
optimizer = optax.adam(learning_rate=scheduler)
opt_state = optimizer.init(psi)

energy_GD = [model.energy(psi)]

print("\nRunning Gradient Descent...")

start = timer()
for step in range(steps):
    # Compute the gradient of the energy expectation value
    grads = grad(loss)(psi)
    # Apply the optimizer to update the MPS
    updates, opt_state = optimizer.update(grads, opt_state, psi)
    psi = optax.apply_updates(psi, updates)
    # Normalize the MPS (does not change the energy)
    psi = psi.normalize()

    energy = model.energy_mpo(psi)
    energy_GD.append(energy)

    if step % 1 == 0:
        print(f"Step {step:>4}, Loss: {energy:>9.5f}, Learning rate: {scheduler(step):>4.2e}, norm: {jnp.sqrt(psi.norm_squared()):.8f}")

energy_GD = jnp.array(energy_GD)
print(f"Optimization completed in {timer() - start:.3f} seconds")

###########################################################################
#                               Plotting                                  #
###########################################################################

plt.figure(figsize=(10, 6))
plt.plot(jnp.arange(steps + 1), energy_GD, label='Gradient Descent', linewidth=2)
plt.plot(jnp.arange(steps + 1), energy_TEBD, label='TEBD', linewidth=2)
plt.plot(jnp.arange(5), energy_DMRG, label='DMRG', linewidth=2)
theoretical_energy = -23.22295943411735664
plt.axhline(y=theoretical_energy, color='r', linestyle='--', label=fr'Theoretical $E_g={theoretical_energy:.5f}$', linewidth=2)
plt.xlim(0, steps)
plt.xlabel("Step")
plt.ylabel("Ground Energy")
plt.title(f"Ground State Optimization for the TFI hamiltonian with L={L}, J={J}, g={g}")
plt.grid()
plt.legend()
plt.savefig(os.path.join(images_dir, "ground_state_optimization.png"), dpi=300)

relative_error_GD = jnp.abs((theoretical_energy - energy_GD) / theoretical_energy)
relative_error_TEBD = jnp.abs((theoretical_energy - energy_TEBD) / theoretical_energy)
relative_error_DMRG = jnp.abs((theoretical_energy - energy_DMRG) / theoretical_energy)

plt.figure(figsize=(10, 6))
plt.plot(jnp.arange(steps + 1), relative_error_GD, label='Gradient Descent', linewidth=2)
plt.plot(jnp.arange(steps + 1), relative_error_TEBD, label='TEBD', linewidth=2)
plt.plot(jnp.arange(5), relative_error_DMRG, label='DMRG', linewidth=2)
plt.xlim(0, steps)
plt.ylim(1.e-15, 1.e-1)
plt.xlabel("Step")
plt.ylabel("Relative Error")
plt.title(f"Relative Error in Ground State Energy for TFI Model with L={L}, J={J}, g={g}")
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig(os.path.join(images_dir, "relative_error_ground_state.png"), dpi=300)
plt.show()