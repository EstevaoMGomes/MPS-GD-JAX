import os
from time import perf_counter as timer

import jax.numpy as jnp
from jax import jit, grad, config, block_until_ready
config.update("jax_enable_x64", True)
import optax

from src.MPS import MPS, init_spinup_MPS, init_spinright_MPS
from src.TFI import TFIModel
from src.TEBD import TEBD_engine
from src.DMRG import DMRGEngine
from src.ExactDiag import finite_gs_energy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 16})

images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

L = 14
J = 1.0
g = 1.5

theoretical_energies = {
    0.5: -14.01899646121673726,  # ground state energy for g=0.5    
    1.0: -17.47100405473177176,  # ground state energy for g=1.0
    1.5: -23.22295943411735664   # ground state energy for g=1.5
}

energies_DMRG = []
energies_TEBD = []
energies_GD = []

theoretical_energies = {}

g_list = [0.5, 1.0, 1.5]
for g in g_list:
    theoretical_energies[g] = finite_gs_energy(L, J, g)

    if g <= J:
        init_MPS = init_spinright_MPS
    else:
        init_MPS = init_spinup_MPS

    print(f"Transverse-field Ising model with L={L}, J={J}, g={g}")

    model = TFIModel(L, J, g)

    steps = 100
    psi = init_MPS(L, 30, noise=True, eps=1e-5)

    print("Chi:", psi.get_chi())
    print(f"<psi|sigma_z|psi> =  {psi.site_expectation_value(model.sigmaz).sum():.8f}")
    print(f"<psi|sigma_x|psi> =   {psi.site_expectation_value(model.sigmax).sum():.8f}")
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
    energies_DMRG.append(energy_DMRG)

    ###########################################################################
    #                                TEBD                                     #
    ###########################################################################

    print("\nRunning TEBD...")

    psi_TEBD = init_MPS(L, 2, noise=False)
    energy_TEBD = [model.energy(psi_TEBD)]
    scheduler = optax.schedules.exponential_decay(1.e-1, 30, 0.1, staircase=False)
    
    for step in range(steps):
        tebd = TEBD_engine(psi_TEBD, model, chi_max=30, eps=1e-10, dt=scheduler(step))
        energy = tebd.run(1, order=2)
        energy_TEBD += energy

        if step % 5 == 0:
            print(f"TEBD ground state energy: {energy[-1]:.5f}")

    energy_TEBD = jnp.array(energy_TEBD)
    energies_TEBD.append(energy_TEBD)

    ###########################################################################
    #                                 GD                                      #
    ###########################################################################

    @jit
    def loss(psi):
        """Loss function to minimize, which is the energy expectation value."""
        return model.energy_mpo(psi) / psi.norm_squared()

    @jit
    def loss_grad(psi):
        """Gradient of the loss function."""
        energy = model.energy_mpo(psi)
        norm_squared = psi.norm_squared()

        energy_grad = model.energy_mpo_grad(psi)
        norm_grad = psi.norm_squared_grad()

        Bs_grad = [energy_grad[i]/ norm_squared - energy * norm_grad[i] / norm_squared**2  for i in range(psi.L)]

        return MPS(Bs_grad, psi.Ss)

    scheduler = optax.schedules.exponential_decay(2.e-2, 30, 0.5, staircase=False)
    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(psi)

    energy_GD = [model.energy(psi)]

    print("\nRunning Gradient Descent...")

    t_grad_analytical = 0

    if g == 1.5:
        grad_error_avg = 0
        grad_error_std = 0
        grad_error_max = 0

        t_grad_jax = 0

        # First run for compilation
        grad_analytical = block_until_ready(loss_grad(psi))
        grads = block_until_ready(grad(loss)(psi))

    start = timer()
    for step in range(steps):
        t_start_analytical = timer()
        grad_analytical = block_until_ready(loss_grad(psi))
        t_grad_analytical += timer() - t_start_analytical

        if g == 1.5:
            t_start_jax = timer()
            grads = block_until_ready(grad(loss)(psi))
            t_grad_jax += timer() - t_start_jax

            for i in range(psi.L):
                errors = grad_analytical.Bs[i] - grads.Bs[i]
                grad_error_avg += jnp.mean(jnp.abs(errors))
                grad_error_std += jnp.std(errors)
                grad_error_max = max(grad_error_max, jnp.max(jnp.abs(errors)))

        # Apply the optimizer to update the MPS
        updates, opt_state = optimizer.update(grad_analytical, opt_state, psi)
        psi = optax.apply_updates(psi, updates)

        # Normalize the MPS (does not change the energy)
        psi = psi.normalize()

        energy = model.energy_mpo(psi)
        energy_GD.append(energy)

        if step % 5 == 0:
            print(f"Step {step:>4}, Loss: {energy:>9.5f}, Learning rate: {scheduler(step):>4.2e}")

    if g == 1.5:
        grad_error_avg /= steps
        grad_error_std /= steps

        t_grad_analytical /= steps
        t_grad_jax /= steps

    energy_GD = jnp.array(energy_GD)
    energies_GD.append(energy_GD)

    print(f"Optimization completed in {timer() - start:.3f} seconds\n")

###########################################################################
#                               Plotting                                  #
###########################################################################

# Plotting energy decay

fig, ax1 = plt.subplots(figsize=(10, 6))

for i, energy_DMRG in enumerate(energies_DMRG):
    ax1.plot(jnp.arange(5), energy_DMRG, label='DMRG' if i == 0 else "", linewidth=2, color="limegreen")
for i, energy_TEBD in enumerate(energies_TEBD):
    ax1.plot(jnp.arange(steps + 1), energy_TEBD, label='TEBD' if i == 0 else "", linewidth=2, color="blue")
for i, energy_GD in enumerate(energies_GD):
    ax1.plot(jnp.arange(steps + 1), energy_GD, label='Gradient Descent' if i == 0 else "", linewidth=2, color="red")

ax1.set_xlabel("Steps", fontsize=17)
ax1.set_ylabel("Ground Energy", fontsize=17)
ax1.set_xlim(0, steps)
ax1.set_ylim(-24, -12)

ax2 = ax1.twinx()
ax2.set_ylim(-24, -12)

# Store ticks and labels
yticks = []
yticklabels = []

for i, (g, theoretical_energy) in enumerate(theoretical_energies.items()):
    ax1.axhline(y=theoretical_energy, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label = "Exact Diagonalization" if i == 0 else "")
    yticks.append(theoretical_energy)
    yticklabels.append(fr'$g={g}$')

# Set the custom ticks and labels
ax2.set_yticks(yticks)
ax2.set_yticklabels(yticklabels, fontsize=17)

ax1.minorticks_on()
ax1.grid(which='major', axis='y', alpha=0.3, linestyle='--')
ax1.grid(which='both', axis='x', alpha=0.3, linestyle='--')
ax1.legend(loc='center right', bbox_to_anchor=(0.98, 0.30), fontsize=15)

# Remove right spine for cleaner look
ax1.spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save with high quality
plt.savefig(os.path.join(images_dir, "ground_state_optimization.png"), dpi=300, bbox_inches='tight', facecolor='white')


# Plotting relative error
plt.figure(figsize=(10, 6))

markers = ["s", "D", "o"]
for i, (g, theoretical_energy) in enumerate(theoretical_energies.items()):
    relative_error_DMRG = jnp.abs((theoretical_energy - energies_DMRG[i]) / theoretical_energy)
    relative_error_TEBD = jnp.abs((theoretical_energy - energies_TEBD[i]) / theoretical_energy)
    relative_error_GD = jnp.abs((theoretical_energy - energies_GD[i]) / theoretical_energy)

    plt.plot(jnp.arange(5), relative_error_DMRG, marker=markers[i], markevery=10, color="limegreen", linewidth=2)
    plt.plot(jnp.arange(steps + 1), relative_error_TEBD, marker=markers[i], markevery=10, color="blue", linewidth=2)
    plt.plot(jnp.arange(steps + 1), relative_error_GD, marker=markers[i], markevery=10, color="red", linewidth=2)

plt.xlim(0, steps)
plt.ylim(1.e-6, 1.e-0)
plt.xlabel("Steps", fontsize=17)
plt.ylabel("Relative Error", fontsize=17)
plt.yscale('log')

# Custom legend (just measurements)
custom_lines = [
    Line2D([0], [0], color='limegreen'),
    Line2D([0], [0], color='blue'),
    Line2D([0], [0], color='red')
]
legend1 = plt.legend(custom_lines, ['DMRG', 'TEBD', 'Gradient Descent'], loc='upper right', fontsize=15)
plt.gca().add_artist(legend1)  # Add first legend manually

# Second legend
custom_markers = [
    Line2D([0], [0], color='black', marker='s', linestyle=''),
    Line2D([0], [0], color='black', marker='D', linestyle=''),
    Line2D([0], [0], color='black', marker='o', linestyle='')
]
legend2 = plt.legend(custom_markers, [f'g={g_list[0]}', f'g={g_list[1]}', f'g={g_list[2]}'], loc='upper right', fontsize=15, bbox_to_anchor=(0.65, 1.0))
plt.gca().add_artist(legend2)  # Add second legend manually

plt.grid()
plt.minorticks_on()
plt.grid(which='both', alpha=0.3, linestyle='--')
plt.savefig(os.path.join(images_dir, "relative_error_ground_state.png"), dpi=300)

# Plotting gradient comparison
fig, ax1 = plt.subplots(figsize=(10, 6))

# Define colors
colors = {
    'errors': "#FF7F0E",      # Modern orange
    'analytical': "#0000FF",  # Deep blue
    'jax': '#FF0000',         # Deep red
    'edge': "#000000"         # Black for edges
}

# Data preparation
error_data = [
    ("Average", jnp.abs(grad_error_avg)),
    ("Std Dev", jnp.abs(grad_error_std)),
    ("Maximum", jnp.abs(grad_error_max)),
]

error_labels = [item[0] for item in error_data]
error_vals = [item[1] for item in error_data]

# Create main error bars
X_axis = jnp.arange(len(error_labels))
bar_width = 0.55 

bars1 = ax1.bar(X_axis, error_vals, bar_width, 
                color=colors['errors'], edgecolor=colors['edge'], 
                linewidth=1.5, alpha=0.8, label="Gradient Errors")

# Formatting for left y-axis 
ax1.set_xticks(X_axis)
ax1.set_xticklabels(error_labels, fontsize=15)
ax1.set_ylabel("Absolute Error", fontsize=17)
ax1.set_yscale("log")
ax1.set_ylim(1e-14, 1e-12)
ax1.grid(True, which='both', alpha=0.3, linestyle='--')
ax1.tick_params(axis='y', labelsize=15)

# Create second y-axis for runtime
ax2 = ax1.twinx()
ax2.set_ylabel('Computation Time (s)', fontsize=17)
ax2.set_yscale('log')
ax2.set_ylim(1e-4, 1e-2)
ax2.tick_params(axis='y', labelsize=15)

# Add runtime bars
runtime_x = len(error_labels) + 0.5
bar_width_runtime = 0.45

# Add space for runtime bars
ax1.set_xlim(-0.5, runtime_x + 0.8)
ax2.set_xlim(-0.5, runtime_x + 0.8)

bars2 = ax2.bar(runtime_x - bar_width_runtime/2, t_grad_analytical, bar_width_runtime, 
                label="Analytical Gradient", color=colors['analytical'], 
                edgecolor=colors['edge'], linewidth=1.2, alpha=0.8)

bars3 = ax2.bar(runtime_x + bar_width_runtime/2, t_grad_jax, bar_width_runtime, 
                label="JAX Gradient", color=colors['jax'], 
                edgecolor=colors['edge'], linewidth=1.2, alpha=0.8)

# Update x-axis labels to include runtime
all_labels = error_labels + ["Runtime"]
ax1.set_xticks(list(X_axis) + [runtime_x])
ax1.set_xticklabels(all_labels, fontsize=15)

# Create a unified legend in the original position
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=15)

# Remove top and right spines for cleaner look
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save with high quality
plt.savefig(os.path.join(images_dir, "gradient_comparison.png"), dpi=300, bbox_inches='tight', facecolor='white')

plt.show()