import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, value_and_grad, grad
import optax
from MPS import MPS, init_spinup_MPS, overlap
from TFI import TFIModel

L = 14
J = 1.0
g = 1.5

print(f"Transverse-field Ising model with L={L}, J={J}, g={g}")

model = TFIModel(L, J, g) # True ground energy = -23.2229594341174 

sigma_z = model.sigmaz
sigma_x = model.sigmax


MPS_all_up = init_spinup_MPS(L, 30, noise=True, eps=1e-4)

print("Chi:", MPS_all_up.get_chi())
print(f"<psi|sigma_z|psi> =  {MPS_all_up.site_expectation_value(sigma_z).sum():.8f}")
print(f"<psi|sigma_x|psi> =   {MPS_all_up.site_expectation_value(sigma_x).sum():.8f}")
print(f"<psi|psi>         =   {MPS_all_up.norm():.8f}")
print(f"<psi|H|psi>       = {model.energy(MPS_all_up):.8f}")

@jit
def loss(psi):
    """Loss function to minimize, which is the energy expectation value."""
    # psi = psi.canonicalize()
    return model.energy(psi)

psi = MPS_all_up.copy()

# optimizer = optax.scale_by_lbfgs()
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(psi)

print("Starting optimization...")
for step in range(401):
    energy, grads = value_and_grad(loss)(psi)
    updates, opt_state = optimizer.update(grads, opt_state, psi)
    psi = optax.apply_updates(psi, updates)

    psi = psi.canonicalize()
    # psi = psi.normalize()
    
    if step % 10 == 0:
        print(f"Step {step:>4}, Loss: {energy:>7.3f}, Norm: {psi.norm():>4.2f}")
        # print(f"<psi|sigma_z|psi> =  {psi.site_expectation_value(sigma_z).sum():>6.3f}")
        # print(f"<psi|sigma_x|psi> =  {psi.site_expectation_value(sigma_x).sum():>6.3f}\n")