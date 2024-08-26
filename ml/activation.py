import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
@jit
def linear(
    z: Float[Array, "..."],
) -> Float[Array, "..."]:
    return z


@jaxtyped(typechecker=typechecked)
@jit
def sigmoid(z: Float[Array, "..."]) -> Float[Array, "..."]:
    return 1 / (1 + jnp.exp(-z))


@jaxtyped(typechecker=typechecked)
@jit
def relu(z: Float[Array, "..."]) -> Float[Array, "..."]:
    return jax.lax.max(0.0, z)
