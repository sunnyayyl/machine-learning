import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import jaxtyped
from typeguard import typechecked

from ml.definition import FloatScalar


@jaxtyped(typechecker=typechecked)
@jit
def linear(
    z: FloatScalar,
) -> FloatScalar:
    return z


@jaxtyped(typechecker=typechecked)
@jit
def sigmoid(z: FloatScalar) -> FloatScalar:
    return 1 / (1 + jnp.exp(-z))


@jaxtyped(typechecker=typechecked)
@jit
def relu(z: FloatScalar) -> FloatScalar:
    return jax.lax.max(0.0, z)
