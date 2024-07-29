import jax.numpy as jnp
from jax import jit
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked

from ml.definition import FloatScalar


@jaxtyped(typechecker=typechecked)
@jit
def l2_regularization(
    w: Float[Array, "feature_size"],
    lambda_: FloatScalar,
) -> FloatScalar:
    return (jnp.mean(jnp.pow(w, 2))) * lambda_


@jaxtyped(typechecker=typechecked)
@jit
def no_regularization(
    w: Float[Array, "feature_size"],
    lambda_: FloatScalar,
) -> FloatScalar:
    return 0.0
