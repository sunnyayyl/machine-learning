import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked

from ml.definition import FloatScalar


@jaxtyped(typechecker=typechecked)
@jit
def linear_predict(
    x: Float[Array, "feature_size"],
    w: Float[Array, "feature_size"],
    b: FloatScalar,
) -> FloatScalar:
    return jnp.dot(w, x) + b


@jaxtyped(typechecker=typechecked)
@jit
def linear_predict_batch(
    x: Float[Array, "data_count feature_size"],
    w: Float[Array, "feature_size"],
    b: FloatScalar,
) -> Float[Array, "data_count"]:
    return vmap(lambda x: linear_predict(x, w, b))(x)
