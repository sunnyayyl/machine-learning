import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked

from ml.definition import FloatScalar
from ml.regression.linear import linear_predict


@jaxtyped(typechecker=typechecked)
@jit
def sigmoid(x: FloatScalar) -> FloatScalar:
    return 1 / (1 + jnp.exp(-x))


@jaxtyped(typechecker=typechecked)
@jit
def logistic_predict(
    x: Float[Array, "feature_size"], w: Float[Array, "feature_size"], b: FloatScalar
) -> FloatScalar:
    return sigmoid(linear_predict(x, w, b))


@jaxtyped(typechecker=typechecked)
@jit
def logistic_predict_batch(
    x: Float[Array, "data_count feature_size"],
    w: Float[Array, "feature_size"],
    b: FloatScalar,
) -> Float[Array, "data_count"]:
    return vmap(lambda x: logistic_predict(x, w, b))(x)
