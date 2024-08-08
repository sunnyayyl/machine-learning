from functools import partial

import jax.numpy as jnp
from jax import vmap, jit
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked

from ml.activation import linear
from ml.definition import FloatScalar
from ml.definition.functions import ActivationFunction


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames=("activation_function",))
def predict(
    x: Float[Array, "feature_size"],
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    activation_function: ActivationFunction = linear,
) -> FloatScalar:
    return activation_function(jnp.dot(w, x) + b)


def predict_batch(
    x: Float[Array, "data_count feature_size"],
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    activation_function: ActivationFunction = linear,
) -> Float[Array, "data_count"]:
    return vmap(partial(predict, w=w, b=b, activation_function=activation_function))(x)
