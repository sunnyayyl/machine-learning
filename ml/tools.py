from functools import partial
from typing import Callable, Iterable

import jax.numpy as jnp
from jax import jit, random
from jaxtyping import Array, Float, jaxtyped, ArrayLike
from typeguard import typechecked

from ml.definition import FloatScalar
from ml.definition.functions import PredictFunction


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames=("shape", "y_function"))
def generate_data(
    key: ArrayLike,
    shape: Iterable[int],
    minval: ArrayLike,
    maxval: ArrayLike,
    y_function: Callable[[ArrayLike], ArrayLike],
) -> tuple[Float[Array, "input_size"], Float[Array, "input_size"]]:
    _, subkey = random.split(key)
    x = jnp.sort(random.uniform(key=subkey, shape=shape, minval=minval, maxval=maxval))
    y = y_function(x)
    return x, y


def compare_predictions(
    x_train: Array,
    y_train: Array,
    w: Array,
    b: FloatScalar,
    *,
    predict_function: PredictFunction,
):
    for i in range(x_train.shape[0]):
        x = jnp.array(x_train[i])
        print(f"Predicted: {predict_function(x, w, b)}, Target: {y_train[i]}")
