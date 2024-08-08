from functools import partial
from typing import Callable, Iterable

import jax.numpy as jnp
from jax import jit, random
from jaxtyping import Array, Float, jaxtyped, ArrayLike, PRNGKeyArray
from typeguard import typechecked

from ml.definition import FloatScalar
from ml.definition.functions import PredictBatchFunction


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames=("shape", "y_function"))
def generate_data(
    key: ArrayLike,
    shape: Iterable[int],
    minval: ArrayLike,
    maxval: ArrayLike,
    y_function: Callable[[ArrayLike], ArrayLike],
    min_noise: float = 0.0,
    max_noise: float = 0.0,
) -> tuple[PRNGKeyArray, Float[Array, "input_size"], Float[Array, "input_size"]]:
    key, subkey, subkey1, subkey2, subkey3, subkey4 = random.split(key, 6)
    x = jnp.sort(random.uniform(key=subkey, shape=shape, minval=minval, maxval=maxval))

    y = y_function(x) + random.uniform(
        subkey4, shape, minval=min_noise, maxval=max_noise
    )
    x += +random.uniform(subkey1, shape, minval=min_noise, maxval=max_noise)

    return key, x, y


def compare_predictions(
    x_train: Array,
    y_train: Array,
    w: Array,
    b: FloatScalar,
    *,
    predict_batch_function: PredictBatchFunction,
):
    for i in range(x_train.shape[0]):
        x = jnp.array(x_train[i])
        print(f"Predicted: {predict_batch_function(x, w, b)}, Target: {y_train[i]}")
