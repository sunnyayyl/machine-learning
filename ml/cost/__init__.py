from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked

from ml.definition import FloatScalar
from ml.definition.functions import PredictFunction
from ml.regression.linear import linear_predict_batch
from ml.regression.logistic import logistic_predict_batch


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def logistic_cost(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    predict_function: PredictFunction = logistic_predict_batch,
) -> FloatScalar:
    y_predict = vmap(lambda x: predict_function(x_train, w, b))(x_train)
    return jnp.mean(
        -y_train * jnp.log(y_predict) - (1 - y_train) * jnp.log(1 - y_predict)
    )


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def mean_squared_error(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    predict_function: PredictFunction = linear_predict_batch,
) -> FloatScalar:
    y_predict = predict_function(x_train, w, b)
    return jnp.mean((y_train - y_predict) ** 2)
