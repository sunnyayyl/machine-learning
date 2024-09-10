from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import jaxtyped, Array, Float
from typeguard import typechecked

from ml.activation import sigmoid, softmax
from ml.definition import FloatScalar
from ml.definition.functions import PredictFunction
from ml.predict import predict


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def regression_logistic_cost(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    predict_function: PredictFunction = partial(predict, activation_function=sigmoid),
) -> FloatScalar:
    y_predict = vmap(partial(predict_function, w=w, b=b))(x_train)
    return logistic_cost(y_train, y_predict)


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def regression_mean_squared_error(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    predict_function: PredictFunction = predict,
) -> FloatScalar:
    y_predict = vmap(partial(predict_function, w=w, b=b))(x_train)
    return mean_squared_error(y_train, y_predict)


@jit
def logistic_cost(y_train: Array, y_predict: Array) -> Array:
    return jnp.mean(
        -y_train * jnp.log(y_predict) - (1 - y_train) * jnp.log(1 - y_predict)
    )


@jit
def mean_squared_error(y_train: Array, y_predict: Array) -> Array:
    return jnp.mean(jnp.pow(y_train - y_predict, 2))


@jit
def softmax_cross_entropy(y_train: Array, y_predict: Array) -> Array:
    return -jnp.log(jnp.take_along_axis(softmax(y_train), y_predict, -1))
