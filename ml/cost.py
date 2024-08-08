from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import jaxtyped, Array, Float
from typeguard import typechecked

from ml.activation import sigmoid
from ml.definition import FloatScalar
from ml.definition.functions import PredictFunction
from ml.predict import predict


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def logistic_cost(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    predict_function: PredictFunction = partial(predict, activation_function=sigmoid),
) -> FloatScalar:
    y_predict = vmap(partial(predict_function, w=w, b=b))(x_train)
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
    predict_function: PredictFunction = predict,
) -> FloatScalar:
    y_predict = vmap(partial(predict_function, w=w, b=b))(x_train)
    return jnp.mean((y_train - y_predict) ** 2)
