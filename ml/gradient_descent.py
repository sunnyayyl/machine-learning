from functools import partial
from typing import Optional

import jax.numpy as jnp
from jax import jit, jacfwd, grad
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked

from ml.definition import FloatScalar, History
from ml.definition.functions import (
    PredictFunction,
    CostFunction,
    RegularizationFunction,
    CallbackFunction,
)
from ml.regularization import no_regularization


@jaxtyped(typechecker=typechecked)
def default_callback(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
): ...


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames=("cost_function", "regularization_function"))
def grad_descend(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    learning_rate: FloatScalar,
    cost_function: CostFunction,
    regularization_function: RegularizationFunction = no_regularization,
    lambda_: FloatScalar = 0.0,
) -> tuple[
    Float[Array, "feature_size"], FloatScalar, Float[Array, "feature_size"], FloatScalar
]:
    @jit
    def new_function(w, b):
        return cost_function(w, b, x_train, y_train) + regularization_function(
            w, lambda_
        )

    w_grad = jacfwd(lambda w: new_function(w, b))(w)
    b_grad = grad(new_function, argnums=1)(w, b)
    temp_w = w - learning_rate * w_grad
    temp_b = b - learning_rate * b_grad
    return temp_w, temp_b, w_grad, b_grad


def gradient_descend_training_loop(
    x_train: Array,
    y_train: Array,
    *,
    w: Optional[Array] = None,
    b: Optional[FloatScalar] = None,
    learning_rate: float,
    epoches: int,
    cost_function: CostFunction,
    predict_function: Optional[PredictFunction] = None,
    verbose: bool = False,
    keep_cost_history: bool = False,
    keep_parameter_history: bool = False,
    callback: CallbackFunction = default_callback,
    regularization_function: RegularizationFunction = no_regularization,
    lambda_: FloatScalar = 0.0,
) -> tuple[Array, FloatScalar, History]:
    if w is None:
        w = jnp.zeros(x_train.shape[1], dtype=float)
    if b is None:
        b = 0.0
    if predict_function is not None:
        cost_function = jit(partial(cost_function, predict_function=predict_function))
    cost_history = (
        [cost_function(w, b, x_train, y_train)] if keep_cost_history else None
    )
    w_history = [w] if keep_parameter_history else None
    b_history = [b] if keep_parameter_history else None
    for epoch in range(epoches):
        w, b, w_grad, b_grad = grad_descend(
            w,
            b,
            x_train,
            y_train,
            learning_rate,
            cost_function,
            regularization_function=regularization_function,
            lambda_=lambda_,
        )
        if verbose:
            print(f"Epoch {epoch} w: {w} b:{b} w_grad: {w_grad} b_grad: {b_grad}")
        if keep_cost_history:
            cost_history.append(cost_function(w, b, x_train, y_train))
        if keep_parameter_history:
            w_history.append(w)
            b_history.append(b)

        callback(w, b)
    return w, b, History(cost=cost_history, w=w_history, b=b_history)
