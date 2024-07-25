from functools import partial
from typing import Optional, Iterable, Protocol

import jax.numpy as jnp
from jax import jit, vmap, grad, jacfwd
from jaxtyping import Array, Float, jaxtyped, ArrayLike
from typeguard import typechecked

Scalar = Float[ArrayLike, ""]


class PredictFunction(Protocol):
    def __call__(
        self,
        x: Float[Array, "data_count feature_size"],
        w: Float[Array, "feature_size"],
        b: Scalar,
    ) -> Float[Array, "data_count"]: ...


class NormalizerFunction(Protocol):
    def __call__(
        self, n: Float[ArrayLike, "..."], argnums: Optional[Iterable[int]] = None
    ) -> Float[Array, "..."]: ...


class CostFunction(Protocol):
    def __call__(
        self,
        w: Float[Array, "feature_size"],
        b: Scalar,
        x_train: Float[Array, "data_count feature_size"],
        y_train: Float[Array, "data_count"],
        predict_function: PredictFunction,
    ) -> Scalar: ...


@jaxtyped(typechecker=typechecked)
@jit
def __denormalized(
    normalized_x: Float[ArrayLike, "..."],
    argnums: Optional[Iterable[int]] = None,
    *,
    x_train_mean,
    divisor,
) -> Float[Array, "..."]:
    if argnums is not None:
        argnums = jnp.array(argnums)
        divisor = jnp.take(divisor, argnums, axis=0)
        x_train_mean = jnp.take(x_train_mean, argnums, axis=0)
    return normalized_x * divisor + x_train_mean


@jaxtyped(typechecker=typechecked)
@jit
def __normalizer(
    x: Float[ArrayLike, "..."],
    argnums: Optional[Iterable[int]] = None,
    *,
    x_train_mean,
    divisor,
) -> Float[Array, "..."]:
    if argnums is not None:
        argnums = jnp.array(argnums)
        divisor = jnp.take(divisor, argnums, axis=0)
        x_train_mean = jnp.take(x_train_mean, argnums, axis=0)
    return (x - x_train_mean) / divisor


@jaxtyped(typechecker=typechecked)
def get_z_score_normalizer(
    training_data: Float[Array, "data_count feature_size"]
) -> tuple[NormalizerFunction, NormalizerFunction]:
    x_train_mean = jnp.mean(training_data, axis=0)
    x_train_std = jnp.std(training_data, axis=0)

    return jit(
        partial(__normalizer, x_train_mean=x_train_mean, divisor=x_train_std)
    ), jit(partial(__denormalized, x_train_mean=x_train_mean, divisor=x_train_std))


@jaxtyped(typechecker=typechecked)
def get_mean_normalizer(
    training_data: Float[Array, "..."]
) -> tuple[NormalizerFunction, NormalizerFunction]:
    x_train_mean = jnp.mean(training_data, axis=0)
    difference = jnp.max(training_data, axis=0) - jnp.min(training_data, axis=0)
    return jit(
        partial(__normalizer, x_train_mean=x_train_mean, divisor=difference)
    ), jit(partial(__denormalized, x_train_mean=x_train_mean, divisor=difference))


@jaxtyped(typechecker=typechecked)
@jit
def linear_predict(
    x: Float[Array, "feature_size"], w: Float[Array, "feature_size"], b: Scalar
) -> Scalar:
    return jnp.dot(w, x) + b


@jaxtyped(typechecker=typechecked)
@jit
def linear_predict_all(
    x: Float[Array, "data_count feature_size"],
    w: Float[Array, "feature_size"],
    b: Scalar,
) -> Float[Array, "data_count"]:
    return vmap(lambda x: linear_predict(x, w, b))(x)


@jaxtyped(typechecker=typechecked)
@jit
def sigmoid(x: Scalar) -> Scalar:
    return 1 / (1 + jnp.exp(-x))


@jaxtyped(typechecker=typechecked)
@jit
def logistic_predict(
    x: Float[Array, "feature_size"], w: Float[Array, "feature_size"], b: Scalar
) -> Scalar:
    return sigmoid(linear_predict(x, w, b))


@jaxtyped(typechecker=typechecked)
@jit
def logistic_predict_all(
    x: Float[Array, "data_count feature_size"],
    w: Float[Array, "feature_size"],
    b: Scalar,
) -> Float[Array, "data_count"]:
    return vmap(lambda x: logistic_predict(x, w, b))(x)


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def logistic_cost(
    w: Float[Array, "feature_size"],
    b: Scalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    predict_function: PredictFunction = logistic_predict_all,
) -> Scalar:
    y_predict = vmap(lambda x: predict_function(x_train, w, b))(x_train)
    return jnp.mean(
        -y_train * jnp.log(y_predict) - (1 - y_train) * jnp.log(1 - y_predict)
    )


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def mean_squared_error(
    w: Float[Array, "feature_size"],
    b: Scalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    predict_function: PredictFunction = linear_predict_all,
) -> Scalar:
    y_predict = predict_function(x_train, w, b)
    return jnp.mean((y_train - y_predict) ** 2)


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="cost_function")
def grad_descend(
    w: Float[Array, "feature_size"],
    b: Scalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    learning_rate: Scalar,
    cost_function: CostFunction,
) -> tuple[Float[Array, "feature_size"], Scalar, Float[Array, "feature_size"], Scalar]:
    w_grad = jacfwd(lambda w: cost_function(w, b, x_train, y_train))(w)
    b_grad = grad(cost_function, argnums=1)(w, b, x_train, y_train)
    temp_w = w - learning_rate * w_grad
    temp_b = b - learning_rate * b_grad
    return temp_w, temp_b, w_grad, b_grad


def gradient_descend_training_loop(
    x_train: Array,
    y_train: Array,
    *,
    w: Optional[Array] = None,
    b: Optional[Float] = None,
    learning_rate: float,
    epoches: int,
    cost_function: CostFunction,
    predict_function: Optional[PredictFunction] = None,
    verbose=False,
    cost_history=False,
) -> tuple[Array, Scalar, Optional[list[Scalar]]]:
    if w is None:
        w = jnp.zeros(x_train.shape[1], dtype=float)
    if b is None:
        b = 0.0
    if predict_function is not None:
        cost_function = jit(partial(cost_function, predict_function=predict_function))
    history = []
    for epoch in range(epoches):
        w, b, w_grad, b_grad = grad_descend(
            w,
            b,
            x_train,
            y_train,
            learning_rate,
            cost_function,
        )
        if verbose:
            print(f"Epoch w: {epoch} {w} b:{b} w_grad: {w_grad} b_grad: {b_grad}")
        if cost_history:
            history.append(cost_function(w, b, x_train, y_train))
    return w, b, history if len(history) > 0 else None
