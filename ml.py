from typing import Callable

import jax.numpy as jnp
from jax import jit, Array, vmap, grad, jacfwd
from jax.numpy import float32


def get_normalizer(training_data: Array) -> Callable[[Array], Array]:
    x_train_mean = jnp.mean(training_data, axis=0)
    x_train_std = jnp.std(training_data, axis=0)
    return jit(lambda x: (x - x_train_mean) / x_train_std)


def get_mean_normalizer(training_data: Array) -> Callable[[Array], Array]:
    x_train_mean = jnp.mean(training_data, axis=0)
    difference = jnp.max(training_data, axis=0) - jnp.min(training_data, axis=0)
    return jit(lambda x: (x - x_train_mean) / difference)


@jit
def predict(x: Array, w: Array, b: float32) -> float32:
    return jnp.dot(w, x) + b


@jit
def predict_all(
    x: Array,
    w: Array,
    b: float32,
) -> Array:
    return vmap(lambda x: predict(x, w, b))(x)


@jit
def cost(
    w: Array,
    b: float32,
    x_train: Array,
    y_train: Array,
) -> float32:
    y_predict = predict_all(x_train, w, b)
    return jnp.mean((y_train - y_predict) ** 2)


@jit
def grad_decend(
    w: Array,
    b: float32,
    learning_rate: float32,
    x_train: Array,
    y_train: Array,
):
    w_grad = jacfwd(lambda w: cost(w, b, x_train, y_train))(w)
    b_grad = grad(cost, argnums=1)(w, b, x_train, y_train)
    temp_w = w - learning_rate * w_grad
    temp_b = b - learning_rate * b_grad
    return temp_w, temp_b, w_grad, b_grad
