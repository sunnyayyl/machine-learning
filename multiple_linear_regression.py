import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jaxtyping import Array, Float64
from jax.numpy import float64
from typing import Callable

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def get_normalizer(training_data: Float64[Array, "..."]) -> Callable:
    x_train_mean = jnp.mean(training_data, axis=0)
    x_train_std = jnp.std(training_data, axis=0)
    return jit(lambda x: (x - x_train_mean) / x_train_std)


@jit
def predict(x: Float64[Array, "..."], w: Float64[Array, "..."], b: float64) -> float64:
    return jnp.dot(w, x) + b


@jit
def predict_all(
        x: Float64[Array, "..."],
        w: Float64[Array, "..."],
        b: float64,
) -> Float64[Array, "..."]:
    return vmap(lambda x: predict(x, w, b))(x)


@jit
def cost(
        w: Float64[Array, "..."],
        b: float64,
        x_train: Float64[Array, "..."],
        y_train: Float64[Array, "..."],
) -> float64:
    y_predict = predict_all(x_train, w, b)
    return jnp.mean((y_train - y_predict) ** 2)


@jit
def grad_decend(
        w: Float64[Array, "..."],
        b: float64,
        learning_rate: float64,
        x_train: Float64[Array, "..."],
        y_train: Float64[Array, "..."],
):
    w_grad = jax.jacfwd(lambda w: cost(w, b, x_train, y_train))(w)
    b_grad = jax.grad(cost, argnums=1)(w, b, x_train, y_train)
    temp_w = w - learning_rate * w_grad
    temp_b = b - learning_rate * b_grad
    return temp_w, temp_b, w_grad, b_grad


x_train_raw = jnp.array(
    [[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]], dtype=float64
)
y_train = jnp.array([460, 232, 178], dtype=float64)

normalizer = get_normalizer(x_train_raw)
x_train = normalizer(x_train_raw)
w = jnp.zeros(x_train.shape[1])
b = 0.0

for epoch in range(2500):
    w, b, w_grad, b_grad = grad_decend(w, b, 0.01, x_train, y_train)
    print(f"Epoch w: {epoch} {w} b:{b} w_grad: {w_grad} b_grad: {b_grad}")

for i in range(x_train_raw.shape[0]):
    x = jnp.array(x_train_raw[i])
    x = normalizer(x)
    print(f"Predicted: {predict(x, w, b)}, Target: {y_train[i]}")
