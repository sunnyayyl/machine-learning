from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap
from jax.numpy import float32
from jaxtyping import Array, Float32

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def get_normalizer(training_data: Float32[Array, "..."]) -> Callable:
    x_train_mean = jnp.mean(training_data, axis=0)
    x_train_std = jnp.std(training_data, axis=0)
    return jit(lambda x: (x - x_train_mean) / x_train_std)


def get_mean_normalizer(training_data: Float32[Array, "..."]) -> Callable:
    x_train_mean = jnp.mean(training_data, axis=0)
    difference = jnp.max(training_data, axis=0) - jnp.min(training_data, axis=0)
    return jit(lambda x: (x - x_train_mean) / difference)


@jit
def predict(x: Float32[Array, "..."], w: Float32[Array, "..."], b: float32) -> float32:
    return jnp.dot(w, x) + b


@jit
def predict_all(
    x: Float32[Array, "..."],
    w: Float32[Array, "..."],
    b: float32,
) -> Float32[Array, "..."]:
    return vmap(lambda x: predict(x, w, b))(x)


@jit
def cost(
    w: Float32[Array, "..."],
    b: float32,
    x_train: Float32[Array, "..."],
    y_train: Float32[Array, "..."],
) -> float32:
    y_predict = predict_all(x_train, w, b)
    return jnp.mean((y_train - y_predict) ** 2)


@jit
def grad_decend(
    w: Float32[Array, "..."],
    b: float32,
    learning_rate: float32,
    x_train: Float32[Array, "..."],
    y_train: Float32[Array, "..."],
):
    w_grad = jax.jacfwd(lambda w: cost(w, b, x_train, y_train))(w)
    b_grad = jax.grad(cost, argnums=1)(w, b, x_train, y_train)
    temp_w = w - learning_rate * w_grad
    temp_b = b - learning_rate * b_grad
    return temp_w, temp_b, w_grad, b_grad


x_train_raw = jnp.array(
    [[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]], dtype=float32
)
y_train = jnp.array([460, 232, 178], dtype=float32)

get_normalizer = get_mean_normalizer
normalizer = get_normalizer(x_train_raw)
x_train = normalizer(x_train_raw)

epoches = 40
w = jnp.zeros(x_train.shape[1])
b = 0.0
lr = 0.9
history = []
for epoch in range(epoches):
    w, b, w_grad, b_grad = grad_decend(w, b, lr, x_train, y_train)
    history.append(cost(w, b, x_train, y_train))
    # print(f"Epoch w: {epoch} {w} b:{b} w_grad: {w_grad} b_grad: {b_grad}")

for i in range(x_train_raw.shape[0]):
    x = jnp.array(x_train_raw[i])
    x = normalizer(x)
    print(f"Predicted: {predict(x, w, b)}, Target: {y_train[i]}")

plt.plot(
    history,
)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()
