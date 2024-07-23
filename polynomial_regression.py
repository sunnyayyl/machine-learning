from functools import partial
from typing import Callable

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, random, Array, vmap
from jax.numpy import float32

import ml


@partial(jit, static_argnames="data_size")
def generate_data(key: Array, data_size: int) -> tuple[float32, float32]:
    _, subkey = random.split(key)
    x = jnp.sort(random.uniform(key=subkey, shape=(data_size,), minval=0, maxval=500))
    # x = jnp.arange(-200.0, 200.0, step=30.0)
    y = 2 * jnp.pow(x, 2) + 6 * jnp.pow(x, 3) + 1
    return x, y


@jit
def mod_feature(x: Array) -> Array:
    return jnp.c_[x, x**2, x**3]


@partial(jit, static_argnames="normalizer")
def moded_normalize(x: Array, normalizer: Callable[[Array], Array]) -> Array:
    x = mod_feature(x)[0]
    return normalizer(x)


@partial(jit, static_argnames="normalizer")
def mapped_modded_normalize(x: Array, normalizer: Callable[[Array], Array]) -> Array:
    return vmap(lambda x: moded_normalize(x, normalizer))(x)


key = random.PRNGKey(12345)
x_raw, y_train = generate_data(key, 50)
x_train = mod_feature(x_raw)
normalizer = ml.get_z_score_normalizer(x_train)
x_train = normalizer(x_train)

w, b, history = ml.gradient_descend_training_loop(
    x_train,
    y_train,
    learning_rate=0.1,
    epoches=800,
    cost_function=ml.mean_squared_error,
    verbose=True,
    cost_history=True,
)

print(f"w: {w} b: {b}")
plt.plot(
    history,
)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()

mapped_x = mapped_modded_normalize(x_raw, normalizer)
assert (mapped_x == x_train).all()
prediction = ml.linear_predict_all(mapped_x, w, b)
plt.scatter(x_raw, y_train)
plt.plot(x_raw, prediction)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
