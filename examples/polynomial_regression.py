from functools import partial
from typing import Callable

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, random, Array, vmap
from matplotlib.animation import ArtistAnimation

from ml.cost import mean_squared_error
from ml.gradient_descent import gradient_descend_training_loop
from ml.normalizer import get_z_score_normalizer
from ml.regression.linear import linear_predict_batch
from ml.tools import generate_data

matplotlib.use("TkAgg")


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
x_raw, y_train = generate_data(
    key, (50,), 0, 500, jit(lambda x: 5 * x**3 - 2 * x**2 + 3 * x + 6)
)
x_train = mod_feature(x_raw)
normalizer, invert_normalizer = get_z_score_normalizer(x_train)
x_train = normalizer(x_train)

w, b, history = gradient_descend_training_loop(
    x_train,
    y_train,
    learning_rate=0.1,
    epoches=200,
    cost_function=mean_squared_error,
    verbose=True,
    keep_cost_history=True,
    keep_parameter_history=True,
    w=random.uniform(random.split(key)[1], (3,), float, -50000, 0),
    b=random.uniform(random.split(key)[1], (1,), float, -50000, 0)[0],
)

fig, ax = plt.subplots()
to_predict = jnp.arange(0, 500, 5, dtype=float).reshape(-1, 1)
to_predict_mapped = mapped_modded_normalize(to_predict, normalizer)
ax.scatter(x_raw, y_train)
artists = []
for i in range(len(history["w"]) - 1):
    text = ax.text(0.5, 0.85, f"Epoch: {i}", transform=ax.transAxes)
    prediction = linear_predict_batch(
        to_predict_mapped, history["w"][i], history["b"][i]
    )
    artists.append(ax.plot(to_predict, prediction, "black") + [text])
ani = ArtistAnimation(
    fig=fig, artists=artists, interval=10000 / len(artists), blit=False
)
ani.save("video/polynomial_regression.mp4")
plt.show()

print(f"w: {w} b: {b}")
plt.plot(
    history["cost"],
)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()
