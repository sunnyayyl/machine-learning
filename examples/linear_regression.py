from functools import partial

import jax.numpy as jnp
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from jax import random, jit
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked

from ml.activation import linear
from ml.cost import regression_mean_squared_error
from ml.definition import FloatScalar
from ml.gradient_descent import gradient_descend_training_loop
from ml.normalizer import get_z_score_normalizer
from ml.predict import predict_batch
from ml.tools import compare_predictions, generate_data


@jaxtyped(typechecker=typechecked)
def animate_gradient_decent(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    artists: list,
    ax: plt.axis,
):
    x = jnp.arange(0, 500, dtype=jnp.float32).reshape(-1, 1)
    y = predict_batch(normalizer(x), w, b, linear)
    predict_line = ax.plot(x, y, c="black")
    artists.append(predict_line)


matplotlib.use("TkAgg")
key = random.PRNGKey(12345)
key, x_raw, y_train = generate_data(key, (50,), 0, 500, jit(lambda x: 3 * x + 20))
x_raw = x_raw.reshape(-1, 1)
normalizer, invert_normalizer = get_z_score_normalizer(x_raw)
x_train = normalizer(x_raw)
w = jnp.zeros(x_train.shape[1], dtype=float)
b = 0.0

fig, ax = plt.subplots()
ax.scatter(x_raw, y_train)
artists = []
w, b, history = gradient_descend_training_loop(
    x_train,
    y_train,
    cost_function=regression_mean_squared_error,
    keep_cost_history=True,
    verbose=True,
    learning_rate=0.1,
    epoches=50,
    callback=partial(animate_gradient_decent, artists=artists, ax=ax),
)
ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=300)
ani.save("video/linear_regression.mp4")
plt.show()

compare_predictions(x_train, y_train, w, b, predict_batch_function=predict_batch)

plt.plot(
    history["cost"],
)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()
