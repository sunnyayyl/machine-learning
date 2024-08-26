from functools import partial

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from jaxtyping import Float, Array
from jaxtyping import jaxtyped
from typeguard import typechecked

import ml
from ml.activation import sigmoid
from ml.cost import regression_logistic_cost
from ml.gradient_descent import gradient_descend_training_loop
from ml.normalizer import get_mean_normalizer
from ml.predict import predict, predict_batch
from ml.tools import compare_predictions


@jaxtyped(typechecker=typechecked)
def animate_gradient_decent(
    w: Float[Array, "feature_size"],
    b: ml.FloatScalar,
    artists=list,
    ax=plt.axis,
):
    x1 = jnp.arange(-2.0, 4.0, dtype=jnp.float32)
    x2 = ((-b) - x1 * w[0]) / w[1]
    predict_line = ax.plot(
        denormalize(x1, argnums=(0,)), denormalize(x2, argnums=(1,)), c="black"
    )
    artists.append(predict_line)


x_raw = jnp.array(
    [[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]],
)
y_train = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.float32)  # .reshape(-1, 1)

# x_train = x_raw
normalizer, denormalize = get_mean_normalizer(x_raw)
x_train = normalizer(x_raw)
fig, ax = plt.subplots()
plt.xlim([-4, 8])
plt.ylim([8, -8])
ax.scatter(x_raw[:, 0], x_raw[:, 1], c=y_train)
artists = []
w, b, history = gradient_descend_training_loop(
    x_train,
    y_train,
    epoches=800,
    learning_rate=0.1,
    verbose=True,
    cost_function=regression_logistic_cost,
    cost_history=True,
    predict_function=predict,
    callback=partial(animate_gradient_decent, artists=artists, ax=ax),
    w=jnp.array([-3.0, 3.0], dtype=jnp.float32),
    b=3.0,
)
ani = animation.ArtistAnimation(
    fig=fig, artists=artists, interval=3000 / len(artists), blit=True
)
ani.save("logistic_regression.mp4")
plt.show()
print(f"w: {w} b: {b}")
plt.plot(
    history,
)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()
prediction = predict_batch(x_train, w, b, sigmoid)
plt.scatter(x_raw[:, 0], x_raw[:, 1], c=y_train)
x1 = jnp.arange(-2.0, 4.0, dtype=jnp.float32)
x2 = ((-b) - x1 * w[0]) / w[1]
plt.plot(denormalize(x1, argnums=(0,)), denormalize(x2, argnums=(1,)))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
compare_predictions(x_train, y_train, w, b, predict_batch_function=predict)
