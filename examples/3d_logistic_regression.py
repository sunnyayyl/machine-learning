import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ml.cost import regression_logistic_cost
from ml.gradient_descent import grad_descend
from ml.normalizer import get_mean_normalizer

x_raw = jnp.array(
    [
        [0.5, 1.5, -5],
        [1.0, 1.0, -3],
        [1.5, 0.5, 0],
        [3.0, 0.5, 3],
        [2.0, 2.0, 4],
        [1.0, 2.5, 3],
    ],
)
y_train = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.float32)  # .reshape(-1, 1)

key = random.PRNGKey(54321)
normalizer, inverse_normalizer = get_mean_normalizer(x_raw)
x_train = normalizer(x_raw)
w = random.uniform(key, (3,), float, -10.0, 10.0)
b = -5.0
epoches = 3000

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlim3d(-10, 10)
ax.set_ylim3d(-10, 10)
ax.set_zlim3d(-10, 10)
ax.scatter(x_raw[:, 0], x_raw[:, 1], x_raw[:, 2], c=y_train)
x1 = jnp.arange(-10, 10, dtype=jnp.float32).reshape(-1, 1)
x2 = jnp.arange(-10, 10, dtype=jnp.float32).reshape(-1, 1)
x3 = ((-b) - w[0] * x1 - w[1] * x2) / w[2]
line = ax.plot(x1, x2, x3, c="black")[0]
current_epoch = 0
history = []
plot = None


def animation(frame):
    global w, b
    w, b, w_grad, b_grad = grad_descend(
        w,
        b,
        x_train,
        y_train,
        0.19,
        regression_logistic_cost,
    )
    history.append(regression_logistic_cost(w, b, x_train, y_train))
    x3 = ((-b) - w[0] * x1 - w[1] * x2) / w[2]
    line.set_data(
        inverse_normalizer(x1, argnums=(0,)).flatten(),
        inverse_normalizer(x2, argnums=(1,)).flatten(),
    )
    line.set_3d_properties(inverse_normalizer(x3, argnums=(2,)).flatten())
    return (line,)


ani = FuncAnimation(fig, animation, frames=epoches, interval=10000 / epoches, blit=True)
ani.save("video/logistic_regression_3d.mp4")
plt.show()

plt.plot(
    history,
)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()
