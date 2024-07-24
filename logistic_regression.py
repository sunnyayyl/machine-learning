import jax.numpy as jnp
import matplotlib.pyplot as plt

import ml

x_raw = jnp.array(
    [[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]],
)
y_train = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.float32)  # .reshape(-1, 1)

x_train = x_raw
"""normalizer = ml.get_z_score_normalizer(x_raw)
x_train = normalizer(x_raw)"""

w, b, history = ml.gradient_descend_training_loop(
    x_train,
    y_train,
    epoches=500,
    learning_rate=0.1,
    verbose=True,
    cost_function=ml.logistic_cost,
    cost_history=True,
    predict_function=ml.logistic_predict_all,
)
print(f"w: {w} b: {b}")
plt.plot(
    history,
)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()
prediction = ml.logistic_predict_all(x_train, w, b)
plt.scatter(x_raw[:, 0], x_raw[:, 1], c=y_train)
x1 = jnp.arange(-5, 5)
x2 = ((-b) - x1 * w[0]) / w[1]
print(f"x1: {x1} x2: {x2}")
plt.plot(x1, x2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
for i in range(x_raw.shape[0]):
    x = jnp.array(x_raw[i])
    print(f"Predicted: {ml.logistic_predict(x, w, b)}, Target: {y_train[i]}")
