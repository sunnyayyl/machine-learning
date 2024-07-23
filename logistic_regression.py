import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import ml

x_raw = jnp.array(
    [[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]],
)
normalizer = ml.get_z_score_normalizer(x_raw)
x_train = normalizer(x_raw)
y_train = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.float32)  # .reshape(-1, 1)

w, b, history = ml.gradient_descend_training_loop(
    x_train,
    y_train,
    epoches=500,
    learning_rate=0.1,
    verbose=True,
    cost_function=ml.logistic_cost,
    cost_history=True,
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
plt.plot(jax.vmap(lambda x: jax.numpy.pow(w, -1.0) * (-b))(jax.numpy.arange(0, 3)))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
for i in range(x_raw.shape[0]):
    x = jnp.array(x_raw[i])
    x = normalizer(x)
    print(f"Predicted: {ml.logistic_predict(x, w, b)}, Target: {y_train[i]}")
