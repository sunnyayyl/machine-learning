import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.numpy import float32

import ml

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


x_train_raw = jnp.array(
    [[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]], dtype=float32
)
y_train = jnp.array([460, 232, 178], dtype=float32)

normalizer,inverse_normalizer = ml.get_mean_normalizer(x_train_raw)
x_train = normalizer(x_train_raw)
print(x_train)
w, b, history = ml.gradient_descend_training_loop(
    x_train,
    y_train,
    learning_rate=0.9,
    epoches=40,
    cost_function=ml.mean_squared_error,
    keep_cost_history=True,
)
ml.compare_predictions(
    x_train,
    y_train,
    w,
    b,
    predict_function=ml.linear_predict
)

plt.plot(
    history["cost"]
)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()
