import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.numpy import float32

from ml.cost import regression_mean_squared_error
from ml.gradient_descent import gradient_descend_training_loop
from ml.normalizer import get_mean_normalizer
from ml.predict import predict
from ml.tools import compare_predictions

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


x_train_raw = jnp.array(
    [[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]], dtype=float32
)
y_train = jnp.array([460, 232, 178], dtype=float32)

normalizer, inverse_normalizer = get_mean_normalizer(x_train_raw)
x_train = normalizer(x_train_raw)
print(x_train)
w, b, history = gradient_descend_training_loop(
    x_train,
    y_train,
    learning_rate=0.9,
    epoches=40,
    cost_function=regression_mean_squared_error,
    keep_cost_history=True,
)
compare_predictions(x_train, y_train, w, b, predict_batch_function=predict)

plt.plot(history["cost"])
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()
