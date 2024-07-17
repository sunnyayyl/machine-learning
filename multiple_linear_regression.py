import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.numpy import float32

from ml import get_mean_normalizer, grad_decend, cost

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


x_train_raw = jnp.array(
    [[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]], dtype=float32
)
y_train = jnp.array([460, 232, 178], dtype=float32)

get_normalizer = get_mean_normalizer
normalizer = get_normalizer(x_train_raw)
x_train = normalizer(x_train_raw)
epoches = 40
print(x_train)
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
