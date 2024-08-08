import jax.numpy as jnp

from ml.activation import sigmoid, relu, linear
from ml.nn import Dense

"""def sequential(inputs, w1, w2, b1, b2):
    a1 = _dense(inputs, w1, b1, activation=relu)
    a2 = _dense(a1, w2, b2, activation=sigmoid)
    return a2


a_out = sequential(
    jnp.zeros(4),
    jnp.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=float),
    jnp.zeros((1, 2)),
    jnp.zeros(2),
    jnp.zeros(1),
)
print(a_out)
grad = jax.jacfwd(sequential, argnums=(1, 2, 3, 4))(
    jnp.zeros(4),
    jnp.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=float),
    jnp.zeros((1, 2)),
    jnp.zeros(2),
    jnp.zeros(1),
)
print(grad)
"""
a = jnp.zeros_like(-100, float, (4,))
layers = [
    Dense(4, 4, activation=linear),
    Dense(2, activation=relu),
    Dense(1, activation=sigmoid),
]
for i, v in enumerate(layers):
    if i > 0:
        v.init(layers[i - 1])
    print(v)
    a = v(a)
    print(a)
