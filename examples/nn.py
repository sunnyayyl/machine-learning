from functools import partial

import jax
import matplotlib.pyplot as plt
import optax
from jax import random
from jaxtyping import Float, Array, PyTree, PRNGKeyArray

from ml.cost import mean_squared_error
from ml.definition import FloatScalar
from ml.nn import Dense, partition, combine
from ml.normalizer import get_z_score_normalizer
from ml.tools import generate_data, sample


@jax.tree_util.register_pytree_node_class
class Model:
    def __init__(self, key: PRNGKeyArray):
        key, key1, key2, key3, key4, key5 = random.split(key, 6)
        self.layers = [
            Dense(1, 3, key=key3, activation=jax.nn.relu),
            # Dense(25, 10, key=key1, activation=jax.nn.relu),
            Dense(3, 1, key=key4),
        ]

    def __call__(self, x: Float[Array, "feature_size"]) -> Float[Array, "unit"]:
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"Model({self.layers})"

    def tree_flatten(self):
        return self.layers, None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        del _aux_data
        model = cls.__new__(cls)
        model.layers = children
        return model


key = random.PRNGKey(123)
key, subkey = random.split(key)
model = Model(subkey)
params, static = partition(model)
print(combine(params, static))


@partial(jax.jit, static_argnames=("static",))
def loss(
    params: PyTree,
    static: PyTree,
    x: Float[Array, "batch feature"],
    y: Float[Array, "batch 1"],
) -> FloatScalar:
    combined_model = combine(params, static)
    prediction = jax.vmap(combined_model)(x)
    # return jax.numpy.abs(y - prediction).mean()
    return mean_squared_error(y, prediction)


optimizer = optax.adam(0.001)


@partial(jax.jit, static_argnames=("static",))
def train_step(param, static, x, y, opt_state):
    value, grad = jax.value_and_grad(loss)(param, static, x, y)
    update, opt_states = optimizer.update(grad, opt_state)
    grad = jax.tree.map(lambda x: jax.numpy.clip(x, -1.0, 1.0), grad)
    # jax.debug.print("{}", grad)
    """param = optax.apply_updates(param, update)"""
    # jax.debug.print("Old {}", params.layers[0].w)
    """if not jax.numpy.isnan(value).any():
        print(jax.tree.flatten(grad))
        exit()"""
    param = jax.tree.map(lambda w, g: w - 0.00001 * g, param, grad)
    # jax.debug.print("New {}", params.layers[0].w)
    # jax.debug.print("{}", update)
    #
    return (
        param,
        value,
        opt_states,
    )


def training_loop(param, static, x, y):
    history = []
    opt_state = optimizer.init(param)
    for i in range(3000000):
        param, v, opt_state = train_step(param, static, x, y, opt_state)

        if i % 1000 == 0:
            history.append(v)
            print(f"Epoch: {i} Loss: {v}")
    return param, static, history


key, x, y = generate_data(
    key,
    (60,),
    -100.0,
    100.0,
    lambda x: 3 * x + 4 * x**2 + 110,
    20,
    20,
)

"""inverse_normalizer, normalizer = get_z_score_normalizer(x)
y_inverse_noramlizer, y_normalizer = get_z_score_normalizer(y)
x_norm = normalizer(x).reshape(-1, 1)
y_norm = y_normalizer(y).reshape(-1, 1)
"""
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
print(f"Before: {loss(params, static, x, y)}")
print(jax.tree.flatten(params)[0])
plt.plot(x, (jax.vmap(model)(x)), c="yellow")

params, static, history = training_loop(params, static, x, y)
model = combine(params, static)
print(f"After: {loss(params, static, x, y)}")
print(jax.tree.flatten(params)[0])

plt.scatter(x, y)
plt.plot(x, (jax.vmap(model)(x)), c="red")

plt.show()

plt.plot(history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
