from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random
from jax import jit
from jaxtyping import jaxtyped, Float, Array, PRNGKeyArray, PyTree
from typeguard import typechecked

from ml.activation import linear
from ml.definition.functions import ActivationFunction


@jaxtyped(typechecker=typechecked)
@jit
def _dense(
    a_in: Float[Array, "feature_size"],
    w: Float[Array, "feature_size unit"],
) -> Float[Array, "unit"]:
    a_out = jnp.matmul(a_in, w)
    return a_out


@jaxtyped(typechecker=typechecked)
@jit
def _dense_bias(
    a_in: Float[Array, "feature_size"],
    w: Float[Array, "feature_size unit"],
    b: Float[Array, "unit"],
) -> Float[Array, "unit"]:
    a_out = jnp.matmul(a_in, w) + b
    return a_out


class Layer(ABC):
    @abstractmethod
    def __init__(self, units: int):
        self.units = units

    @abstractmethod
    def __call__(self, a_in): ...

    @abstractmethod
    def __repr__(self): ...


@jax.tree_util.register_pytree_node_class
class Dense(Layer, ABC):
    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        input_size: int,
        units: int,
        add_bias: bool = True,
        activation: ActivationFunction = linear,
        *,
        key: PRNGKeyArray,
    ):
        key, key1, key2 = jax.random.split(key, 3)
        w = jax.random.normal(key1, (units, input_size))
        # w = jnp.zeros_like(10.0, float, (units, input_size))
        if add_bias:
            b = jax.random.normal(key2, (units,))
            # b = jnp.zeros_like(20.0, float, (units,))
        else:
            b = None
        self.units = units
        self.w = w
        self.b = b
        self.add_bias = add_bias
        self.activation = activation

    @jaxtyped(typechecker=typechecked)
    def __call__(self, a_in: Float[Array, "feature_size"]) -> Float[Array, "unit"]:
        if self.add_bias:
            return self.activation(_dense_bias(a_in, self.w.T, self.b))
        else:
            return self.activation(_dense(a_in, self.w.T))

    def __repr__(self) -> str:
        if isinstance(self.w, jnp.ndarray) and isinstance(self.b, jnp.ndarray):
            if self.add_bias:
                return f"Dense(units={self.units}, w.shape={self.w.shape}, b.shape={self.b.shape}, activation={self.activation})"
            else:
                return f"Dense(units={self.units}, w.shape={self.w.shape}, activation={self.activation})"
        else:
            if self.add_bias:
                return f"Dense(units={self.units}, w.shape={self.w}, b.shape={self.b}, activation={self.activation})"
            else:
                return f"Dense(units={self.units}, w.shape={self.w}, activation={self.activation})"

    def tree_flatten(self):
        return (self.units, self.w, self.b, self.add_bias, self.activation), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        layer = cls.__new__(cls)
        layer.units, layer.w, layer.b, layer.add_bias, layer.activation = children
        return layer


@jaxtyped(typechecker=typechecked)
def filter_parameter_bool(obj: PyTree, reverse=False) -> PyTree:
    jax.tree_util.tree_map(lambda x: isinstance(x, jax.Array) != reverse, obj)


@jaxtyped(typechecker=typechecked)
def filter_parameter(obj: PyTree, reverse=False) -> PyTree:
    return jax.tree_util.tree_map(
        lambda x: x if isinstance(x, jax.Array) != reverse else None, obj
    )


@jaxtyped(typechecker=typechecked)
def partition(obj: PyTree) -> tuple[PyTree, PyTree]:
    return filter_parameter(obj), filter_parameter(obj, reverse=True)


@jaxtyped(typechecker=typechecked)
def combine(obj1: PyTree, obj2: PyTree) -> PyTree:
    return jax.tree.map(
        lambda x, y: x if x is not None else y, obj1, obj2, is_leaf=_is_none
    )


def _is_none(x):
    return x is None
