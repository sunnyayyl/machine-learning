from abc import ABC, abstractmethod
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked

from ml.activation import linear
from ml.definition.functions import ActivationFunction
from ml.predict import predict


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames=("activation",))
def _dense(
    a_in: Float[Array, "feature_size"],
    w: Float[Array, "unit feature_size"],
    b: Float[Array, "unit"],
    activation: ActivationFunction = linear,
) -> Float[Array, "unit"]:
    a_out = vmap(partial(predict, x=a_in, activation_function=activation))(w=w, b=b)
    return a_out


class Layer(ABC):
    @abstractmethod
    def __init__(self, units: int):
        self.units = units

    @abstractmethod
    def init(self, last_layer): ...

    @abstractmethod
    def __call__(self, a_in): ...

    @abstractmethod
    def jacfwd(self, a_in): ...

    @abstractmethod
    def __repr__(self): ...


class Dense(Layer):
    units: int

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        units: int,
        feature_size: Optional[int] = None,
        initial_w: Optional[Float[Array, "unit feature_size"]] = None,
        initial_b: Optional[Float[Array, "unit"]] = None,
        activation: ActivationFunction = linear,
    ):
        self.units = units
        if feature_size is None and initial_w is None:
            self.need_initial_w = True
        else:
            self.need_initial_w = False
            self.w = (
                initial_w
                if initial_w is not None
                else jnp.zeros((self.units, feature_size))
            )

        self.b = initial_b if initial_b is not None else jnp.zeros((self.units,))
        self.activation = activation

    def init(self, last_layer: Layer):
        if self.need_initial_w:
            self.w = jnp.zeros((self.units, last_layer.units))

    @jaxtyped(typechecker=typechecked)
    @partial(jit, static_argnames="self")
    def __call__(self, a_in: Float[Array, "feature_size"]) -> Float[Array, "unit"]:
        return _dense(a_in, self.w, self.b, self.activation)

    @jaxtyped(typechecker=typechecked)
    @partial(jit, static_argnames="self")
    def jacfwd(
        self, a_in: Float[Array, "feature_size"]
    ) -> tuple[Float[Array, "unit feature_size"], Float[Array, "unit"]]:
        return jax.jacfwd(_dense, argnums=(1, 2))(a_in, self.w, self.b, self.activation)

    def __repr__(self) -> str:
        return f"Dense(units={self.units}, w.shape={self.w.shape}, b.shape={self.b.shape}, activation={self.activation})"
