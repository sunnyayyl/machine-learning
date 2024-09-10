import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked


@jit
def softmax(z: Float[Array, "..."]) -> Float[Array, "..."]:
    z = z - jnp.max(z, axis=-1, keepdims=True)
    return jnp.exp(z) / jnp.sum(jnp.exp(z))


@jaxtyped(typechecker=typechecked)
@jit
def linear(
    z: Float[Array, "..."],
) -> Float[Array, "..."]:
    return z


@jaxtyped(typechecker=typechecked)
@jit
def sigmoid(z: Float[Array, "..."]) -> Float[Array, "..."]:
    # return 1 / (1 + jnp.exp(-z))
    return jax.lax.logistic(z)


@jax.custom_jvp
@jit
def relu(z: Float[Array, "..."]) -> Float[Array, "..."]:
    return jax.lax.max(0.0, z)


relu.defjvps(lambda g, ans, x: jax.lax.select(x > 0, g, jax.lax.full_like(g, 0.0)))
