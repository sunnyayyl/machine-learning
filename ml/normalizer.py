from collections.abc import Iterable
from functools import partial
from typing import Optional

import jax.numpy as jnp
from jax import jit
from jaxtyping import jaxtyped, Float, Array, ArrayLike
from typeguard import typechecked

from ml.definition.functions import NormalizerFunction


@jaxtyped(typechecker=typechecked)
@jit
def __invert_normalizer(
    x: Float[ArrayLike, "..."],
    argnums: Optional[Iterable[int]] = None,
    *,
    x_train_mean,
    divisor,
) -> Float[Array, "..."]:
    if argnums is not None:
        argnums = jnp.array(argnums)
        divisor = jnp.take(divisor, argnums, axis=0)
        x_train_mean = jnp.take(x_train_mean, argnums, axis=0)
    return x * divisor + x_train_mean


@jaxtyped(typechecker=typechecked)
@jit
def __normalizer(
    x: Float[ArrayLike, "..."],
    argnums: Optional[Iterable[int]] = None,
    *,
    x_train_mean,
    divisor,
) -> Float[Array, "..."]:
    if argnums is not None:
        argnums = jnp.array(argnums)
        divisor = jnp.take(divisor, argnums, axis=0)
        x_train_mean = jnp.take(x_train_mean, argnums, axis=0)
    return (x - x_train_mean) / divisor


@jaxtyped(typechecker=typechecked)
def get_z_score_normalizer(
    training_data: Float[Array, "data_count feature_size"]
) -> tuple[NormalizerFunction, NormalizerFunction]:
    x_train_mean = jnp.mean(training_data, axis=0)
    x_train_std = jnp.std(training_data, axis=0)

    return jit(
        partial(__normalizer, x_train_mean=x_train_mean, divisor=x_train_std)
    ), jit(partial(__invert_normalizer, x_train_mean=x_train_mean, divisor=x_train_std))


@jaxtyped(typechecker=typechecked)
def get_mean_normalizer(
    training_data: Float[Array, "..."]
) -> tuple[NormalizerFunction, NormalizerFunction]:
    x_train_mean = jnp.mean(training_data, axis=0)
    difference = jnp.max(training_data, axis=0) - jnp.min(training_data, axis=0)
    return jit(
        partial(__normalizer, x_train_mean=x_train_mean, divisor=difference)
    ), jit(partial(__invert_normalizer, x_train_mean=x_train_mean, divisor=difference))
