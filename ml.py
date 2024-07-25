from functools import partial
from typing import Optional, Protocol, Callable, Iterable

import jax.numpy as jnp
from jax import jit, vmap, grad, jacfwd, random
from jaxtyping import Array, Float, jaxtyped, ArrayLike
from typeguard import typechecked

FloatScalar = Float[ArrayLike, ""]


class PredictFunction(Protocol):
    def __call__(
        self,
        x: Float[Array, "data_count feature_size"],
        w: Float[Array, "feature_size"],
        b: FloatScalar,
    ) -> Float[Array, "data_count"]: ...


class NormalizerFunction(Protocol):
    def __call__(
        self, x: Float[ArrayLike, "..."], argnums: Optional[Iterable[int]] = None
    ) -> Float[Array, "..."]: ...


class CostFunction(Protocol):
    def __call__(
        self,
        w: Float[Array, "feature_size"],
        b: FloatScalar,
        x_train: Float[Array, "data_count feature_size"],
        y_train: Float[Array, "data_count"],
        predict_function: PredictFunction,
    ) -> FloatScalar: ...


class CallbackFunction(Protocol):
    def __call__(
        self,
        w: Float[Array, "feature_size"],
        b: FloatScalar,
    ): ...


@jaxtyped(typechecker=typechecked)
def default_callback(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
): ...


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


@jaxtyped(typechecker=typechecked)
@jit
def linear_predict(
    x: Float[Array, "feature_size"], w: Float[Array, "feature_size"], b: FloatScalar
) -> FloatScalar:
    return jnp.dot(w, x) + b


@jaxtyped(typechecker=typechecked)
@jit
def linear_predict_all(
    x: Float[Array, "data_count feature_size"],
    w: Float[Array, "feature_size"],
    b: FloatScalar,
) -> Float[Array, "data_count"]:
    return vmap(lambda x: linear_predict(x, w, b))(x)


@jaxtyped(typechecker=typechecked)
@jit
def sigmoid(x: FloatScalar) -> FloatScalar:
    return 1 / (1 + jnp.exp(-x))


@jaxtyped(typechecker=typechecked)
@jit
def logistic_predict(
    x: Float[Array, "feature_size"], w: Float[Array, "feature_size"], b: FloatScalar
) -> FloatScalar:
    return sigmoid(linear_predict(x, w, b))


@jaxtyped(typechecker=typechecked)
@jit
def logistic_predict_all(
    x: Float[Array, "data_count feature_size"],
    w: Float[Array, "feature_size"],
    b: FloatScalar,
) -> Float[Array, "data_count"]:
    return vmap(lambda x: logistic_predict(x, w, b))(x)


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def logistic_cost(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    predict_function: PredictFunction = logistic_predict_all,
) -> FloatScalar:
    y_predict = vmap(lambda x: predict_function(x_train, w, b))(x_train)
    return jnp.mean(
        -y_train * jnp.log(y_predict) - (1 - y_train) * jnp.log(1 - y_predict)
    )


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def mean_squared_error(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    predict_function: PredictFunction = linear_predict_all,
) -> FloatScalar:
    y_predict = predict_function(x_train, w, b)
    return jnp.mean((y_train - y_predict) ** 2)


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="cost_function")
def grad_descend(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    learning_rate: FloatScalar,
    cost_function: CostFunction,
) -> tuple[
    Float[Array, "feature_size"], FloatScalar, Float[Array, "feature_size"], FloatScalar
]:
    w_grad = jacfwd(lambda w: cost_function(w, b, x_train, y_train))(w)
    b_grad = grad(cost_function, argnums=1)(w, b, x_train, y_train)
    temp_w = w - learning_rate * w_grad
    temp_b = b - learning_rate * b_grad
    return temp_w, temp_b, w_grad, b_grad


def gradient_descend_training_loop(
    x_train: Array,
    y_train: Array,
    *,
    w: Optional[Array] = None,
    b: Optional[Float] = None,
    learning_rate: float,
    epoches: int,
    cost_function: CostFunction,
    predict_function: Optional[PredictFunction] = None,
    verbose: bool = False,
    cost_history: bool = False,
    callback: CallbackFunction = default_callback,
) -> tuple[Array, FloatScalar, Optional[list[FloatScalar]]]:
    gradient_descend = GradientDescentTrainingLoop(
        x_train,
        y_train,
        w=w,
        b=b,
        learning_rate=learning_rate,
        cost_function=cost_function,
        predict_function=predict_function,
        verbose=verbose,
        cost_history=cost_history,
    )
    for _ in range(epoches):
        gradient_descend.next_epoch()
        callback(gradient_descend.w, gradient_descend.b)
    return gradient_descend.w, gradient_descend.b, gradient_descend.get_cost_history()


class GradientDescentTrainingLoop:
    def __init__(
        self,
        x_train: Array,
        y_train: Array,
        *,
        w: Optional[Array] = None,
        b: Optional[Float] = None,
        learning_rate: float,
        cost_function: CostFunction,
        predict_function: Optional[PredictFunction] = None,
        verbose: bool = False,
        cost_history: bool = False,
    ):
        if w is None:
            w = jnp.zeros(x_train.shape[1], dtype=float)
        if b is None:
            b = 0.0
        self.x_train = x_train
        self.y_train = y_train

        self.w = w

        self.b = b
        self.learning_rate = learning_rate
        self.current_epoches = 0
        self.cost_function = cost_function
        self.predict_function = predict_function
        self.verbose = verbose
        self.cost_history = cost_history
        self.history = []

    def next_epoch(self):
        self.w, self.b, w_grad, b_grad = grad_descend(
            self.w,
            self.b,
            self.x_train,
            self.y_train,
            self.learning_rate,
            self.cost_function,
        )
        if self.verbose:
            print(
                f"Epoch {self.current_epoches} w: {self.w} b:{self.b} w_grad: {w_grad} b_grad: {b_grad}"
            )
        if self.cost_history:
            self.history.append(
                self.cost_function(self.w, self.b, self.x_train, self.y_train)
            )
        self.current_epoches += 1

    def get_cost_history(self) -> Optional[list[FloatScalar]]:
        if self.cost_history:
            return self.history
        else:
            return None


@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames=("shape", "y_function"))
def generate_data(
    key: ArrayLike,
    shape: Iterable[int],
    minval: ArrayLike,
    maxval: ArrayLike,
    y_function: Callable[[ArrayLike], ArrayLike],
) -> tuple[Float[Array, "input_size"], Float[Array, "input_size"]]:
    _, subkey = random.split(key)
    x = jnp.sort(random.uniform(key=subkey, shape=shape, minval=minval, maxval=maxval))
    y = y_function(x)
    return x, y
