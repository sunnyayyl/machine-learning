from functools import partial
from typing import Optional, Protocol, Callable, Iterable, TypedDict

import jax.numpy as jnp
from jax import jit, vmap, grad, jacfwd, random
from jaxtyping import Array, Float, jaxtyped, ArrayLike
from typeguard import typechecked






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
@partial(jit, static_argnames="predict_function")
def l2_regularization(
        w: Float[Array, "feature_size"],
        b: FloatScalar,
        x_train: Float[Array, "data_count feature_size"],
       y_train: Float[Array, "data_count"],
       predict_function: Optional[PredictFunction] = None,
       *,
       cost_function: CostFunction,
       lambda_: FloatScalar,) -> Float[Array,"input_size"]:
    return cost_function(w,b,x_train,y_train,predict_function)+(jnp.mean(jnp.pow(w,2)))*lambda_

@jaxtyped(typechecker=typechecked)
@partial(jit, static_argnames="predict_function")
def no_regularization(
        w: Float[Array, "feature_size"],
        b: FloatScalar,
        x_train: Float[Array, "data_count feature_size"],
       y_train: Float[Array, "data_count"],
       predict_function: Optional[PredictFunction] = None,
       *,
       cost_function: CostFunction,
       lambda_: FloatScalar,) -> Float[Array,"input_size"]:
    return cost_function(w,b,x_train,y_train,predict_function)+(jnp.mean(jnp.pow(w,2)))*lambda_

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
@partial(jit, static_argnames=("cost_function","regularization_function"))
def grad_descend(
    w: Float[Array, "feature_size"],
    b: FloatScalar,
    x_train: Float[Array, "data_count feature_size"],
    y_train: Float[Array, "data_count"],
    learning_rate: FloatScalar,
    cost_function: CostFunction,
    regularization_function: RegularizationFunction=no_regularization,
    lambda_: FloatScalar=0,
) -> tuple[
    Float[Array, "feature_size"], FloatScalar, Float[Array, "feature_size"], FloatScalar
]:
    new_function=regularization_function(w,b,x_train,y_train,cost_function=cost_function,lambda_=lambda_)
    w_grad = jacfwd(lambda w: new_function(w, b, x_train, y_train))(w)
    b_grad = grad(new_function, argnums=1)(w, b, x_train, y_train)
    temp_w = w - learning_rate * w_grad
    temp_b = b - learning_rate * b_grad
    return temp_w, temp_b, w_grad, b_grad


def gradient_descend_training_loop(
    x_train: Array,
    y_train: Array,
    *,
    w: Optional[Array] = None,
    b: Optional[FloatScalar] = None,
    learning_rate: float,
    epoches: int,
    cost_function: CostFunction,
    predict_function: Optional[PredictFunction] = None,
    verbose: bool = False,
    keep_cost_history: bool = False,
    keep_parameter_history: bool = False,
    callback: CallbackFunction = default_callback,
    regularization_function: RegularizationFunction=no_regularization,
    lambda_: FloatScalar=0.0,
) -> tuple[Array, FloatScalar, History]:
    if w is None:
        w = jnp.zeros(x_train.shape[1], dtype=float)
    if b is None:
        b = 0.0
    if predict_function is not None:
        cost_function = jit(partial(cost_function, predict_function=predict_function))
    cost_history=[cost_function(w, b, x_train, y_train)] if keep_cost_history else None
    w_history=[w] if keep_parameter_history else None
    b_history=[b] if keep_parameter_history else None
    for epoch in range(epoches):
        w, b, w_grad, b_grad = grad_descend(
            w,
            b,
            x_train,
            y_train,
            learning_rate,
            cost_function,
            regularization_function=regularization_function,
            lambda_=lambda_,
        )
        if verbose:
            print(f"Epoch {epoch} w: {w} b:{b} w_grad: {w_grad} b_grad: {b_grad}")
        if keep_cost_history:
            cost_history.append(cost_function(w, b, x_train, y_train))
        if keep_parameter_history:
            w_history.append(w)
            b_history.append(b)


        callback(w, b)
    return w, b, History(cost=cost_history,w=w_history,b=b_history)


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

def compare_predictions(
        x_train: Array,
        y_train: Array,
        w: Array ,
        b: FloatScalar,
        *,
        predict_function: PredictFunction
):
    for i in range(x_train.shape[0]):
        x = jnp.array(x_train[i])
        print(f"Predicted: {predict_function(x, w, b)}, Target: {y_train[i]}")
