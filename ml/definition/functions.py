from typing import Protocol, Optional, Iterable

from jaxtyping import Float, Array, ArrayLike

from ml.definition import FloatScalar


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
        predict_function: Optional[PredictFunction] = None,
    ) -> FloatScalar: ...


class CallbackFunction(Protocol):
    def __call__(
        self,
        w: Float[Array, "feature_size"],
        b: FloatScalar,
    ): ...


class RegularizationFunction(Protocol):
    def __call__(
        self,
        w: Float[Array, "feature_size"],
        b: FloatScalar,
        x_train: Float[Array, "data_count feature_size"],
        y_train: Float[Array, "data_count"],
        predict_function: Optional[PredictFunction] = None,
        *,
        cost_function: CostFunction,
        lambda_: FloatScalar,
    ) -> FloatScalar: ...
