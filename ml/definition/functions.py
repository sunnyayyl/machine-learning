from typing import Protocol,Optional,TypedDict,Iterable
from jaxtyping import Float,Array

from ml.definition import FloatScalar


class History(TypedDict):
    cost: Optional[list[FloatScalar]]
    w: Optional[list[Float[Array, "feature_size"]]]
    b: Optional[list[FloatScalar]]
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
    )->FloatScalar: ...
