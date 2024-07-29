import typing

import jaxtyping

FloatScalar = jaxtyping.Float[jaxtyping.ArrayLike, ""]


class History(typing.TypedDict):
    cost: typing.Optional[list[FloatScalar]]
    w: typing.Optional[list[jaxtyping.Float[jaxtyping.Array, "feature_size"]]]
    b: typing.Optional[list[FloatScalar]]
