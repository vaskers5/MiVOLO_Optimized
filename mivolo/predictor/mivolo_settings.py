from typing import Union

import torch
from pydantic import Field, field_validator

from mivolo.common_types import FrozenModel, NonEmptyStr


class MivoloSettings(FrozenModel):
    detector_weights: NonEmptyStr
    checkpoint: NonEmptyStr
    device: str = Field(None, validate_default=True)
    with_persons: bool = True
    disable_faces: bool = False
    draw: bool = False
    batch_size: int = Field(default=1, ge=1)

    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, value: Union[str, None]) -> str:
        if value is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return value
