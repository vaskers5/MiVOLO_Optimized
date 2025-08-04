import re
from typing import Annotated, AsyncGenerator, Generator

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    HttpUrl,
    StringConstraints,
    TypeAdapter,
)
from s3path import PureS3Path

http_url_adapter = TypeAdapter(HttpUrl)
str_adapter = TypeAdapter(str)

HttpUrlStr = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

type YieldFixture[T] = Generator[T, None, None]
type AsyncYieldFixture[T] = AsyncGenerator[T, None]


def str_to_pattern(value: str) -> re.Pattern:
    return re.compile(str_adapter.validate_python(value), flags=re.IGNORECASE)


CaseInsensitivePattern = Annotated[re.Pattern, BeforeValidator(str_to_pattern)]
S3UriPath = Annotated[
    str, AfterValidator(lambda value: PureS3Path.from_uri(value).as_uri())
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)
