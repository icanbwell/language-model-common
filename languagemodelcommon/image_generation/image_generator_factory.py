from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from languagemodelcommon.aws.aws_client_factory import AwsClientFactory
from languagemodelcommon.image_generation.image_generator import (
    ImageGenerator,
)
from languagemodelcommon.image_generation.aws_image_generator import (
    AwsImageGenerator,
)
from languagemodelcommon.image_generation.openai_image_generator import (
    OpenAIImageGenerator,
)

if TYPE_CHECKING:
    from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
        LanguageModelCommonEnvironmentVariables,
    )


class ImageGeneratorFactory:
    def __init__(
        self,
        *,
        aws_client_factory: AwsClientFactory,
        environment_variables: "LanguageModelCommonEnvironmentVariables | None" = None,
    ) -> None:
        self.aws_client_factory = aws_client_factory
        if self.aws_client_factory is None:
            raise ValueError("aws_client_factory must not be None")
        if not isinstance(self.aws_client_factory, AwsClientFactory):
            raise TypeError(
                "aws_client_factory must be an instance of AwsClientFactory"
            )
        self._environment_variables = environment_variables

    # noinspection PyMethodMayBeStatic
    def get_image_generator(
        self, *, model_name: Literal["aws", "openai"]
    ) -> ImageGenerator:
        match model_name:
            case "aws":
                return AwsImageGenerator(
                    aws_client_factory=self.aws_client_factory,
                    environment_variables=self._environment_variables,
                )
            case "openai":
                return OpenAIImageGenerator(
                    environment_variables=self._environment_variables,
                )
            case _:
                raise ValueError(f"Unsupported model_name: {model_name}")
