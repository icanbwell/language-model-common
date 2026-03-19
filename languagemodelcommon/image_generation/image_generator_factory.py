from typing import Literal

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


class ImageGeneratorFactory:
    def __init__(self, *, aws_client_factory: AwsClientFactory) -> None:
        self.aws_client_factory = aws_client_factory
        if self.aws_client_factory is None:
            raise ValueError("aws_client_factory must not be None")
        if not isinstance(self.aws_client_factory, AwsClientFactory):
            raise TypeError(
                "aws_client_factory must be an instance of AwsClientFactory"
            )

    # noinspection PyMethodMayBeStatic
    def get_image_generator(
        self, *, model_name: Literal["aws", "openai"]
    ) -> ImageGenerator:
        match model_name:
            case "aws":
                return AwsImageGenerator(aws_client_factory=self.aws_client_factory)
            case "openai":
                return OpenAIImageGenerator()
            case _:
                raise ValueError(f"Unsupported model_name: {model_name}")
