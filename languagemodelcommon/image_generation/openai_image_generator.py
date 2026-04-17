from __future__ import annotations

import base64
import logging
import os
from typing import override, Literal, Optional, TYPE_CHECKING

from openai import AsyncOpenAI
from openai.types import ImagesResponse

from languagemodelcommon.image_generation.image_generator import (
    ImageGenerator,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

if TYPE_CHECKING:
    from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
        LanguageModelCommonEnvironmentVariables,
    )

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.IMAGE_GENERATION)


class OpenAIImageGenerator(ImageGenerator):
    def __init__(
        self,
        *,
        environment_variables: "LanguageModelCommonEnvironmentVariables | None" = None,
    ) -> None:
        self._environment_variables = environment_variables

    async def _invoke_model_async(
        self,
        prompt: str,
        image_size: Literal[
            "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
        ],
    ) -> bytes:
        """Asynchronous OpenAI image generation"""

        openai_api_key: Optional[str] = (
            self._environment_variables.openai_api_key
            if self._environment_variables
            else os.environ.get("OPENAI_API_KEY")
        )
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        client = AsyncOpenAI(api_key=openai_api_key)

        response: ImagesResponse = await client.images.generate(
            model="dall-e-3",  # You can change to "dall-e-2" if needed
            prompt=prompt,
            size=image_size,
            quality="standard",
            n=1,
            response_format="b64_json",
        )

        # Extract the base64 encoded image and decode
        if response.data is None or len(response.data) == 0:
            raise ValueError("Base64 image is None")

        base64_image: Optional[str] = response.data[0].b64_json
        if base64_image is None:
            raise ValueError("Base64 image is None")
        return base64.b64decode(base64_image)

    @override
    async def generate_image_async(
        self,
        *,
        prompt: str,
        style: Literal["natural", "cinematic", "digital-art", "pop-art"] = "natural",
        image_size: Literal[
            "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
        ] = "1024x1024",
    ) -> bytes:
        """Generate an image using OpenAI DALL-E"""
        if (
            self._environment_variables
            and self._environment_variables.log_input_and_output
        ):
            logger.info(f"Generating image for prompt: {prompt}")

        try:
            # Run model invocation in executor
            image_data: bytes = await self._invoke_model_async(
                prompt=prompt, image_size=image_size
            )

            if (
                self._environment_variables
                and self._environment_variables.log_input_and_output
            ):
                logger.info(f"Image generated successfully for prompt: {prompt}")

            return image_data

        except Exception as e:
            logger.error(f"Error generating image for prompt {prompt}: {str(e)}")
            logger.exception(e, stack_info=True)
            raise
