from abc import abstractmethod, ABC
from typing import Dict

from starlette.responses import StreamingResponse, JSONResponse

from languagemodelcommon.schema.openai.image_generation import (
    ImageGenerationRequest,
)


class BaseImageGenerationProvider(ABC):
    @abstractmethod
    async def generate_image_async(
        self,
        *,
        image_generation_request: ImageGenerationRequest,
        headers: Dict[str, str],
    ) -> StreamingResponse | JSONResponse: ...
