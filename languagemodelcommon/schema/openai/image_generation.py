# Copied from openai/resources/images.py
from typing import TypedDict, Union, Optional, Literal
from openai import NotGiven

# noinspection PyProtectedMember
from openai._types import Headers, Query, Body
import httpx
from openai.types import ImageModel


class ImageGenerationRequest(TypedDict, total=False):
    prompt: str
    model: Union[str, ImageModel, None] | NotGiven
    n: Optional[int] | NotGiven
    quality: Literal["standard", "hd"] | NotGiven
    response_format: Optional[Literal["url", "b64_json"]] | NotGiven
    size: (
        Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]]
        | NotGiven
    )
    style: Optional[Literal["vivid", "natural"]] | NotGiven
    user: str | NotGiven
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Headers | None
    extra_query: Query | None
    extra_body: Body | None
    timeout: float | httpx.Timeout | None | NotGiven
