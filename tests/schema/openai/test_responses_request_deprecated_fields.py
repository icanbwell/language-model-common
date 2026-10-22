"""`thread_id` and `user_id` on ResponsesRequest are deprecated (BAI-1256)."""

import warnings
from typing import Any

import pytest

from languagemodelcommon.schema.openai.responses import ResponsesRequest


def _payload(**extra: Any) -> dict[str, Any]:
    return {"model": "m", "input": "hi", **extra}


class TestDeprecatedRequestFields:
    @pytest.mark.parametrize("field", ["thread_id", "user_id"])
    def test_supplying_a_deprecated_field_warns_and_is_still_accepted(
        self, field: str
    ) -> None:
        with pytest.warns(DeprecationWarning, match=field):
            request = ResponsesRequest.model_validate(_payload(**{field: "value"}))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert getattr(request, field) == "value"

    def test_supplying_both_warns_for_each(self) -> None:
        with pytest.warns(DeprecationWarning) as recorded:
            ResponsesRequest.model_validate(_payload(thread_id="t", user_id="u"))

        messages = " ".join(str(w.message) for w in recorded)
        assert "thread_id" in messages
        assert "user_id" in messages

    def test_request_without_deprecated_fields_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ResponsesRequest.model_validate(_payload())

    @pytest.mark.parametrize("field", ["thread_id", "user_id"])
    def test_json_schema_marks_the_field_deprecated(self, field: str) -> None:
        schema = ResponsesRequest.model_json_schema()

        assert schema["properties"][field]["deprecated"] is True
