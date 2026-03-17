"""Tests for the TokenReducer utility."""

from __future__ import annotations


from languagemodelcommon.utilities.auth.token_reducer import TokenReducer


def _long_text() -> str:
    return " ".join(str(i) for i in range(40))


class TestTokenReducer:
    """TokenReducer behaviour tests."""

    def test_reduce_tokens_truncates_from_end(self) -> None:
        reducer = TokenReducer(truncation_strategy="end")
        text = _long_text()
        tokens = reducer.encoding.encode(text)
        reduced = reducer.reduce_tokens(text, max_tokens=10)
        assert reducer.encoding.encode(reduced) == tokens[:10]

    def test_reduce_tokens_truncates_from_beginning(self) -> None:
        reducer = TokenReducer(truncation_strategy="beginning")
        text = _long_text()
        tokens = reducer.encoding.encode(text)
        reduced = reducer.reduce_tokens(text, max_tokens=12)
        assert reducer.encoding.encode(reduced) == tokens[-12:]

    def test_reduce_tokens_smart_preserves_start_and_end(self) -> None:
        reducer = TokenReducer(truncation_strategy="smart")
        text = _long_text()
        tokens = reducer.encoding.encode(text)
        reduced = reducer.reduce_tokens(text, max_tokens=14, preserve_start=4)
        reduced_tokens = reducer.encoding.encode(reduced)
        assert reduced_tokens[:4] == tokens[:4]
        assert reduced_tokens[4:] == tokens[-10:]

    def test_reduce_tokens_smart_defaults_to_end_when_preserve_invalid(self) -> None:
        reducer = TokenReducer(truncation_strategy="smart")
        text = _long_text()
        tokens = reducer.encoding.encode(text)
        reduced = reducer.reduce_tokens(text, max_tokens=8, preserve_start=20)
        assert reducer.encoding.encode(reduced) == tokens[:8]

    def test_reduce_tokens_returns_original_when_within_limit(self) -> None:
        reducer = TokenReducer()
        text = "short"
        assert reducer.reduce_tokens(text, max_tokens=100) == text

    def test_count_tokens_matches_encoding_length(self) -> None:
        reducer = TokenReducer()
        text = "count me"
        assert reducer.count_tokens(text) == len(reducer.encoding.encode(text))

    def test_unknown_model_uses_default_encoding(self) -> None:
        reducer = TokenReducer(model="definitely-unknown-model")
        original_tokens = len(reducer.encoding.encode("abcdefghij"))
        reduced = reducer.reduce_tokens("abcdefghij", max_tokens=3)
        reduced_token_count = len(reducer.encoding.encode(reduced))
        assert reduced_token_count == min(3, original_tokens)
