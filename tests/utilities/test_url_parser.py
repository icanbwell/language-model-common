"""Tests for UrlParser utilities."""

from __future__ import annotations

import pytest

from languagemodelcommon.utilities.url_parser import UrlParser


class TestUrlParser:
    def test_parse_s3_uri_returns_bucket_and_path(self) -> None:
        bucket, path = UrlParser.parse_s3_uri("s3://bucket-name/path/to/file.json")
        assert bucket == "bucket-name"
        assert path == "path/to/file.json"

    def test_parse_s3_uri_invalid_scheme_raises(self) -> None:
        with pytest.raises(ValueError):
            UrlParser.parse_s3_uri("https://bucket/path")

    def test_is_github_url_variants(self) -> None:
        assert UrlParser.is_github_url("https://github.com/org/repo")
        assert UrlParser.is_github_url("https://api.github.com/org/repo")
        assert not UrlParser.is_github_url("https://example.com/org/repo")

    def test_get_url_for_file_name_uses_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IMAGE_GENERATION_URL", "https://images")
        result = UrlParser.get_url_for_file_name("image.png")
        assert result == "https://images/image.png"

    def test_combine_path_strips_slashes(self) -> None:
        combined = UrlParser.combine_path("/prefix/", "/nested/file.txt")
        assert combined == "prefix/nested/file.txt"
