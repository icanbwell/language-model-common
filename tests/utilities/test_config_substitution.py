import pytest

from languagemodelcommon.utilities.config_substitution import substitute_env_vars


class TestSubstituteEnvVars:
    def test_string_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "hello")
        assert substitute_env_vars("${MY_VAR}") == "hello"

    def test_string_with_default_when_var_missing(self) -> None:
        result = substitute_env_vars("${MISSING_VAR:-fallback}")
        assert result == "fallback"

    def test_string_with_default_when_var_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRESENT_VAR", "real")
        result = substitute_env_vars("${PRESENT_VAR:-fallback}")
        assert result == "real"

    def test_missing_var_without_default_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing environment variable"):
            substitute_env_vars("${TOTALLY_MISSING_VAR}")

    def test_string_without_pattern_returned_unchanged(self) -> None:
        assert substitute_env_vars("no substitution") == "no substitution"

    def test_multiple_vars_in_one_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        result = substitute_env_vars("${HOST}:${PORT}")
        assert result == "localhost:8080"

    def test_dict_values_substituted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_NAME", "mydb")
        payload = {"database": "${DB_NAME}", "port": 5432}
        result = substitute_env_vars(payload=payload)
        assert result == {"database": "mydb", "port": 5432}

    def test_list_items_substituted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ITEM", "value")
        result = substitute_env_vars(payload=["${ITEM}", "static"])
        assert result == ["value", "static"]

    def test_nested_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECRET", "s3cr3t")
        payload = {"config": [{"key": "${SECRET}"}]}
        result = substitute_env_vars(payload=payload)
        assert result == {"config": [{"key": "s3cr3t"}]}

    def test_non_string_non_collection_returned_unchanged(self) -> None:
        assert substitute_env_vars(payload=42) == 42
        assert substitute_env_vars(payload=None) is None
        assert substitute_env_vars(payload=True) is True

    def test_empty_default_value(self) -> None:
        result = substitute_env_vars("${MISSING_VAR:-}")
        assert result == ""
