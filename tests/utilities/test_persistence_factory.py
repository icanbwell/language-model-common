import pytest

from languagemodelcommon.persistence.persistence_factory import PersistenceFactory
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


class TestPersistenceFactory:
    def _make_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> LanguageModelCommonEnvironmentVariables:
        monkeypatch.setenv("MONGO_LLM_STORAGE_URI", "mongodb://host:27017")
        monkeypatch.setenv("MONGO_LLM_STORAGE_DB_USERNAME", "user")
        monkeypatch.setenv("MONGO_LLM_STORAGE_DB_PASSWORD", "pass")
        monkeypatch.setenv("MONGO_LLM_STORAGE_DB_NAME", "testdb")
        monkeypatch.setenv("MONGO_LLM_STORAGE_STORE_COLLECTION_NAME", "store_col")
        monkeypatch.setenv(
            "MONGO_LLM_STORAGE_CHECKPOINTER_COLLECTION_NAME", "checkpoint_col"
        )
        return LanguageModelCommonEnvironmentVariables()

    def test_create_store_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = self._make_env(monkeypatch=monkeypatch)
        factory = PersistenceFactory(environment_variables=env)
        with factory.create_store(persistence_type="memory") as store:
            assert store is not None

    def test_create_checkpointer_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = self._make_env(monkeypatch=monkeypatch)
        factory = PersistenceFactory(environment_variables=env)
        with factory.create_checkpointer(persistence_type="memory") as checkpointer:
            assert checkpointer is not None

    def test_create_store_unknown_type_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = self._make_env(monkeypatch=monkeypatch)
        factory = PersistenceFactory(environment_variables=env)
        with pytest.raises(ValueError, match="Unknown persistence type"):
            with factory.create_store(persistence_type="redis"):
                pass

    def test_create_checkpointer_unknown_type_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = self._make_env(monkeypatch=monkeypatch)
        factory = PersistenceFactory(environment_variables=env)
        with pytest.raises(ValueError, match="Unknown persistence type"):
            with factory.create_checkpointer(persistence_type="redis"):
                pass

    def test_resolve_mongo_connection_missing_uri_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MONGO_LLM_STORAGE_URI", raising=False)
        env = LanguageModelCommonEnvironmentVariables()
        factory = PersistenceFactory(environment_variables=env)
        with pytest.raises(ValueError, match="mongo_llm_storage_uri"):
            factory._resolve_mongo_connection()

    def test_resolve_mongo_connection_missing_username_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MONGO_LLM_STORAGE_URI", "mongodb://host:27017")
        monkeypatch.delenv("MONGO_LLM_STORAGE_DB_USERNAME", raising=False)
        env = LanguageModelCommonEnvironmentVariables()
        factory = PersistenceFactory(environment_variables=env)
        with pytest.raises(ValueError, match="mongo_llm_storage_db_username"):
            factory._resolve_mongo_connection()

    def test_resolve_mongo_connection_missing_password_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MONGO_LLM_STORAGE_URI", "mongodb://host:27017")
        monkeypatch.setenv("MONGO_LLM_STORAGE_DB_USERNAME", "user")
        monkeypatch.delenv("MONGO_LLM_STORAGE_DB_PASSWORD", raising=False)
        monkeypatch.delenv("MONGO_DB_PASSWORD", raising=False)
        env = LanguageModelCommonEnvironmentVariables()
        factory = PersistenceFactory(environment_variables=env)
        with pytest.raises(ValueError, match="mongo_llm_storage_db_password"):
            factory._resolve_mongo_connection()
