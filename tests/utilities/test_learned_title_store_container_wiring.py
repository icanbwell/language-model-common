import pytest
from simple_container.container.simple_container import SimpleContainer

from languagemodelcommon.container.container_factory import (
    LanguageModelCommonContainerFactory,
)
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.utilities.learned_title_store import LearnedTitleStore
from languagemodelcommon.utilities.tool_title_warmer import ToolTitleWarmer


def _container() -> SimpleContainer:
    return LanguageModelCommonContainerFactory.register_services_in_container(
        container=SimpleContainer()
    )


class TestLearnedTitleStoreContainerWiring:
    def test_no_mongo_uri_registers_no_store_and_handler_still_resolves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MONGO_LLM_STORAGE_URI", raising=False)
        monkeypatch.delenv("MONGO_URI", raising=False)
        container = _container()

        assert container.resolve(LearnedTitleStore) is None
        assert isinstance(container.resolve(ToolEventHandler), ToolEventHandler)

    def test_mongo_uri_registers_a_store_from_env_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MONGO_LLM_STORAGE_URI", "mongodb://localhost:27017")
        monkeypatch.setenv("LEARNED_TITLE_STORE_COLLECTION", "titles_under_test")
        container = _container()

        store = container.resolve(LearnedTitleStore)

        assert isinstance(store, LearnedTitleStore)
        assert store._collection == "titles_under_test"
        assert container.resolve(LearnedTitleStore) is store


class TestToolTitleWarmerWiring:
    def test_no_mongo_uri_registers_no_warmer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MONGO_LLM_STORAGE_URI", raising=False)
        monkeypatch.delenv("MONGO_URI", raising=False)

        assert _container().resolve(ToolTitleWarmer) is None

    def test_mongo_uri_registers_a_warmer_with_env_bounds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MONGO_LLM_STORAGE_URI", "mongodb://localhost:27017")
        monkeypatch.setenv("TOOL_TITLE_WARM_MAX_TITLES", "7")

        warmer = _container().resolve(ToolTitleWarmer)

        assert isinstance(warmer, ToolTitleWarmer)
        assert warmer._max_titles == 7
