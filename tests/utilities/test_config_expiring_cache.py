"""Async tests for ConfigExpiringCache."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from languagemodelcommon.configs.schemas.config_schema import ChatModelConfig
from languagemodelcommon.utilities.cache.config_expiring_cache import (
    ConfigExpiringCache,
)


@pytest.fixture
def sample_models() -> List[ChatModelConfig]:
    return [
        ChatModelConfig(id="model-a", name="Model A", description="Test model A"),
        ChatModelConfig(id="model-b", name="Model B", description="Test model B"),
    ]


class TestConfigExpiringCache:
    """Covers cache validity, mutation, and expiry."""

    @pytest.mark.asyncio
    async def test_set_then_get_returns_value_while_valid(
        self, sample_models: List[ChatModelConfig]
    ) -> None:
        cache = ConfigExpiringCache(ttl_seconds=10.0)
        await cache.set(sample_models)
        result = await cache.get()
        assert result == sample_models

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(
        self, sample_models: List[ChatModelConfig]
    ) -> None:
        cache = ConfigExpiringCache(ttl_seconds=0.01)
        await cache.set(sample_models)
        await asyncio.sleep(0.02)
        assert await cache.get() is None

    @pytest.mark.asyncio
    async def test_clear_resets_cache(
        self, sample_models: List[ChatModelConfig]
    ) -> None:
        cache = ConfigExpiringCache(ttl_seconds=5.0)
        await cache.set(sample_models)
        await cache.clear()
        assert await cache.get() is None

    @pytest.mark.asyncio
    async def test_create_initializes_cache(
        self, sample_models: List[ChatModelConfig]
    ) -> None:
        cache = ConfigExpiringCache(ttl_seconds=5.0)
        created = await cache.create(init_value=sample_models)
        assert created == sample_models
        assert await cache.get() == sample_models
