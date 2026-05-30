from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from languagemodelcommon.utilities.cache.advisory_lock import AdvisoryLock


class TestAdvisoryLockAcquire:
    """Lock acquired when no existing holder."""

    @pytest.mark.asyncio
    @patch("languagemodelcommon.utilities.cache.advisory_lock.uuid.uuid4")
    @patch(
        "languagemodelcommon.utilities.cache.advisory_lock.os.getpid", return_value=1234
    )
    async def test_acquires_when_no_holder(
        self, mock_getpid: MagicMock, mock_uuid: MagicMock
    ) -> None:
        mock_uuid.return_value.hex = "aabbccdd12345678"  # pragma: allowlist secret
        holder_id = "1234-aabbccdd"

        store = AsyncMock()
        store.get = AsyncMock(side_effect=[None, {"holder": holder_id}])
        store.put = AsyncMock()
        store.delete = AsyncMock()

        async with AdvisoryLock(store, "test_lock", ttl_seconds=60) as acquired:
            assert acquired is True

        store.put.assert_called_once()
        store.delete.assert_called_once()


class TestAdvisoryLockSkip:
    """Lock not acquired when another holder exists."""

    @pytest.mark.asyncio
    async def test_skips_when_lock_held(self) -> None:
        store = AsyncMock()
        store.get = AsyncMock(return_value={"holder": "other-worker"})
        store.put = AsyncMock()
        store.delete = AsyncMock()

        async with AdvisoryLock(store, "test_lock", ttl_seconds=60) as acquired:
            assert acquired is False

        store.put.assert_not_called()
        store.delete.assert_not_called()


class TestAdvisoryLockRaceCondition:
    """Lock not acquired when race is lost (read-after-write shows different holder)."""

    @pytest.mark.asyncio
    async def test_lost_race(self) -> None:
        store = AsyncMock()
        store.get = AsyncMock(side_effect=[None, {"holder": "other-winner"}])
        store.put = AsyncMock()
        store.delete = AsyncMock()

        async with AdvisoryLock(store, "test_lock", ttl_seconds=60) as acquired:
            assert acquired is False

        store.delete.assert_not_called()
