import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from key_value.aio.protocols.key_value import AsyncKeyValueProtocol

from languagemodelcommon.configs.config_reader.override_read_result import (
    OverrideOutcome,
    OverrideReadResult,
)
from languagemodelcommon.configs.schemas.config_schema import ChatModelConfig
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.ref_hash import compute_short_ref_hash

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)

# Soft TTL for an override whose MCP server refs could not be resolved: short,
# so it re-resolves soon, without ever re-reading GitHub per request.
UNRESOLVED_MCP_SOFT_TTL_SECONDS = 60


@dataclass(frozen=True, slots=True)
class CachedOverride:
    result: OverrideReadResult
    age_seconds: float
    is_fresh: bool
    soft_ttl_seconds: float
    hard_expires_at: float


class ClientOverrideConfigCache:
    """Per-client override config rows in the shared model-config store.

    Rows live in the same collection as the base model-config cache but in a
    separate key namespace (``v{N}:ovr:{ref_hash}:{client_id}``). ``ovr`` is
    never a 12-hex ref hash, so the base reader's ``v{N}:{ref_hash}:`` prefix
    scan can never return an override row, while ``ConfigReader.clear_cache``
    (prefix ``v{N}:``) still removes them. No in-process state.

    ``client_id`` must already have passed ``ConfigReader._is_valid_client_id``.
    """

    NAMESPACE = "ovr"

    def __init__(
        self,
        *,
        store: AsyncKeyValueProtocol,
        collection: str,
        schema_version: int,
        environment_variables: LanguageModelCommonEnvironmentVariables,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._collection = collection
        self._schema_version = schema_version
        self._environment_variables = environment_variables
        self._clock = clock

    @property
    def store(self) -> AsyncKeyValueProtocol:
        return self._store

    def build_key(self, *, config_path: str, client_id: str) -> str:
        ref_hash = compute_short_ref_hash(config_path)
        return f"v{self._schema_version}:{self.NAMESPACE}:{ref_hash}:{client_id}"

    async def get(self, *, config_path: str, client_id: str) -> CachedOverride | None:
        key = self.build_key(config_path=config_path, client_id=client_id)
        row = await self._store.get(key, collection=self._collection)
        if row is None:
            return None
        if row.get("client_id") != client_id:
            logger.warning(
                "override_cache row client mismatch for key %s, treating as miss", key
            )
            return None
        status = row.get("status")
        age = max(0.0, self._clock() - float(row["fetched_at"]))
        soft_ttl = float(row["soft_ttl_seconds"])
        if status == OverrideOutcome.ABSENT.value:
            result = OverrideReadResult(outcome=OverrideOutcome.ABSENT)
        elif status == OverrideOutcome.FOUND.value:
            models = tuple(
                ChatModelConfig.model_validate(model) for model in row["models"]
            )
            result = OverrideReadResult(
                outcome=OverrideOutcome.FOUND,
                models=models,
                has_unresolved_mcp=bool(row.get("has_unresolved_mcp", False)),
            )
        else:
            return None
        return CachedOverride(
            result=result,
            age_seconds=age,
            is_fresh=age < soft_ttl,
            soft_ttl_seconds=soft_ttl,
            hard_expires_at=float(row["hard_expires_at"]),
        )

    async def put(
        self, *, config_path: str, client_id: str, result: OverrideReadResult
    ) -> None:
        """Write a ``FOUND`` or ``ABSENT`` result. Other outcomes are rejected."""
        env = self._environment_variables
        if result.outcome == OverrideOutcome.ABSENT:
            soft_ttl = float(env.model_config_override_cache_negative_ttl_seconds)
            hard_ttl = soft_ttl
        elif result.outcome == OverrideOutcome.FOUND:
            soft_ttl = float(
                UNRESOLVED_MCP_SOFT_TTL_SECONDS
                if result.has_unresolved_mcp
                else env.model_config_override_cache_soft_ttl_seconds
            )
            hard_ttl = float(env.model_config_override_cache_hard_ttl_seconds)
        else:
            raise ValueError(
                f"Only FOUND and ABSENT overrides may be cached, got {result.outcome}"
            )
        now = self._clock()
        await self._write(
            config_path=config_path,
            client_id=client_id,
            result=result,
            fetched_at=now,
            soft_ttl=soft_ttl,
            hard_expires_at=now + hard_ttl,
        )

    async def defer_refresh(
        self,
        *,
        config_path: str,
        client_id: str,
        cached: CachedOverride,
        retry_after_seconds: float,
    ) -> None:
        """Keep serving *cached* as fresh for a short back-off after a failed refresh.

        The hard expiry is preserved, so a long outage cannot extend the life of
        stale data past the hard TTL.
        """
        await self._write(
            config_path=config_path,
            client_id=client_id,
            result=cached.result,
            fetched_at=self._clock(),
            soft_ttl=retry_after_seconds,
            hard_expires_at=cached.hard_expires_at,
        )

    async def delete(self, *, config_path: str, client_id: str) -> None:
        key = self.build_key(config_path=config_path, client_id=client_id)
        await self._store.delete(key, collection=self._collection)

    async def _write(
        self,
        *,
        config_path: str,
        client_id: str,
        result: OverrideReadResult,
        fetched_at: float,
        soft_ttl: float,
        hard_expires_at: float,
    ) -> None:
        remaining = hard_expires_at - self._clock()
        if remaining <= 0:
            return
        row: dict[str, Any] = {
            "client_id": client_id,
            "status": result.outcome.value,
            "fetched_at": fetched_at,
            "soft_ttl_seconds": soft_ttl,
            "hard_expires_at": hard_expires_at,
            "has_unresolved_mcp": result.has_unresolved_mcp,
            # exclude_unset: the override merge relies on which fields were
            # explicitly set; a full dump would let defaults overwrite base values.
            "models": [m.model_dump(exclude_unset=True) for m in result.models],
        }
        await self._store.put(
            self.build_key(config_path=config_path, client_id=client_id),
            row,
            collection=self._collection,
            ttl=max(1.0, remaining),
        )
