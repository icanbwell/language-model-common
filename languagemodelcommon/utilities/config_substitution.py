import os
import re

from typing import Any

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def substitute_env_vars(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: substitute_env_vars(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [substitute_env_vars(item) for item in payload]
    if isinstance(payload, str):
        return _substitute_string(payload)
    return payload


def _substitute_string(value: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        env_key = match.group(1)
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
        raise ValueError(f"Missing environment variable: {env_key}")

    if "${" not in value:
        return value
    return _ENV_VAR_PATTERN.sub(_replace, value)
