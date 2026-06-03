# Slash Command Interceptor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic slash-command interceptor to `language-model-common` that detects `/command-name` prefixes in user messages, enabling skill invocation and debug toggling through a unified mechanism. Migrate existing debug prefix handling to use this new system.

**Architecture:** A `SlashCommandParser` extracts `/command` prefixes from user messages. A `SlashCommandProcessor` holds registered command handlers (each implementing a `SlashCommandHandler` protocol) and applies them during request wrapper construction. The existing `_apply_debug_prefix_toggle` logic in both request wrappers is replaced by a `DebugCommandHandler`. A `SkillContentResolver` protocol enables consuming apps (like baileyai) to wire skill loading without `language-model-common` depending on the skills framework.

**Tech Stack:** Python 3.12+, Pydantic, pytest, language-model-common, baileyai

---

### Task 1: Slash Command Parser

**Files:**
- Create: `languagemodelcommon/utilities/slash_command/slash_command_parser.py`
- Test: `tests/utilities/slash_command/test_slash_command_parser.py`

- [ ] **Step 1: Create directory and __init__.py**

```bash
mkdir -p languagemodelcommon/utilities/slash_command
touch languagemodelcommon/utilities/slash_command/__init__.py
mkdir -p tests/utilities/slash_command
touch tests/utilities/slash_command/__init__.py
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/utilities/slash_command/test_slash_command_parser.py
import pytest

from languagemodelcommon.utilities.slash_command.slash_command_parser import (
    SlashCommandMatch,
    parse_slash_command,
)


class TestParseSlashCommand:
    def test_simple_command_with_message(self) -> None:
        result = parse_slash_command(content="/debug hello world")
        assert result is not None
        assert result.command_name == "debug"
        assert result.remaining_message == "hello world"

    def test_command_with_hyphenated_name(self) -> None:
        result = parse_slash_command(content="/uspstf-depression-screening what should I check?")
        assert result is not None
        assert result.command_name == "uspstf-depression-screening"
        assert result.remaining_message == "what should I check?"

    def test_command_only_no_message(self) -> None:
        result = parse_slash_command(content="/list-skills")
        assert result is not None
        assert result.command_name == "list-skills"
        assert result.remaining_message == ""

    def test_no_slash_prefix_returns_none(self) -> None:
        result = parse_slash_command(content="hello world")
        assert result is None

    def test_slash_in_middle_of_message_returns_none(self) -> None:
        result = parse_slash_command(content="use /debug mode")
        assert result is None

    def test_url_in_message_returns_none(self) -> None:
        result = parse_slash_command(content="check https://example.com/path")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = parse_slash_command(content="")
        assert result is None

    def test_just_slash_returns_none(self) -> None:
        result = parse_slash_command(content="/")
        assert result is None

    def test_slash_with_spaces_before_command_returns_none(self) -> None:
        result = parse_slash_command(content="  /debug hello")
        assert result is None

    def test_multiple_spaces_between_command_and_message(self) -> None:
        result = parse_slash_command(content="/debug   hello")
        assert result is not None
        assert result.command_name == "debug"
        assert result.remaining_message == "hello"

    def test_colon_style_prefix(self) -> None:
        result = parse_slash_command(content="DEBUG: hello")
        assert result is None

    def test_command_with_underscores(self) -> None:
        result = parse_slash_command(content="/vaccine_eligibility check patient")
        assert result is not None
        assert result.command_name == "vaccine_eligibility"
        assert result.remaining_message == "check patient"

    def test_command_with_numeric_suffix(self) -> None:
        result = parse_slash_command(content="/uspstf-6 check depression")
        assert result is not None
        assert result.command_name == "uspstf-6"
        assert result.remaining_message == "check depression"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/utilities/slash_command/test_slash_command_parser.py -v`
Expected: FAIL with ImportError

- [ ] **Step 4: Implement the parser**

```python
# languagemodelcommon/utilities/slash_command/slash_command_parser.py
from __future__ import annotations

import re
from dataclasses import dataclass


_SLASH_COMMAND_RE = re.compile(r"^/([a-zA-Z][a-zA-Z0-9_-]*)(?:\s+(.*))?$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SlashCommandMatch:
    """Result of detecting a /command prefix in user input."""

    command_name: str
    remaining_message: str


def parse_slash_command(*, content: str) -> SlashCommandMatch | None:
    """Detect a /command-name prefix at the start of a message.

    Returns a SlashCommandMatch if the message starts with a valid
    slash command, otherwise None.
    """
    if not content or not content.startswith("/"):
        return None

    match = _SLASH_COMMAND_RE.match(content)
    if not match:
        return None

    command_name = match.group(1)
    remaining = (match.group(2) or "").strip()

    return SlashCommandMatch(
        command_name=command_name,
        remaining_message=remaining,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/utilities/slash_command/test_slash_command_parser.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add languagemodelcommon/utilities/slash_command/ tests/utilities/slash_command/
git commit -m "BAI-190 add slash command parser with tests"
```

---

### Task 2: Slash Command Handler Protocol and Processor

**Files:**
- Create: `languagemodelcommon/utilities/slash_command/slash_command_handler.py`
- Create: `languagemodelcommon/utilities/slash_command/slash_command_processor.py`
- Test: `tests/utilities/slash_command/test_slash_command_processor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/utilities/slash_command/test_slash_command_processor.py
from __future__ import annotations

from dataclasses import dataclass

import pytest

from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandHandler,
    SlashCommandContext,
    SlashCommandEffect,
)
from languagemodelcommon.utilities.slash_command.slash_command_processor import (
    SlashCommandProcessor,
)


@dataclass
class _FakeEffect(SlashCommandEffect):
    handled_command: str = ""
    handled_remaining: str = ""


class _FakeHandler(SlashCommandHandler):
    def __init__(self, *, commands: set[str]) -> None:
        self._commands = commands

    @property
    def command_names(self) -> set[str]:
        return self._commands

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None:
        return _FakeEffect(
            handled_command=context.command_name,
            handled_remaining=context.remaining_message,
        )


class TestSlashCommandProcessor:
    def test_processes_matching_command(self) -> None:
        handler = _FakeHandler(commands={"debug"})
        processor = SlashCommandProcessor(handlers=[handler])

        effect = processor.process(content="/debug hello")

        assert effect is not None
        assert isinstance(effect, _FakeEffect)
        assert effect.handled_command == "debug"
        assert effect.handled_remaining == "hello"

    def test_returns_none_for_unmatched_command(self) -> None:
        handler = _FakeHandler(commands={"debug"})
        processor = SlashCommandProcessor(handlers=[handler])

        effect = processor.process(content="/unknown hello")

        assert effect is None

    def test_returns_none_for_no_slash(self) -> None:
        handler = _FakeHandler(commands={"debug"})
        processor = SlashCommandProcessor(handlers=[handler])

        effect = processor.process(content="hello world")

        assert effect is None

    def test_first_matching_handler_wins(self) -> None:
        handler1 = _FakeHandler(commands={"debug"})
        handler2 = _FakeHandler(commands={"debug"})
        processor = SlashCommandProcessor(handlers=[handler1, handler2])

        effect = processor.process(content="/debug hello")

        assert effect is not None

    def test_multiple_handlers_different_commands(self) -> None:
        handler1 = _FakeHandler(commands={"debug"})
        handler2 = _FakeHandler(commands={"skill"})
        processor = SlashCommandProcessor(handlers=[handler1, handler2])

        effect1 = processor.process(content="/debug hello")
        effect2 = processor.process(content="/skill load")

        assert effect1 is not None
        assert effect2 is not None
        assert isinstance(effect1, _FakeEffect)
        assert effect1.handled_command == "debug"
        assert isinstance(effect2, _FakeEffect)
        assert effect2.handled_command == "skill"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/utilities/slash_command/test_slash_command_processor.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement the handler protocol**

```python
# languagemodelcommon/utilities/slash_command/slash_command_handler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SlashCommandContext:
    """Context passed to a command handler."""

    command_name: str
    remaining_message: str
    original_content: str


class SlashCommandEffect:
    """Base class for effects produced by command handlers.

    Subclasses carry the data needed to apply the effect (e.g.,
    enable debug logging, inject a system prompt, etc.).
    """


@runtime_checkable
class SlashCommandHandler(Protocol):
    @property
    def command_names(self) -> set[str]:
        """Return the set of command names this handler responds to."""
        ...

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None:
        """Process the command and return an effect, or None to skip."""
        ...
```

- [ ] **Step 4: Implement the processor**

```python
# languagemodelcommon/utilities/slash_command/slash_command_processor.py
from __future__ import annotations

from typing import Sequence

from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
    SlashCommandHandler,
)
from languagemodelcommon.utilities.slash_command.slash_command_parser import (
    parse_slash_command,
)


class SlashCommandProcessor:
    """Dispatches slash commands to registered handlers."""

    def __init__(self, *, handlers: Sequence[SlashCommandHandler]) -> None:
        self._handlers = list(handlers)

    def process(self, *, content: str) -> SlashCommandEffect | None:
        """Parse the content for a slash command and dispatch to the first matching handler."""
        match = parse_slash_command(content=content)
        if match is None:
            return None

        context = SlashCommandContext(
            command_name=match.command_name,
            remaining_message=match.remaining_message,
            original_content=content,
        )

        for handler in self._handlers:
            if match.command_name in handler.command_names:
                effect = handler.handle(context=context)
                if effect is not None:
                    return effect

        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/utilities/slash_command/test_slash_command_processor.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add languagemodelcommon/utilities/slash_command/slash_command_handler.py languagemodelcommon/utilities/slash_command/slash_command_processor.py tests/utilities/slash_command/test_slash_command_processor.py
git commit -m "BAI-190 add slash command handler protocol and processor"
```

---

### Task 3: Debug Command Handler

**Files:**
- Create: `languagemodelcommon/utilities/slash_command/handlers/debug_command_handler.py`
- Create: `languagemodelcommon/utilities/slash_command/handlers/__init__.py`
- Test: `tests/utilities/slash_command/test_debug_command_handler.py`

- [ ] **Step 1: Create handlers directory**

```bash
mkdir -p languagemodelcommon/utilities/slash_command/handlers
touch languagemodelcommon/utilities/slash_command/handlers/__init__.py
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/utilities/slash_command/test_debug_command_handler.py
from __future__ import annotations

import pytest

from languagemodelcommon.utilities.slash_command.handlers.debug_command_handler import (
    DebugCommandHandler,
    DebugCommandEffect,
)
from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
)


class TestDebugCommandHandler:
    def test_handles_debug_command(self) -> None:
        handler = DebugCommandHandler()
        context = SlashCommandContext(
            command_name="debug",
            remaining_message="hello",
            original_content="/debug hello",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, DebugCommandEffect)
        assert effect.stripped_content == "hello"

    def test_command_names_includes_debug(self) -> None:
        handler = DebugCommandHandler()
        assert "debug" in handler.command_names

    def test_handles_custom_prefixes(self) -> None:
        handler = DebugCommandHandler(command_names={"debug", "verbose"})
        context = SlashCommandContext(
            command_name="verbose",
            remaining_message="test",
            original_content="/verbose test",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, DebugCommandEffect)
        assert effect.stripped_content == "test"

    def test_empty_remaining_message(self) -> None:
        handler = DebugCommandHandler()
        context = SlashCommandContext(
            command_name="debug",
            remaining_message="",
            original_content="/debug",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, DebugCommandEffect)
        assert effect.stripped_content == ""
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/utilities/slash_command/test_debug_command_handler.py -v`
Expected: FAIL with ImportError

- [ ] **Step 4: Implement the debug handler**

```python
# languagemodelcommon/utilities/slash_command/handlers/debug_command_handler.py
from __future__ import annotations

from dataclasses import dataclass

from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
)


@dataclass(frozen=True, slots=True)
class DebugCommandEffect(SlashCommandEffect):
    """Signals that debug logging should be enabled and the message content stripped."""

    stripped_content: str


class DebugCommandHandler:
    """Handles /debug commands by enabling debug logging."""

    def __init__(self, *, command_names: set[str] | None = None) -> None:
        self._command_names = command_names or {"debug"}

    @property
    def command_names(self) -> set[str]:
        return self._command_names

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None:
        return DebugCommandEffect(stripped_content=context.remaining_message)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/utilities/slash_command/test_debug_command_handler.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add languagemodelcommon/utilities/slash_command/handlers/ tests/utilities/slash_command/test_debug_command_handler.py
git commit -m "BAI-190 add debug command handler"
```

---

### Task 4: Skill Command Handler with Resolver Protocol

**Files:**
- Create: `languagemodelcommon/utilities/slash_command/handlers/skill_command_handler.py`
- Test: `tests/utilities/slash_command/test_skill_command_handler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/utilities/slash_command/test_skill_command_handler.py
from __future__ import annotations

import pytest

from languagemodelcommon.utilities.slash_command.handlers.skill_command_handler import (
    SkillCommandHandler,
    SkillCommandEffect,
    SkillContentResolver,
)
from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
)


class _FakeResolver(SkillContentResolver):
    def __init__(self, *, skills: dict[str, str]) -> None:
        self._skills = skills

    def get_available_skill_names(self) -> set[str]:
        return set(self._skills.keys())

    async def resolve(self, *, skill_name: str) -> str | None:
        return self._skills.get(skill_name)


class TestSkillCommandHandler:
    def test_matches_registered_skill_name(self) -> None:
        resolver = _FakeResolver(skills={"uspstf-depression-screening": "Skill content here"})
        handler = SkillCommandHandler(resolver=resolver)

        assert "uspstf-depression-screening" in handler.command_names

    def test_produces_skill_effect_for_known_skill(self) -> None:
        resolver = _FakeResolver(skills={"uspstf-depression-screening": "Skill content here"})
        handler = SkillCommandHandler(resolver=resolver)
        context = SlashCommandContext(
            command_name="uspstf-depression-screening",
            remaining_message="check patient age 45",
            original_content="/uspstf-depression-screening check patient age 45",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, SkillCommandEffect)
        assert effect.skill_name == "uspstf-depression-screening"
        assert effect.remaining_message == "check patient age 45"

    def test_returns_none_for_unknown_skill(self) -> None:
        resolver = _FakeResolver(skills={"depression": "content"})
        handler = SkillCommandHandler(resolver=resolver)
        context = SlashCommandContext(
            command_name="unknown-skill",
            remaining_message="hello",
            original_content="/unknown-skill hello",
        )

        effect = handler.handle(context=context)

        assert effect is None

    def test_empty_remaining_message(self) -> None:
        resolver = _FakeResolver(skills={"list-skills": "list all skills"})
        handler = SkillCommandHandler(resolver=resolver)
        context = SlashCommandContext(
            command_name="list-skills",
            remaining_message="",
            original_content="/list-skills",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, SkillCommandEffect)
        assert effect.remaining_message == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/utilities/slash_command/test_skill_command_handler.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement the skill handler and resolver protocol**

```python
# languagemodelcommon/utilities/slash_command/handlers/skill_command_handler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
)


@runtime_checkable
class SkillContentResolver(Protocol):
    """Protocol for resolving skill names to their content.

    Implemented by consuming applications (e.g., baileyai) to connect
    to whatever skill source they use (MCP marketplace, local files, etc.).
    """

    def get_available_skill_names(self) -> set[str]:
        """Return the set of skill names that can be invoked via /command."""
        ...

    async def resolve(self, *, skill_name: str) -> str | None:
        """Resolve a skill name to its full content, or None if not found."""
        ...


@dataclass(frozen=True, slots=True)
class SkillCommandEffect(SlashCommandEffect):
    """Signals that a skill should be loaded and injected into context."""

    skill_name: str
    remaining_message: str


class SkillCommandHandler:
    """Routes /skill-name commands to the skill resolver."""

    def __init__(self, *, resolver: SkillContentResolver) -> None:
        self._resolver = resolver

    @property
    def command_names(self) -> set[str]:
        return self._resolver.get_available_skill_names()

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None:
        available = self._resolver.get_available_skill_names()
        if context.command_name not in available:
            return None

        return SkillCommandEffect(
            skill_name=context.command_name,
            remaining_message=context.remaining_message,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/utilities/slash_command/test_skill_command_handler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add languagemodelcommon/utilities/slash_command/handlers/skill_command_handler.py tests/utilities/slash_command/test_skill_command_handler.py
git commit -m "BAI-190 add skill command handler with resolver protocol"
```

---

### Task 5: Migrate Debug Prefix Handling to SlashCommandProcessor

**Files:**
- Modify: `languagemodelcommon/structures/openai/request/chat_completion_api_request_wrapper.py:87-110`
- Modify: `languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py:79-100`
- Modify: `tests/structures/openai/request/test_responses_api_request_wrapper.py` (in language-model-common)
- Verify: `tests/structures/test_debug_prefix.py` (in baileyai — ensure no behavior change)

- [ ] **Step 1: Update the `__init__.py` module exports**

```python
# languagemodelcommon/utilities/slash_command/__init__.py
from languagemodelcommon.utilities.slash_command.slash_command_parser import (
    SlashCommandMatch,
    parse_slash_command,
)
from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
    SlashCommandHandler,
)
from languagemodelcommon.utilities.slash_command.slash_command_processor import (
    SlashCommandProcessor,
)
from languagemodelcommon.utilities.slash_command.handlers.debug_command_handler import (
    DebugCommandEffect,
    DebugCommandHandler,
)
from languagemodelcommon.utilities.slash_command.handlers.skill_command_handler import (
    SkillCommandEffect,
    SkillCommandHandler,
    SkillContentResolver,
)
```

- [ ] **Step 2: Refactor `ChatCompletionApiRequestWrapper._apply_debug_prefix_toggle`**

Replace the existing `_apply_debug_prefix_toggle` method (lines 90-110) with logic that uses the new `SlashCommandProcessor`. The method must still support the old `DEBUG:` colon-style prefix for backwards compatibility (the colon-style prefix is NOT a slash command — keep it as a simple startswith check alongside the new processor).

The refactored `__init__` section in `chat_completion_api_request_wrapper.py`:

```python
        self._debug_prefixes = environment_variables.debug_prefixes
        self._apply_debug_prefix_toggle()

    def _apply_debug_prefix_toggle(self) -> None:
        from languagemodelcommon.utilities.slash_command import (
            SlashCommandProcessor,
            DebugCommandHandler,
            DebugCommandEffect,
        )
        from languagemodelcommon.utilities.slash_command.slash_command_parser import (
            parse_slash_command,
        )

        debug_handler = DebugCommandHandler()
        processor = SlashCommandProcessor(handlers=[debug_handler])

        colon_prefixes = tuple(p for p in self._debug_prefixes if not p.startswith("/"))

        for message in self._messages:
            if message.role != "user":
                continue
            content = message.content
            if not isinstance(content, str):
                continue

            # Try slash-command style first (/debug ...)
            effect = processor.process(content=content)
            if isinstance(effect, DebugCommandEffect):
                self._enable_debug_logging = True
                stripped_content = effect.stripped_content
            else:
                # Fall back to colon-style prefixes (DEBUG:...)
                matched_prefix = next(
                    (p for p in colon_prefixes if content.startswith(p)), None
                )
                if matched_prefix is None:
                    continue
                self._enable_debug_logging = True
                stripped_content = content[len(matched_prefix):].lstrip()

            if isinstance(message, ChatCompletionApiMessageWrapper):
                if isinstance(message.message, dict):
                    message.message["content"] = stripped_content
                elif hasattr(message.message, "content"):
                    setattr(message.message, "content", stripped_content)
            break
```

- [ ] **Step 3: Refactor `ResponsesApiRequestWrapper._apply_debug_prefix_toggle` similarly**

Same pattern as above, adapted for the Responses API wrapper's message structure.

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `uv run pytest tests/structures/openai/request/test_responses_api_request_wrapper.py -v`
Run (from baileyai): `uv run pytest tests/structures/test_debug_prefix.py -v`
Expected: All PASS — behavior is identical

- [ ] **Step 5: Commit**

```bash
git add languagemodelcommon/utilities/slash_command/__init__.py languagemodelcommon/structures/openai/request/
git commit -m "BAI-190 migrate debug prefix handling to slash command processor"
```

---

### Task 6: Wire Skill Slash Commands in BaileyAI

**Files:**
- Create: `baileyai/services/skill_content_resolver_adapter.py`
- Modify: `baileyai/services/bailey_agent_services.py` (add slash command processing before graph execution)
- Modify: `baileyai/container/container_factory.py` (register new service)
- Test: `tests/services/test_skill_content_resolver_adapter.py`

- [ ] **Step 1: Write tests for the adapter**

```python
# tests/services/test_skill_content_resolver_adapter.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from baileyai.services.skill_content_resolver_adapter import (
    SkillContentResolverAdapter,
)


class TestSkillContentResolverAdapter:
    def test_get_available_skill_names_returns_configured_names(self) -> None:
        adapter = SkillContentResolverAdapter(
            skill_names={"uspstf-depression-screening", "vaccine-eligibility"},
            load_skill_fn=AsyncMock(),
        )
        assert adapter.get_available_skill_names() == {
            "uspstf-depression-screening",
            "vaccine-eligibility",
        }

    @pytest.mark.asyncio
    async def test_resolve_calls_load_skill_fn(self) -> None:
        load_fn = AsyncMock(return_value="Skill content")
        adapter = SkillContentResolverAdapter(
            skill_names={"my-skill"},
            load_skill_fn=load_fn,
        )

        result = await adapter.resolve(skill_name="my-skill")

        assert result == "Skill content"
        load_fn.assert_awaited_once_with(skill_name="my-skill")

    @pytest.mark.asyncio
    async def test_resolve_returns_none_when_fn_returns_none(self) -> None:
        load_fn = AsyncMock(return_value=None)
        adapter = SkillContentResolverAdapter(
            skill_names={"my-skill"},
            load_skill_fn=load_fn,
        )

        result = await adapter.resolve(skill_name="my-skill")

        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from baileyai): `uv run pytest tests/services/test_skill_content_resolver_adapter.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement the adapter**

```python
# baileyai/services/skill_content_resolver_adapter.py
from __future__ import annotations

from typing import Awaitable, Callable

from languagemodelcommon.utilities.slash_command.handlers.skill_command_handler import (
    SkillContentResolver,
)


class SkillContentResolverAdapter:
    """Adapts a callable skill-loading function to the SkillContentResolver protocol.

    Bridges baileyai's MCP-based skill loading with language-model-common's
    slash command system.
    """

    def __init__(
        self,
        *,
        skill_names: set[str],
        load_skill_fn: Callable[..., Awaitable[str | None]],
    ) -> None:
        self._skill_names = skill_names
        self._load_skill_fn = load_skill_fn

    def get_available_skill_names(self) -> set[str]:
        return self._skill_names

    async def resolve(self, *, skill_name: str) -> str | None:
        return await self._load_skill_fn(skill_name=skill_name)
```

- [ ] **Step 4: Wire slash command processing in BaileyAgentService**

In `bailey_agent_services.py`, add a method that processes the last user message through the `SlashCommandProcessor` before graph execution. When a `SkillCommandEffect` is detected, resolve the skill content asynchronously and inject it as a system message. When a `DebugCommandEffect` is detected, enable debug logging and strip the prefix.

This should be called from the `stream_chat_completion` and `process_chat_completion` paths, before the LangGraph agent runs.

- [ ] **Step 5: Register the adapter in the container**

In `container_factory.py`, register `SkillContentResolverAdapter` with the skill names derived from the model config's `skills` field or from the plugin marketplace's available skills list.

- [ ] **Step 6: Run all tests**

Run (from baileyai): `uv run pytest tests/services/test_skill_content_resolver_adapter.py tests/structures/test_debug_prefix.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add baileyai/services/skill_content_resolver_adapter.py baileyai/services/bailey_agent_services.py baileyai/container/container_factory.py tests/services/test_skill_content_resolver_adapter.py
git commit -m "BAI-190 wire skill slash commands in baileyai"
```

---

### Task 7: Update Package Exports and Verify Full Integration

**Files:**
- Verify: All existing tests pass across both repos
- Clean up: Ensure `__init__.py` exports are correct

- [ ] **Step 1: Run full test suite in language-model-common**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 2: Run full test suite in baileyai**

Run (from baileyai): `uv run pytest tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git commit -m "BAI-190 finalize slash command interceptor integration"
```
