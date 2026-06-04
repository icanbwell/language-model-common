# Slash Commands

Slash commands let users trigger actions by typing `/command-name` at the start of a message. The system parses the command, routes it to a handler, and produces an effect that the hosting application interprets.

## Architecture

```
User message: "/reload do something"
        │
        ▼
┌─────────────────────┐
│ SlashCommandParser   │  Extracts command_name="reload", remaining="do something"
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ SlashCommandProcessor│  Iterates handlers until one matches
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ SlashCommandHandler  │  Returns a SlashCommandEffect (or None)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Caller (Preprocessor)│  Interprets the effect (inject content, raise action, etc.)
└─────────────────────┘
```

**Key files:**
- `languagemodelcommon/utilities/slash_command/slash_command_parser.py` — regex parser
- `languagemodelcommon/utilities/slash_command/slash_command_processor.py` — routes to handlers
- `languagemodelcommon/utilities/slash_command/slash_command_handler.py` — protocol + base types
- `languagemodelcommon/utilities/slash_command/handlers/` — handler implementations

## Command Name Rules

Command names must match: `^[a-zA-Z][a-zA-Z0-9_-]*$`

- Starts with a letter
- Contains letters, digits, hyphens, or underscores
- Case-sensitive (prefer lowercase with hyphens)

Examples: `reload`, `reload-skills`, `my-skill`, `debug`

## Handler Types

### 1. Skill Commands (`SkillCommandHandler`)

Dynamic commands whose names come from a `SkillContentResolver`. When matched, the handler produces a `SkillCommandEffect` which the preprocessor uses to load skill content as a system message.

**Adding a new skill command:** Register the skill name in the `SkillContentResolver` implementation. No code changes needed in the handler itself.

### 2. Reload Commands (implemented in baileyai)

Static action commands that trigger side effects (cache clearing, server reloads). The `ReloadCommandHandler` lives in `baileyai/services/message_preprocessing/handlers/reload_command_handler.py` — not in this library. It produces a `ReloadCommandEffect` with a `ReloadTarget` enum.

**Current commands (defined in baileyai):**
| Command | Target | Behavior |
|---------|--------|----------|
| `/reload` | `ReloadTarget.MODELS` | Clears model config + tool caches |
| `/reload-skills` | `ReloadTarget.SKILLS` | Clears skills/tool cache |

### 3. Debug Commands (`DebugCommandHandler`)

Strips the `/debug` prefix and produces a `DebugCommandEffect` with the remaining content.

## Adding a New Slash Command

### Option A: Add a command to an existing handler

If the new command fits an existing handler category (e.g., a new reload target):

1. Add the command name → target mapping in the handler class
2. Add test coverage
3. Handle the new target in the caller's callback

**Example: Adding `/reload-prompts` (in baileyai)**

```python
# In baileyai/services/message_preprocessing/handlers/reload_command_handler.py
class ReloadTarget(Enum):
    MODELS = "models"
    SKILLS = "skills"
    PROMPTS = "prompts"  # ← add

_COMMAND_MAP: dict[str, ReloadTarget] = {
    "reload": ReloadTarget.MODELS,
    "reload-skills": ReloadTarget.SKILLS,
    "reload-prompts": ReloadTarget.PROMPTS,  # ← add
}
```

Then handle `ReloadTarget.PROMPTS` in the reload callback where the effect is consumed.

### Option B: Create a new handler

For a new category of commands with distinct behavior:

1. Create a handler class in `languagemodelcommon/utilities/slash_command/handlers/`
2. Define an effect dataclass
3. Export from `__init__.py`
4. Register the handler in the `SlashCommandPreprocessor`
5. Handle the new effect type in the preprocessor

**Template:**

```python
# languagemodelcommon/utilities/slash_command/handlers/my_command_handler.py
from __future__ import annotations

from dataclasses import dataclass

from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
)


@dataclass(frozen=True, slots=True)
class MyCommandEffect(SlashCommandEffect):
    some_data: str


class MyCommandHandler:
    @property
    def command_names(self) -> set[str]:
        return {"my-command"}

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None:
        return MyCommandEffect(some_data=context.remaining_message)
```

Then register in the preprocessor (in baileyai):

```python
# In baileyai SlashCommandPreprocessor.__init__
handlers: list[SlashCommandHandler] = [
    ReloadCommandHandler(),
    MyCommandHandler(),  # ← add
    SkillCommandHandler(resolver=skill_content_resolver),
]
```

### Option C: Add a dynamic skill command

If the command loads content (instructions, context) rather than performing an action:

1. Add the skill name to your `SkillContentResolver` implementation
2. Implement `resolve(skill_name=...)` to return the content for that name
3. No handler code changes needed — `SkillCommandHandler` handles it automatically

## Effect Types

| Effect | Source | Behavior in baileyai |
|--------|--------|---------------------|
| `SkillCommandEffect` | language-model-common | Loads skill content as a system message before the user's message |
| `ReloadCommandEffect` | baileyai | Executes reload callback, short-circuits LLM with confirmation |
| `DebugCommandEffect` | language-model-common | Strips debug prefix (handling depends on consumer) |

### Action effects vs. content effects

- **Content effects** (skills): Transform the message list and continue to the LLM
- **Action effects** (reload): Execute a side effect and return a direct response without calling the LLM

Action effects raise `SlashCommandActionComplete` from the preprocessor, which is caught by `BaileyAgentService` to short-circuit the response.

## Testing

Each handler should have unit tests covering:
- `command_names` returns the expected set
- `handle()` returns the correct effect for each command
- `handle()` returns `None` for unknown commands

Tests for the preprocessor should cover:
- The effect is produced and handled correctly
- Messages are unchanged when no command matches
- Edge cases (empty messages, no human message, resolver returns None)

Test files:
- `tests/utilities/slash_command/test_skill_command_handler.py`
- `tests/utilities/slash_command/test_debug_command_handler.py`
- `tests/utilities/slash_command/test_slash_command_parser.py`
- `tests/utilities/slash_command/test_slash_command_processor.py`
- `tests/utilities/message_preprocessing/test_composite_message_preprocessor.py`

Reload handler tests live in baileyai:
- `baileyai/tests/services/message_preprocessing/handlers/test_reload_command_handler.py`
- `baileyai/tests/services/message_preprocessing/test_slash_command_preprocessor.py`
