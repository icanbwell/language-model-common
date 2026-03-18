from typing import runtime_checkable, Protocol


@runtime_checkable
class PromptLibraryEnvironmentVariables(Protocol):
    @property
    def prompt_library_path(self) -> str | None: ...
