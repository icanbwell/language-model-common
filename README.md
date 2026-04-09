# language-model-common

A shared Python framework for building LLM-powered agent applications with LangChain and LangGraph. It provides reusable infrastructure for configuration management, protocol conversion, MCP tool integration, file management, and more.

## Features

- **Multi-source configuration loading** — Read LLM model configs from local filesystem, AWS S3, or GitHub repositories with TTL-based caching and client-specific overrides
- **LangGraph-to-OpenAI protocol conversion** — Stream LangGraph agent output as OpenAI-compatible Server-Sent Events (SSE) for chat completion APIs
- **MCP (Model Context Protocol) integration** — Tool discovery with BM25 search ranking, OAuth 2.1/OIDC support, and dynamic client registration
- **Prompt template library** — Load and manage prompt templates from organized directory structures with GitHub auto-download support
- **File management abstraction** — Unified interface for local and AWS S3 storage with factory-based backend selection
- **Token and cost management** — Token reduction for long conversations and usage metadata tracking via tiktoken
- **Authentication and authorization** — OAuth with PKCE, JWT token validation, OIDC provider discovery
- **Image generation** — Provider abstraction over OpenAI and AWS Bedrock image generators
- **OCR extraction** — AWS Textract integration behind a factory interface
- **Dependency injection container** — Pre-wired service registry using `simple-container` with singleton lifecycle management

## Installation

```bash
pip install language-model-common
```

**Requirements:** Python >= 3.10

## Quick Start

### Reading Model Configurations

```python
from languagemodelcommon.configs.config_reader import ConfigReader

config_reader = ConfigReader(
    config_paths=["./configs"],  # local path, s3:// URI, or github:// path
)

# Load base configs with optional client-specific overrides
models = await config_reader.read_model_configs_async(client_id="my-client")
```

Configuration sources are selected by path prefix:
- **Local:** `./configs` or `/absolute/path`
- **S3:** `s3://bucket-name/path`
- **GitHub:** Managed via `GithubConfigRepoManager` with automatic background refresh

### Loading Prompts

```python
from languagemodelcommon.configs.prompt_library import PromptLibraryManager

prompt_manager = PromptLibraryManager(config_paths=["./configs"])
prompt_text = prompt_manager.get_prompt("system-prompt")
```

Prompts are loaded from `.md` or `.txt` files in a `prompts/` subdirectory of your config paths.

### Streaming LangGraph Output as OpenAI SSE

```python
from languagemodelcommon.converters import LangGraphToOpenAIConverter

converter = LangGraphToOpenAIConverter(
    streaming_manager=streaming_manager,
    token_reducer=token_reducer,
)

response = await converter.stream_response(
    graph=my_langgraph,
    messages=messages,
    config=config,
)
```

### File Management

```python
from languagemodelcommon.file_managers import FileManagerFactory

factory = FileManagerFactory(aws_client_factory=aws_factory)

# Automatically selects local or S3 backend based on the folder path
manager = factory.create("s3://my-bucket")
await manager.save_file_async(file_data, "s3://my-bucket", "output.json", "application/json")
```

### MCP Tool Discovery

```python
from languagemodelcommon.mcp import ToolCatalog

catalog = ToolCatalog()
catalog.register_server("my-server", url="http://localhost:8080")

# BM25-ranked search for relevant tools
results = catalog.search("search patient records")
```

### Using the Dependency Injection Container

```python
from languagemodelcommon.container import LanguageModelCommonContainerFactory
from simple_container import SimpleContainer

container = SimpleContainer()
LanguageModelCommonContainerFactory.register_services_in_container(container)

config_reader = container.resolve(ConfigReader)
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GITHUB_CONFIG_REPO_URL` | GitHub zipball API URL for config repo | — |
| `GITHUB_TOKEN` | GitHub token for authenticated requests | — |
| `GITHUB_CACHE_FOLDER` | Local cache directory for GitHub configs | `/tmp/github_config_cache` |
| `CONFIG_CACHE_TIMEOUT_SECONDS` | Config refresh interval (seconds) | `120` |
| `GITHUB_TIMEOUT` | HTTP timeout for GitHub requests (seconds) | `300` |

## Development

### Prerequisites

- Docker and Docker Compose

### Setup

```bash
make init
```

This builds the development Docker image, locks dependencies, and sets up pre-commit hooks.

### Running Tests

```bash
make tests
```

Tests run inside Docker using pytest with async support (`asyncio_mode = auto`).

### Other Commands

```bash
make shell       # Open a shell in the dev container
make update      # Update dependencies and rebuild
make build       # Build distribution package
make package     # Publish to PyPI
make testpackage # Publish to TestPyPI
```

## Project Structure

```
languagemodelcommon/
├── auth/              # Token storage and auth managers
├── aws/               # AWS client factory
├── configs/           # Config reader, prompt library, schemas
├── container/         # DI container factory
├── converters/        # LangGraph-to-OpenAI conversion, streaming
├── exceptions/        # Custom exception types
├── file_managers/     # Local and S3 file management
├── graph/             # LangGraph utilities
├── history/           # Conversation history management
├── http/              # HTTP client factory
├── image_generation/  # Image generation providers
├── markdown/          # HTML/CSV to Markdown converters
├── mcp/               # MCP client, tool catalog, OAuth
├── models/            # LLM model definitions
├── mocks/             # Test mocks and fakes
├── ocr/               # OCR extraction (AWS Textract)
├── persistence/       # LangGraph checkpoint/store backends
├── schema/            # OpenAI-compatible schema definitions
├── state/             # LangGraph state definitions
├── structures/        # Request/response wrappers
├── tools/             # Resilient tool base class, MCP tools
└── utilities/         # Logging, caching, token reduction, security
```

## License

Apache License 2.0
