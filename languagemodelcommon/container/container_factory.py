from typing import Any

from simple_container.container.simple_container import SimpleContainer

from languagemodelcommon.aws.aws_client_factory import AwsClientFactory
from languagemodelcommon.configs.config_reader.config_reader import ConfigReader
from languagemodelcommon.configs.config_reader.github_directory_helper import (
    GitHubDirectoryHelper,
)
from languagemodelcommon.configs.config_reader.mcp_json_fetcher import McpJsonFetcher
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)
from languagemodelcommon.converters.langgraph_to_openai_converter import (
    LangGraphToOpenAIConverter,
)
from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)
from languagemodelcommon.converters.streaming_manager import LangGraphStreamingManager
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.file_managers.file_writer import FileWriter
from languagemodelcommon.file_managers.file_manager_factory import FileManagerFactory
from languagemodelcommon.image_generation.image_generator_factory import (
    ImageGeneratorFactory,
)
from languagemodelcommon.image_generation.managers.image_generation_manager import (
    ImageGenerationManager,
)
from languagemodelcommon.image_generation.providers.image_generation_provider import (
    ImageGenerationProvider,
)
from languagemodelcommon.ocr.ocr_extractor_factory import OCRExtractorFactory
from languagemodelcommon.persistence.persistence_factory import PersistenceFactory
from key_value.aio.stores.base import BaseStore as KeyValueBaseStore

from languagemodelcommon.utilities.cache.config_expiring_cache import (
    ConfigExpiringCache,
)
from languagemodelcommon.mcp.mcp_client.tool_list_store_mongo import MongoToolListStore
from languagemodelcommon.utilities.cache.snapshot_cache_store import (
    create_cache_store,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.token_reducer.token_reducer import TokenReducer
from languagemodelcommon.utilities.tool_display_name_mapper import (
    ToolDisplayNameMapper,
)


class LanguageModelCommonContainerFactory:
    @staticmethod
    def register_services_in_container(
        *, container: SimpleContainer
    ) -> SimpleContainer:

        container.singleton(
            service_type=LanguageModelCommonEnvironmentVariables,
            factory=lambda c: LanguageModelCommonEnvironmentVariables(),
        )
        container.singleton(
            service_type=GitHubDirectoryHelper,
            factory=lambda c: GitHubDirectoryHelper(
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
            ),
        )
        # we want only one instance of the cache so we use singleton
        container.singleton(
            service_type=ConfigExpiringCache,
            factory=lambda c: ConfigExpiringCache(
                ttl_seconds=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).config_cache_timeout_seconds,
            ),
        )
        container.singleton(
            service_type=PromptLibraryManager,
            factory=lambda c: PromptLibraryManager(
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
                github_directory_helper=c.resolve(GitHubDirectoryHelper),
            ),
        )

        def _create_mcp_json_fetcher(c: Any) -> McpJsonFetcher | None:
            url = c.resolve(LanguageModelCommonEnvironmentVariables).plugins_mcp_server
            return McpJsonFetcher(plugins_mcp_server_url=url) if url else None

        container.singleton(
            service_type=McpJsonFetcher, factory=_create_mcp_json_fetcher
        )
        container.singleton(
            service_type=KeyValueBaseStore,
            factory=lambda c: create_cache_store(
                cache_type=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).snapshot_cache_type,
                mongo_url=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).mongo_llm_storage_uri,
                mongo_db_name=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).mongo_llm_storage_db_name
                or "language_model_gateway",
                mongo_username=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).mongo_llm_storage_db_username,
                mongo_password=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).mongo_llm_storage_db_password,
                collection=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).snapshot_cache_collection_name,
            ),
        )
        container.singleton(
            service_type=MongoToolListStore,
            factory=lambda c: MongoToolListStore(
                store=create_cache_store(
                    cache_type=c.resolve(
                        LanguageModelCommonEnvironmentVariables
                    ).snapshot_cache_type,
                    mongo_url=c.resolve(
                        LanguageModelCommonEnvironmentVariables
                    ).mongo_llm_storage_uri,
                    mongo_db_name=c.resolve(
                        LanguageModelCommonEnvironmentVariables
                    ).mongo_llm_storage_db_name
                    or "language_model_gateway",
                    mongo_username=c.resolve(
                        LanguageModelCommonEnvironmentVariables
                    ).mongo_llm_storage_db_username,
                    mongo_password=c.resolve(
                        LanguageModelCommonEnvironmentVariables
                    ).mongo_llm_storage_db_password,
                    collection="mcp_tool_cache",
                ),
            ),
        )
        container.singleton(
            service_type=ConfigReader,
            factory=lambda c: ConfigReader(
                cache=c.resolve(ConfigExpiringCache),
                prompt_library_manager=c.resolve(PromptLibraryManager),
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
                mcp_json_fetcher=c.resolve(McpJsonFetcher),
                github_directory_helper=c.resolve(GitHubDirectoryHelper),
                snapshot_cache_store=c.resolve(KeyValueBaseStore),
            ),
        )
        container.singleton(
            service_type=TokenReducer,
            factory=lambda c: TokenReducer(
                model=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).default_llm_model,
            ),
        )

        # --- Request-scoped services (one instance per HTTP request) ---

        container.request_scoped(
            service_type=StreamBufferManager,
            factory=lambda c: StreamBufferManager(
                flush_interval_seconds=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).streaming_buffer_flush_interval_seconds,
                enabled=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ).enable_streaming_buffering,
            ),
        )

        container.request_scoped(
            service_type=StreamDebugOutputManager,
            factory=lambda c: StreamDebugOutputManager(),
        )

        container.request_scoped(
            service_type=ToolEventHandler,
            factory=lambda c: ToolEventHandler(
                debug_file_writer=c.resolve(FileWriter),
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
                tool_display_name_mapper=c.resolve(ToolDisplayNameMapper),
                stream_buffer_manager=c.resolve(StreamBufferManager),
                stream_debug_output_manager=c.resolve(StreamDebugOutputManager),
            ),
        )

        container.request_scoped(
            service_type=LangGraphStreamingManager,
            factory=lambda c: LangGraphStreamingManager(
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
                debug_file_writer=c.resolve(FileWriter),
                token_reducer=c.resolve(TokenReducer),
                tool_event_handler=c.resolve(ToolEventHandler),
                stream_buffer_manager=c.resolve(StreamBufferManager),
                stream_debug_output_manager=c.resolve(StreamDebugOutputManager),
            ),
        )

        container.request_scoped(
            service_type=LangGraphToOpenAIConverter,
            factory=lambda c: LangGraphToOpenAIConverter(
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
                token_reducer=c.resolve(TokenReducer),
                streaming_manager=c.resolve(LangGraphStreamingManager),
                stream_debug_output_manager=c.resolve(StreamDebugOutputManager),
            ),
        )

        container.singleton(
            service_type=FileWriter,
            factory=lambda c: FileWriter(
                file_manager_factory=c.resolve(FileManagerFactory),
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
            ),
        )
        container.singleton(
            service_type=FileManagerFactory,
            factory=lambda c: FileManagerFactory(
                aws_client_factory=c.resolve(AwsClientFactory),
            ),
        )

        container.singleton(
            service_type=AwsClientFactory,
            factory=lambda c: AwsClientFactory(
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
            ),
        )

        container.singleton(
            service_type=ImageGeneratorFactory,
            factory=lambda c: ImageGeneratorFactory(
                aws_client_factory=c.resolve(AwsClientFactory),
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
            ),
        )

        container.singleton(
            service_type=ImageGenerationProvider,
            factory=lambda c: ImageGenerationProvider(
                image_generator_factory=c.resolve(ImageGeneratorFactory),
                file_manager_factory=c.resolve(FileManagerFactory),
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
            ),
        )
        container.singleton(
            service_type=ImageGenerationManager,
            factory=lambda c: ImageGenerationManager(
                image_generation_provider=c.resolve(ImageGenerationProvider)
            ),
        )

        container.singleton(
            service_type=PersistenceFactory,
            factory=lambda c: PersistenceFactory(
                environment_variables=c.resolve(LanguageModelCommonEnvironmentVariables)
            ),
        )

        container.singleton(
            service_type=OCRExtractorFactory,
            factory=lambda c: OCRExtractorFactory(
                aws_client_factory=c.resolve(AwsClientFactory),
                file_manager_factory=c.resolve(FileManagerFactory),
            ),
        )

        return container
