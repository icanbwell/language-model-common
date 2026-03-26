import os

from langchain_ai_skills_framework.container.container_factory import (
    LangchainAISkillsFrameworkContainerFactory,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)

from simple_container.container.simple_container import SimpleContainer

from languagemodelcommon.aws.aws_client_factory import AwsClientFactory
from languagemodelcommon.configs.config_reader.config_reader import ConfigReader
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)
from languagemodelcommon.converters.langgraph_to_openai_converter import (
    LangGraphToOpenAIConverter,
)
from languagemodelcommon.file_managers.file_writer import FileWriter
from languagemodelcommon.converters.streaming_manager import LangGraphStreamingManager
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
from languagemodelcommon.utilities.cache import ConfigExpiringCache
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.token_reducer.token_reducer import TokenReducer
from languagemodelcommon.utilities.tool_friendly_name_mapper import (
    ToolFriendlyNameMapper,
)


class LanguageModelCommonContainerFactory:
    @staticmethod
    def register_services_in_container(
        *, container: SimpleContainer
    ) -> SimpleContainer:

        LangchainAISkillsFrameworkContainerFactory.register_services_in_container(
            container=container,
        )

        container.singleton(
            LanguageModelCommonEnvironmentVariables,
            lambda c: LanguageModelCommonEnvironmentVariables(),
        )
        # we want only one instance of the cache so we use singleton
        container.singleton(
            ConfigExpiringCache,
            lambda c: ConfigExpiringCache(
                ttl_seconds=(
                    int(os.environ["CONFIG_CACHE_TIMEOUT_SECONDS"])
                    if os.environ.get("CONFIG_CACHE_TIMEOUT_SECONDS")
                    else 60 * 60
                )
            ),
        )
        container.singleton(
            PromptLibraryManager,
            lambda c: PromptLibraryManager(
                environment_variables=c.resolve(LanguageModelCommonEnvironmentVariables)
            ),
        )
        container.singleton(
            ConfigReader,
            lambda c: ConfigReader(
                cache=c.resolve(ConfigExpiringCache),
                prompt_library_manager=c.resolve(PromptLibraryManager),
            ),
        )
        container.singleton(
            LangGraphToOpenAIConverter,
            lambda c: LangGraphToOpenAIConverter(
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
                token_reducer=c.resolve(TokenReducer),
                streaming_manager=c.resolve(LangGraphStreamingManager),
                skill_loader=c.resolve(SkillLoaderProtocol),
            ),
        )
        container.singleton(
            TokenReducer,
            lambda c: TokenReducer(
                model=os.environ.get("DEFAULT_LLM_MODEL", "gpt-3.5-turbo"),
            ),
        )

        container.singleton(
            LangGraphStreamingManager,
            lambda c: LangGraphStreamingManager(
                environment_variables=c.resolve(
                    LanguageModelCommonEnvironmentVariables
                ),
                debug_file_writer=c.resolve(FileWriter),
                token_reducer=c.resolve(TokenReducer),
                tool_friendly_name_mapper=c.resolve(ToolFriendlyNameMapper),
            ),
        )
        container.singleton(
            FileWriter,
            lambda c: FileWriter(
                file_manager_factory=c.resolve(FileManagerFactory),
            ),
        )
        container.singleton(
            FileManagerFactory,
            lambda c: FileManagerFactory(
                aws_client_factory=c.resolve(AwsClientFactory),
            ),
        )

        container.singleton(
            AwsClientFactory,
            lambda c: AwsClientFactory(),
        )

        container.singleton(
            ImageGeneratorFactory,
            lambda c: ImageGeneratorFactory(
                aws_client_factory=c.resolve(AwsClientFactory)
            ),
        )

        container.singleton(
            ImageGenerationProvider,
            lambda c: ImageGenerationProvider(
                image_generator_factory=c.resolve(ImageGeneratorFactory),
                file_manager_factory=c.resolve(FileManagerFactory),
            ),
        )
        container.singleton(
            ImageGenerationManager,
            lambda c: ImageGenerationManager(
                image_generation_provider=c.resolve(ImageGenerationProvider)
            ),
        )

        container.singleton(
            PersistenceFactory,
            lambda c: PersistenceFactory(
                environment_variables=c.resolve(LanguageModelCommonEnvironmentVariables)
            ),
        )

        container.singleton(
            OCRExtractorFactory,
            lambda c: OCRExtractorFactory(
                aws_client_factory=c.resolve(AwsClientFactory),
                file_manager_factory=c.resolve(FileManagerFactory),
            ),
        )

        return container
