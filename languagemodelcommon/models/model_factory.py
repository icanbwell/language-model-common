from __future__ import annotations

import json
import logging
import os
from typing import List, Any, Dict, TYPE_CHECKING

from google.oauth2 import service_account
from google.oauth2.service_account import Credentials
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from languagemodelcommon.aws.aws_client_factory import AwsClientFactory
from types_boto3_bedrock_runtime.client import BedrockRuntimeClient

from languagemodelcommon.configs.schemas.config_schema import (
    ModelConfig,
    ModelParameterConfig,
    ChatModelConfig,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

if TYPE_CHECKING:
    from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
        LanguageModelCommonEnvironmentVariables,
    )

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.LLM)


class ModelFactory:
    def __init__(
        self,
        *,
        environment_variables: "LanguageModelCommonEnvironmentVariables | None" = None,
        aws_client_factory: AwsClientFactory | None = None,
    ) -> None:
        self._environment_variables = environment_variables
        self._aws_client_factory = aws_client_factory or AwsClientFactory(
            environment_variables=environment_variables
        )

    # noinspection PyMethodMayBeStatic
    def get_model(self, chat_model_config: ChatModelConfig) -> BaseChatModel:
        if chat_model_config is None:
            raise ValueError("chat_model_config must not be None")
        if not isinstance(chat_model_config, ChatModelConfig):
            raise TypeError(
                f"chat_model_config must be ChatModelConfig, got {type(chat_model_config)}"
            )
        model_config: ModelConfig | None = chat_model_config.model
        if model_config is None:
            # if no model configuration is provided, use the default model
            default_model_provider: str = (
                self._environment_variables.default_model_provider
                if self._environment_variables
                else os.environ.get("DEFAULT_MODEL_PROVIDER", "bedrock")
            )
            default_model_name: str = (
                self._environment_variables.default_model_name
                if self._environment_variables
                else os.environ.get(
                    "DEFAULT_MODEL_NAME",
                    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                )
            )
            model_config = ModelConfig(
                provider=default_model_provider, model=default_model_name
            )

        if model_config is None:
            raise ValueError("model_config must not be None")

        model_vendor: str = model_config.provider
        model_name: str = model_config.model

        model_parameters: List[ModelParameterConfig] | None = (
            chat_model_config.model_parameters
        )

        # convert model_parameters to dict
        model_parameters_dict: Dict[str, Any] = {}
        if model_parameters is not None:
            model_parameter: ModelParameterConfig
            for model_parameter in model_parameters:
                model_parameters_dict[model_parameter.key] = model_parameter.value

        logger.debug(f"Creating ChatModel with parameters: {model_parameters_dict}")
        model_parameters_dict["model"] = model_name
        # model_parameters_dict["streaming"] = True
        llm: BaseChatModel
        if model_vendor == "openai":
            llm = ChatOpenAI(**model_parameters_dict)
        elif model_config.provider == "google":
            scoped_credentials = self.get_google_credentials()
            model_parameters_dict["credentials"] = scoped_credentials
            llm = ChatGoogleGenerativeAI(**model_parameters_dict)
        elif model_config.provider == "bedrock":
            aws_credentials_profile = (
                self._environment_variables.aws_credentials_profile
                if self._environment_variables
                else os.environ.get("AWS_CREDENTIALS_PROFILE")
            )
            aws_region_name = (
                self._environment_variables.aws_region
                if self._environment_variables
                else os.environ.get("AWS_REGION", "us-east-1")
            )

            bedrock_client: BedrockRuntimeClient = (
                self._aws_client_factory.create_bedrock_client()
            )

            # Extract thinking config if present — it must be passed via
            # additional_model_request_fields, not as a top-level kwarg.
            additional_fields: Dict[str, Any] = {}
            thinking_budget: int | None = None
            raw_budget = model_parameters_dict.pop("thinking_budget_tokens", None)
            if raw_budget is not None:
                thinking_budget = int(raw_budget)
            if thinking_budget and thinking_budget > 0:
                additional_fields["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                logger.info(
                    "Extended thinking enabled for %s with budget_tokens=%d",
                    model_name,
                    thinking_budget,
                )

            if additional_fields:
                model_parameters_dict["additional_model_request_fields"] = (
                    additional_fields
                )

            llm = ChatBedrockConverse(
                client=bedrock_client,
                provider="anthropic",
                credentials_profile_name=aws_credentials_profile,
                region_name=aws_region_name,
                **model_parameters_dict,
            )
        elif model_config.provider == "openai":
            llm = ChatOpenAI(**model_parameters_dict)
        else:
            raise ValueError(
                f"Unsupported model vendor: {model_vendor} and model_provider: {model_config.provider} for {model_name}"
            )

        return llm

    def get_google_credentials(self) -> Credentials:
        service_account_json = (
            self._environment_variables.google_credentials_json
            if self._environment_variables
            else os.getenv("GOOGLE_CREDENTIALS_JSON")
        )
        if not service_account_json:
            raise RuntimeError(
                "GOOGLE_CREDENTIALS_JSON env var not set. Please set the environment variable with your service account JSON."
            )
        try:
            creds_info = json.loads(service_account_json)
        except json.JSONDecodeError:
            raise RuntimeError(
                "GOOGLE_CREDENTIALS_JSON is not valid JSON. Please check the formatting of your credentials."
            )
        logger.debug(
            "Loaded GOOGLE_CREDENTIALS_JSON for client_email=%s, project_id=%s, available_keys=%s",
            creds_info.get("client_email"),
            creds_info.get("project_id"),
            [key for key in creds_info.keys() if key != "private_key"],
        )
        required_fields = ["client_email", "private_key", "project_id"]
        missing_fields = [field for field in required_fields if field not in creds_info]
        if missing_fields:
            raise RuntimeError(
                f"Missing required fields in credentials: {', '.join(missing_fields)}. Please check your service account JSON."
            )
        creds: Credentials = service_account.Credentials.from_service_account_info(
            creds_info
        )  # type: ignore[no-untyped-call]
        scoped_credentials: Credentials = creds.with_scopes(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )
        return scoped_credentials
