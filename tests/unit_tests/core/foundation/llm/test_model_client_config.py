# coding: utf-8
import pytest

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ProviderType


def test_model_client_config_accepts_supported_providers():
    cfg = ModelClientConfig(
        client_provider=ProviderType.OpenAI,
        api_key="sk-test",
        api_base="http://localhost",
    )
    assert cfg.client_provider == ProviderType.OpenAI

    cfg2 = ModelClientConfig(
        client_provider=ProviderType.SiliconFlow,
        api_key="sk-test",
        api_base="http://localhost",
    )
    assert cfg2.client_provider == ProviderType.SiliconFlow


def test_model_client_config_model_validate_invalid_provider_raises_base_error():
    with pytest.raises(BaseError) as error:
        ModelClientConfig.model_validate(
            {
                "client_provider": "invalid_provider",
                "api_key": "sk-test",
                "api_base": "http://localhost",
            }
        )
    assert error.value.code == StatusCode.MODEL_CLIENT_CONFIG_INVALID.code
