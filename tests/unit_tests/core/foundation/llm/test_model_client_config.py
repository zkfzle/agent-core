# coding: utf-8
import pytest
from pydantic import ValidationError

from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig


def test_model_client_config_accepts_supported_providers():
    cfg = ModelClientConfig(
        client_provider="OpenAI",
        api_key="sk-test",
        api_base="http://localhost",
    )
    assert cfg.client_provider == "OpenAI"

    cfg2 = ModelClientConfig(
        client_provider="SiliconFlow",
        api_key="sk-test",
        api_base="http://localhost",
    )
    assert cfg2.client_provider == "SiliconFlow"


def test_model_client_config_rejects_unknown_provider():
    with pytest.raises(ValidationError) as error:
        ModelClientConfig.model_validate(
            {
                "client_provider": "invalid_provider",
                "api_key": "sk-test",
                "api_base": "http://localhost",
            }
        )
    assert error.value.errors()[0]["type"] == "literal_error"
