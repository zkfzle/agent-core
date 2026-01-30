from typing import List

from openjiuwen.core.utils.llm.model_utils.model_factory import ModelFactory


class OpenAIClient:
    def __init__(self, model_name, base_url, api_key, **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

        self._llm = ModelFactory().get_model("openai", api_key=self.api_key, api_base=base_url, timeout=300)

    async def ainvoke(self, messages, *args, **kwargs):
        try:
            result = await self._llm.ainvoke(self.model_name, messages, *args, **kwargs)
            keys = kwargs.get("stop")
            if keys:
                s_keys = [k.replace("/", "") for k in keys]
                for i in range(len(keys)):
                    if s_keys[i] in result.content:
                        result.content += keys[i]
            return result
        except Exception as e:
            if "Error code: 400" in str(e):
                first_assistant_message_idx = None
                for i, message in enumerate(messages):
                    if isinstance(message, dict):
                        role = message["role"]
                    else:
                        role = message.role
                    if role == "assistant":
                        first_assistant_message_idx = i
                        break
                if first_assistant_message_idx is not None:
                    messages.pop(first_assistant_message_idx)
                return await self._llm.ainvoke(self.model_name, messages, *args, **kwargs)
            else:
                raise e

    def invoke(self, messages, *args, **kwargs):
        try:
            result = self._llm.invoke(self.model_name, messages, *args, **kwargs)
            keys = kwargs.get("stop")
            if keys:
                s_keys = [k.replace("/", "") for k in keys]
                for i in range(len(keys)):
                    if s_keys[i] in result.content:
                        result.content += keys[i]
            return result
        except Exception as e:
            if "Error code: 400" in str(e):
                first_assistant_message_idx = None
                for i, message in enumerate(messages):
                    if isinstance(message, dict):
                        role = message["role"]
                    else:
                        role = message.role
                    if role == "assistant":
                        first_assistant_message_idx = i
                        break
                if first_assistant_message_idx is not None:
                    messages.pop(first_assistant_message_idx)
                return self._llm.invoke(self.model_name, messages, *args, **kwargs)

