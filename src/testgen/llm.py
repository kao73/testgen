from functools import cache

from langchain_core.language_models import BaseChatModel

from testgen.settings import Settings


class ChatModel:
    """LLM connector"""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    @cache
    def client(self) -> BaseChatModel:
        """Returns ChatGPT client to communicate with LLM"""
        model_settings = self.settings.model
        model_type = model_settings.type
        params = model_settings.params
        if model_type.name == 'openai':
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(**params)
        else:
            raise NotImplementedError(f'Unsupported model type: {model_type.name}')
        return client
