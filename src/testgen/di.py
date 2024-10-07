from dependency_injector import containers, providers

from testgen.llm import ChatModel
from testgen.service.python import CodeExtractor
from testgen.settings import Settings


class DIContainer(containers.DeclarativeContainer):
    settings = providers.Singleton(
        Settings
    )

    model = providers.Singleton(
        ChatModel,
        settings
    )

    code_extractor = providers.Singleton(
        CodeExtractor,
        settings
    )
