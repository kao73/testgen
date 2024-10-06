from abc import ABC, abstractmethod
from typing import Any, Dict, final

from dependency_injector.wiring import Provide, inject
from langgraph.graph.state import CompiledStateGraph

from testgen.di import DIContainer
from testgen.llm import ChatModel
from testgen.settings import Settings


class BaseGraph(ABC):

    @inject
    def __init__(
            self,
            settings: Settings = Provide[DIContainer.settings],
            model: ChatModel = Provide[DIContainer.model],
    ):
        self.settings = settings
        self.model = model

        # Check if the child class has defined 'input_schema'
        if not hasattr(self, 'input_schema'):
            raise NotImplementedError(f"{self.__class__.__name__} must define 'input_schema'")

    @final
    @property
    def name(self) -> str:
        name = getattr(self, 'node_name', None)
        if not name:
            raise NotImplementedError(f"{self.__class__.__name__} must define 'node_name'")
        return name

    @abstractmethod
    def build(self) -> CompiledStateGraph:
        ...

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        graph = self.build()
        response = graph.invoke(input_data)
        return response
