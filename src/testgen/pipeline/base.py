from abc import ABC, abstractmethod

from dependency_injector.wiring import Provide, inject
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from testgen.di import DIContainer
from testgen.llm import ChatModel


class BasePipeline(ABC):

    @inject
    def __init__(self, model: ChatModel = Provide[DIContainer.model]):
        self.model = model

    def get_pipeline(self) -> Runnable:
        preprocessor = self.get_preprocessor()
        prompt = self.get_prompt()
        model = self.get_model()
        parser = self.get_output_parser()
        postprocessor = self.get_postprocessor()
        return preprocessor | prompt | model | parser | postprocessor

    def get_preprocessor(self) -> Runnable:
        """Returns preprocessor to convert input data into prompt variables"""
        return RunnableLambda(lambda x: x)

    @abstractmethod
    def get_prompt(self) -> ChatPromptTemplate:
        """Returns pipeline prompt template"""
        ...

    def get_model(self) -> BaseChatModel:
        """Returns the model object"""
        return self.model.client

    def get_output_parser(self) -> BaseOutputParser:
        """Returns output parser"""
        return StrOutputParser()

    def get_postprocessor(self) -> Runnable:
        """Returns postprocessor to convert parsed model output to desired format."""
        return RunnableLambda(lambda x: x)
