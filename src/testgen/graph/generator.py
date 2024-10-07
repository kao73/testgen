import logging
from pathlib import Path
from typing import List, Annotated

from dependency_injector.wiring import Provide, inject
from langchain_core.messages.base import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from testgen.di import DIContainer
from testgen.graph.base import BaseGraph
from testgen.graph.processor import ProcessorGraph
from testgen.models import FunctionMessage, TestFileMessage
from testgen.pipeline.merge import MergePipeline
from testgen.service.python import CodeExtractor

logger = logging.getLogger(__name__)


class InputGeneratorState(TypedDict):
    files: Annotated[List[BaseMessage], add_messages]


class OutputGeneratorState(TypedDict):
    files: Annotated[List[BaseMessage], add_messages]
    tests: Annotated[List[BaseMessage], add_messages]


class GeneratorState(InputGeneratorState, OutputGeneratorState):
    functions: Annotated[List[BaseMessage], add_messages]


class GeneratorGraph(BaseGraph):
    node_name = 'Generator'
    input_schema = InputGeneratorState

    @inject
    def __init__(
            self,
            code_extractor: CodeExtractor = Provide[DIContainer.code_extractor],
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.code_extractor = code_extractor
        self.merge_pipeline = MergePipeline().get_pipeline()

    def describe(self, state: InputGeneratorState) -> GeneratorState:
        """Describe python file and extract class names, methods and functions"""
        files = list(state['files'])
        functions = []
        for file in files:
            file.functions = self.code_extractor.extract_functions(file.content)
            for func in file.functions:
                func_message = FunctionMessage(
                    name=func.name,
                    content=func.body,
                    file_message=file,
                )
                functions.append(func_message)
        return {
            'functions': functions,
            'files': state['files'],
            'tests': [],
        }

    def merge(self, state: GeneratorState) -> OutputGeneratorState:
        files = state['files']
        functions = state['functions']
        tests = []
        for file in files:
            file_functions = list(filter(lambda x: x.file_message.id == file.id, functions))
            if len(file_functions) > 1:
                generated_code = self.merge_pipeline.invoke(file_functions)
            else:
                generated_code = file_functions[0].generated_code
            # generate test file name
            source_file = Path(file.id)
            test_file = source_file.with_name(f"test_{source_file.name}")
            test = TestFileMessage(
                id=test_file,
                content=generated_code,
            )
            file.test = test
            tests.append(test)
        return {
            'files': files,
            'tests': tests,
        }

    def build(self) -> CompiledStateGraph:
        graph_builder = StateGraph(
            input=InputGeneratorState,
            state_schema=GeneratorState,
            output=OutputGeneratorState
        )

        # define nodes
        graph_builder.add_node('Describe', self.describe)
        processor = ProcessorGraph()
        graph_builder.add_node(processor.name, processor.build(), input=processor.input_schema)
        graph_builder.add_node('Merge', self.merge)

        # define edges
        graph_builder.add_edge(START, 'Describe')
        graph_builder.add_conditional_edges(
            'Describe',
            lambda state: [
                Send(processor.name, {'function': function})
                for function in state['functions']
            ],
            [processor.name]
        )
        graph_builder.add_edge(processor.name, 'Merge')
        graph_builder.add_edge('Merge', END)

        graph = graph_builder.compile()
        return graph


if __name__ == '__main__':
    di = DIContainer()
    di.wire(packages=[
        'testgen',
        'testgen.graph',
        'testgen.tools',
    ])
    g = GeneratorGraph()
    response = g.run({'files': []})
    print(response)
