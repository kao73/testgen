import logging
from typing import List, Annotated

from langchain_core.messages.base import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from testgen.di import DIContainer
from testgen.graph.base import BaseGraph
from testgen.graph.processor import ProcessorGraph
from testgen.models.code import CodeBlockType, CodeBlockMessage
from testgen.pipeline.describe import DescribePipeline, FileDescription
from testgen.pipeline.merge import MergePipeline

logger = logging.getLogger(__name__)


class InputGeneratorState(TypedDict):
    files: Annotated[List[BaseMessage], add_messages]


class OutputGeneratorState(TypedDict):
    tests: Annotated[List[BaseMessage], add_messages]


class GeneratorState(InputGeneratorState, OutputGeneratorState):
    code_blocks: Annotated[List[BaseMessage], add_messages]


class GeneratorGraph(BaseGraph):
    node_name = 'Generator'
    input_schema = InputGeneratorState

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.describe_pipeline = DescribePipeline()
        self.merge_pipeline = MergePipeline().get_pipeline()

    def describe(self, state: InputGeneratorState) -> GeneratorState:
        """Describe python file and extract class names, methods and functions"""
        files = list(state['files'])
        descriptions: List[FileDescription] = self.describe_pipeline.get_pipeline().batch(files)
        # collect all code blocks
        code_blocks = []
        for file in files:
            desc = next((desc for desc in descriptions if file.id == desc.file_path), None)
            if desc:
                file.description = desc
                # collect file code blocks
                file_code_blocks = []
                for func_desc in desc.functions:
                    func_desc._type = CodeBlockType.function
                    func_desc._file = desc
                    file_code_blocks.append(func_desc)
                for class_desc in desc.classes:
                    for method_desc in class_desc.methods:
                        method_desc._type = CodeBlockType.method
                        method_desc._file = desc
                        method_desc._class = class_desc
                        file_code_blocks.append(method_desc)
                file.code_blocks = file_code_blocks
                code_blocks.extend(file_code_blocks)
            else:
                logger.warning('No code descriptions found for %s file: ', file.id)
        code_blocks = [
            CodeBlockMessage(description=b)
            for b in code_blocks
        ]
        return {
            'code_blocks': code_blocks,
            'files': [],
            'tests': []
        }

    def merge(self, state: GeneratorState) -> GeneratorState:
        files = state['files']
        code_blocks = state['code_blocks']
        # collect generated blocks for each file
        for file in files:
            file_code_blocks = list(filter(lambda x: x.description._file.file_path == file.id, code_blocks))
            if len(file_code_blocks) > 1:
                generated_code = self.merge_pipeline.invoke(file_code_blocks)
            else:
                generated_code = file_code_blocks[0].generated_code
            file.generated_code = generated_code
        pass

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
                Send(processor.name, {'code_block': code_block})
                for code_block in state['code_blocks']
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
