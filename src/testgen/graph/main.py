from typing import List, Annotated

from langchain_core.messages.base import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from testgen.graph.base import BaseGraph
from testgen.graph.generator import GeneratorGraph
from testgen.graph.scanner import ScannerGraph
from testgen.graph.writer import WriterGeneratorGraph


class InputMainState(TypedDict):
    source_folder: str
    target_folder: str


class OutputMainState(TypedDict):
    tests: Annotated[List[BaseMessage], add_messages]


class MainState(InputMainState, OutputMainState):
    files: Annotated[List[BaseMessage], add_messages]


class MainGraph(BaseGraph):
    name = 'Main'
    input_schema = InputMainState

    def build(self) -> CompiledStateGraph:
        graph_builder = StateGraph(
            state_schema=MainState,
            input=InputMainState,
            output=OutputMainState
        )

        # define nodes
        scanner = ScannerGraph()
        graph_builder.add_node(scanner.name, scanner.build(), input=scanner.input_schema)
        generator = GeneratorGraph()
        graph_builder.add_node(generator.name, generator.build(), input=generator.input_schema)
        writer = WriterGeneratorGraph()
        graph_builder.add_node(writer.name, writer.build(), input=writer.input_schema)

        # define edges
        graph_builder.add_edge(START, scanner.name)
        graph_builder.add_edge(scanner.name, generator.name)
        graph_builder.add_edge(generator.name, writer.name)
        graph_builder.add_edge(writer.name, END)

        graph = graph_builder.compile()
        return graph
