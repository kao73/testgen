from typing import List, Annotated

from langchain_core.messages.base import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from testgen.di import DIContainer
from testgen.graph.base import BaseGraph
from testgen.tools.storage import list_files


class InputScannerState(TypedDict):
    source_folder: str


class OutputScannerState(TypedDict):
    files: Annotated[List[BaseMessage], add_messages]


class ScannerState(InputScannerState, OutputScannerState):
    pass


class ScannerGraph(BaseGraph):
    node_name = 'Scanner'
    input_schema = InputScannerState

    @staticmethod
    def scan_source_folder(state: InputScannerState) -> ScannerState:
        """Scan file storage for all available Python files"""
        source_folder = state['source_folder']
        files = list_files(
            folder=source_folder,
        )
        return {
            'files': files,
        }

    @staticmethod
    def filter(state: ScannerState) -> ScannerState:
        """Filter messages that not suitable for generating unit tests"""
        # TODO implement real filter using LLM
        files = list(state['files'])
        filtered = []
        for file in files:
            if '3' in file.id:
                filtered.append(RemoveMessage(id=file.id))
        return {
            'files': filtered
        }

    def build(self) -> CompiledStateGraph:
        graph_builder = StateGraph(
            input=InputScannerState,
            state_schema=ScannerState,
            output=OutputScannerState
        )

        # define nodes
        graph_builder.add_node('Scan', self.scan_source_folder)
        graph_builder.add_node('Filter', self.filter)

        # define edges
        graph_builder.add_edge(START, 'Scan')
        graph_builder.add_edge('Scan', 'Filter')
        graph_builder.add_edge('Filter', END)

        graph = graph_builder.compile()
        return graph


if __name__ == '__main__':
    di = DIContainer()
    di.wire(packages=[
        'testgen',
        'testgen.graph',
        'testgen.tools',
    ])
    g = ScannerGraph()
    response = g.run({'source_folder': 'src'})
    print(response)
