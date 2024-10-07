from typing import List, Annotated

from langchain_core.messages.base import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from testgen.di import DIContainer
from testgen.graph import BaseGraph
from testgen.models import FileMessage
from testgen.tools import write_files


class InputWriterState(TypedDict):
    target_folder: str
    tests: Annotated[List[BaseMessage], add_messages]


class OutputWriterState(TypedDict):
    pass


class WriterState(InputWriterState, OutputWriterState):
    pass


class WriterGraph(BaseGraph):
    node_name = 'Writer'
    input_schema = InputWriterState

    def write(self, state: InputWriterState) -> WriterState:
        target_folder = state['target_folder']
        tests = state['tests']
        write_files(
            folder=target_folder,
            files=tests
        )
        return {
            'tests': tests
        }

    def build(self) -> CompiledStateGraph:
        graph_builder = StateGraph(
            input=InputWriterState,
            state_schema=WriterState,
            output=OutputWriterState
        )

        # define nodes
        graph_builder.add_node('Write', self.write)

        # define edges
        graph_builder.add_edge(START, 'Write')
        graph_builder.add_edge('Write', END)

        graph = graph_builder.compile()
        return graph


if __name__ == '__main__':
    di = DIContainer()
    di.wire(packages=[
        'testgen',
        'testgen.graph',
        'testgen.tools',
    ])
    g = WriterGraph()
    response = g.run({'tests': [
        FileMessage(content='test', id='tests/test1.py')
    ]})
    print(response)
