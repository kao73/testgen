import logging
from typing import List, Annotated

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.branch import RunnableBranch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from testgen.di import DIContainer
from testgen.graph.base import BaseGraph
from testgen.models import CodeBlockDescription
from testgen.models.code import CodeBlockMessage, FileDescription, CodeBlockType, ClassDescription
from testgen.pipeline import FormatFunctionPipeline, FormatMethodPipeline
from testgen.pipeline.explain import ExplainPipeline
from testgen.pipeline.generate import GeneratePipeline
from testgen.pipeline.plan import PlanPipeline

logger = logging.getLogger(__name__)


class InputProcessorState(TypedDict):
    code_block: BaseMessage


class OutputProcessorState(TypedDict):
    code_blocks: Annotated[List[BaseMessage], add_messages]


class ProcessorState(InputProcessorState, OutputProcessorState):
    source_code: str
    messages: Annotated[List[BaseMessage], add_messages]


class ProcessorGraph(BaseGraph):
    node_name = 'Processor'
    input_schema = InputProcessorState

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explain_pipeline = ExplainPipeline()
        self.format_function_pipeline = FormatFunctionPipeline().get_pipeline()
        self.format_method_pipeline = FormatMethodPipeline().get_pipeline()
        self.plan_pipeline = PlanPipeline()
        self.generate_pipeline = GeneratePipeline()

    def format(self, state: InputProcessorState) -> ProcessorState:
        code_block = state['code_block']
        code_type = code_block.description._type.name
        branch = RunnableBranch(
            (lambda x: code_type == 'function', self.format_function_pipeline),
            (lambda x: code_type == 'method', self.format_method_pipeline),
            RunnableLambda(lambda x: x.body)
        )
        formatted_code = branch.invoke(code_block)
        return {
            'source_code': formatted_code,
        }

    def explain(self, state: ProcessorState) -> ProcessorState:
        code_block = state['code_block']
        source_code = state['source_code']
        input_data = {
            'code_block': code_block,
            'source_code': source_code,
        }
        pipeline = self.explain_pipeline.get_pipeline()
        response = pipeline.invoke(input_data)
        # grab chat messages for the further processing
        history = (self.explain_pipeline.get_preprocessor() | self.explain_pipeline.get_prompt()).invoke(input_data)
        messages = history.to_messages()
        messages = add_messages(messages, AIMessage(content=response))
        return {
            'messages': messages
        }
        pass

    def plan(self, state: ProcessorState) -> ProcessorState:
        messages = state['messages']
        input_data = {
            'messages': messages,
        }
        response = self.plan_pipeline.get_pipeline().invoke(input_data)
        # grab chat messages for the further processing
        history = (self.plan_pipeline.get_preprocessor() | self.plan_pipeline.get_prompt()).invoke(input_data)
        messages = add_messages(messages, history.to_messages())
        messages.append(AIMessage(content=response))
        return {
            'messages': messages
        }

    def generate(self, state: ProcessorState) -> OutputProcessorState:
        messages = state['messages']
        code_block = state['code_block']
        input_data = {
            'messages': messages,
        }
        response = self.generate_pipeline.get_pipeline().invoke(input_data)
        code_block.generated_code = response
        return {
            'code_blocks': [state['code_block']]
        }

    def build(self) -> CompiledStateGraph:
        graph_builder = StateGraph(
            state_schema=ProcessorState,
            input=InputProcessorState,
            output=OutputProcessorState
        )

        # define nodes
        graph_builder.add_node('Format', self.format)
        graph_builder.add_node('Explain', self.explain)
        graph_builder.add_node('Plan', self.plan)
        graph_builder.add_node('Generate', self.generate)

        # define edges
        graph_builder.add_edge(START, 'Format')
        graph_builder.add_edge('Format', 'Explain')
        graph_builder.add_edge('Explain', 'Plan')
        graph_builder.add_edge('Plan', 'Generate')
        graph_builder.add_edge('Generate', END)

        graph = graph_builder.compile()
        return graph


if __name__ == '__main__':
    di = DIContainer()
    di.wire(packages=[
        'testgen',
        'testgen.graph',
        'testgen.tools',
    ])
    g = ProcessorGraph()
    response = g.run({'code_block': CodeBlockMessage(
        description=CodeBlockDescription(
            name='client',
            body="""\
model_settings = self.settings.model
model_type = model_settings.type
params = model_settings.params
if model_type.name == 'openai':
    from langchain_openai import ChatOpenAI
    client = ChatOpenAI(**params)
else:
    raise NotSupportedError(f'Unsupported model type: {model_type.name}')
return client
""",
            docstring='Returns ChatGPT client to communicate with LLM',
            signature='def client(self) -> BaseChatModel',
            _type=CodeBlockType.method,
            _file=FileDescription(
                file_path='folder1/test2.py',
                imports="""
from functools import cache
from sqlite3 import NotSupportedError

from langchain_core.language_models import BaseChatModel

from testgen.settings import Settings
""",
                classes=[],
                functions=[],
            ),
            _class=ClassDescription(
                name='ChatModel',
                docstring='LLM connector',
                signature='class ChatModel',
                constructor="""
def __init__(self, settings: Settings):
    self.settings = settings
"""
            ),
        )
    )})
    print(response)
