from typing import Dict, Any

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from testgen.pipeline.base import BasePipeline


class GeneratedCode(BaseModel):
    source_code: str = Field(description='Source code of generated unit test')


class GeneratePipeline(BasePipeline):

    def get_preprocessor(self) -> Runnable:
        def func(input_data: Dict[str, Any]) -> Dict[str, str]:
            messages = input_data['messages']
            messages = list(filter(lambda x: not isinstance(x, SystemMessage), messages))
            return {
                'messages': messages,
                'format_instructions': self.get_output_parser().get_format_instructions()
            }

        return RunnableLambda(func)

    def get_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessage(content="""\
You are a world-class Python developer with an eagle eye for unintended bugs and edge cases.\
You write careful, accurate unit tests. \
When asked to reply only with code, you write all of your code in a single block.\
""")
        user_message = HumanMessagePromptTemplate.from_template("""\
Using Python and the `unittest` package, write a suite of unit tests for the function, following the cases above. \
Include helpful comments to explain each line.
Make sure the generated Python code is compilable.
**Do not include explanation of the code.**

{format_instructions}
""")
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name='messages'),
            user_message
        ])
        return prompt

    def get_output_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=GeneratedCode)

    def get_postprocessor(self) -> Runnable:
        return RunnableLambda(lambda x: x.source_code)
