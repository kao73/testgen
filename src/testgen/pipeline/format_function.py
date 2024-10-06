from typing import Dict

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers.pydantic import PydanticOutputParser

from testgen.models import CodeBlockDescription
from testgen.pipeline.base import BasePipeline
from pydantic import BaseModel, Field

class FormattedCode(BaseModel):
    source_code: str = Field(description='Formatted Python source code')

class FormatFunctionPipeline(BasePipeline):

    def get_preprocessor(self) -> Runnable:
        def func(code_block: CodeBlockDescription) -> Dict[str, str]:
            desc = code_block.description
            return {
                'file': desc._file.file_path,
                'imports': desc._file.imports,
                'signature': desc.signature,
                'docstring': desc.docstring,
                'body': desc.body,
                'format_instructions': self.get_output_parser().get_format_instructions()
            }

        return RunnableLambda(func)

    def get_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessage(content="""\
You are an expert in Python code formatting and PEP 8 standards. \
Your task is to review and fix Python code snippets by applying proper formatting according to the \
Python Enhancement Proposal (PEP 8) guidelines. \
Ensure that the code is properly indented, but do not change names and variables. \
Provide well-formatted, readable Python code.\
""")
        user_message = HumanMessagePromptTemplate.from_template("""\
Here is a Python code snippet that needs to be formatted according to PEP 8 standards. \
Please review and correct the formatting based on proper indentation, line length (maximum 79 characters per line), \
and any other applicable guidelines. \
Make sure the Python code is compilable.
**Do not explain changes you made.**

```python
{imports}

{signature}
'''{docstring}'''
{body}
```

Respond in JSON format of the following schema:
```json
{format_instructions}
```
""")
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            user_message
        ])
        return prompt

    def get_output_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=FormattedCode)

    def get_postprocessor(self) -> Runnable:
        return RunnableLambda(lambda x: x.source_code)