from typing import Dict

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from testgen.models import FileMessage
from testgen.models.code import FileDescription
from testgen.pipeline.base import BasePipeline


class DescribePipeline(BasePipeline):

    def get_preprocessor(self) -> Runnable:
        def func(file: FileMessage) -> Dict[str, str]:
            return {
                'file_path': file.id,
                'python_code': file.content,
                'format_instructions': self.get_output_parser().get_format_instructions()
            }

        return RunnableLambda(func)

    def get_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessage(content="""\
You are a world-class Python developer with an eagle eye for analysing Python code structure. \
Carefully review source code with great detail and accuracy.\
""")
        user_message = HumanMessagePromptTemplate.from_template("""\
Review the Python code of the file with full path {file_path}. \
Describe the Python source code to extract:
- classes and methods with source code 
- functions with source codes
**Do not provide the code explanation***

```python
{python_code}
```

{format_instructions}
""")
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            user_message
        ])
        return prompt

    def get_output_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=FileDescription)
