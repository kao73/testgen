from typing import Dict

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from testgen.models import FunctionMessage
from testgen.pipeline.base import BasePipeline


class ExplainPipeline(BasePipeline):

    def get_preprocessor(self) -> Runnable:
        def func(function: FunctionMessage) -> Dict[str, str]:
            return {
                'full_path': function.file_message.id,
                'full_source_code': function.file_message.content,
                'function_code': function.content,
            }

        return RunnableLambda(func)

    def get_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessage(content="""\
You are a world-class Python developer with an eagle eye for unintended bugs and edge cases. \
You have a full source code of a Python module and you need to carefully explain code of a single \
function with great detail and accuracy. \
You organize your explanations in markdown-formatted, bulleted lists.\
""")
        full_source_code = HumanMessagePromptTemplate.from_template("""\
Full Python module source code with full path: {full_path}:

```python
{full_source_code}
```
""")
        user_message = HumanMessagePromptTemplate.from_template("""\
Review and explain the following Python function of the module above. \
Review what each element of the function is doing precisely and what the author's intentions may have been. \
Organize your explanation as a markdown-formatted, bulleted list.

```python
{function_code}
```
""")
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            full_source_code,
            user_message
        ])
        return prompt
