from typing import Dict, Any

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from testgen.pipeline.base import BasePipeline


class ExplainPipeline(BasePipeline):

    def get_preprocessor(self) -> Runnable:
        def func(input_data: Dict[str, Any]) -> Dict[str, str]:
            code_block = input_data['code_block']
            source_code = input_data['source_code']
            return {
                'file': code_block.description._file.file_path,
                'source_code': source_code
            }

        return RunnableLambda(func)

    def get_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessage(content="""\
You are a world-class Python developer with an eagle eye for unintended bugs and edge cases. \
You carefully explain code with great detail and accuracy. \
You organize your explanations in markdown-formatted, bulleted lists.\
""")
        user_message = HumanMessagePromptTemplate.from_template("""\
Review and explain the following Python function of the {file} module. \
Review what each element of the function is doing precisely and what the author's intentions may have been. \
Organize your explanation as a markdown-formatted, bulleted list.

```python
{source_code}
```
""")
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            user_message
        ])
        return prompt
