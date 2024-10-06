from typing import Dict, List

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from testgen.models.code import CodeBlockMessage
from testgen.pipeline.base import BasePipeline


class MergePipeline(BasePipeline):

    def get_preprocessor(self) -> Runnable:
        def func(code_blocks: List[CodeBlockMessage]) -> Dict[str, str]:
            unit_test_files = [
                f"```python\n{b.generated_code}\n```"
                for b in code_blocks
            ]
            unit_test_files = '\n'.join(unit_test_files)
            return {
                'unit_test_files': unit_test_files,
            }

        return RunnableLambda(func)

    def get_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessage(content="""\
You are a Python code assistant. \
Your task is to merge two or more Python unit test files into a single one, while preserving all functionality and \
ensuring that no test cases are duplicated. \
Analyze each test file, combine them logically, and modify any necessary imports or setup configurations \
to ensure that the merged file will run correctly without errors. \
Retain all docstrings and comments.\
""")
        user_message = HumanMessagePromptTemplate.from_template("""\
I have two Python unit test files that I need to merge into a single one. \
Please analyze the test cases, imports, and setup configurations of each file and combine them into one \
coherent unit test file. \
Make sure to avoid duplication of test cases and ensure the merged file will run correctly without errors.
**Do not explain the Python code**
Respond by merged Python file in raw format

{unit_test_files}
""")
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            user_message
        ])
        return prompt
