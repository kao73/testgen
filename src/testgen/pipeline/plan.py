from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from testgen.pipeline.base import BasePipeline


class PlanPipeline(BasePipeline):

    def get_prompt(self) -> ChatPromptTemplate:
        user_message = HumanMessage("""\
A good unit test suite should aim to:
- Test the function's behavior for a wide range of possible inputs
- Test edge cases that the author may not have foreseen
- Take advantage of the features of `unittest` to make the tests easy to write and maintain
- Be easy to read and understand, with clean code and descriptive names
- Be deterministic, so that the tests always pass or fail in the same way

To help unit test the function above, list diverse scenarios that the function should be able to handle \
(and under each scenario, include a few examples as sub-bullets).\
""")
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name='messages'),
            user_message
        ])
        return prompt
