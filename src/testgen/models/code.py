from typing import Literal, Optional, List

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class FunctionDescription(BaseModel):
    """Description of Python function (global function or class method)"""
    name: str = Field(description='Name of the function or method')
    body: str = Field(
        description='Full source code of the function or method including decorators, name, docstring and comments')


class FileMessage(BaseMessage):
    """Message to keep information about a file"""

    type: Literal['file'] = 'file'
    """The type of the message (used for deserialization). Defaults to "file"."""

    functions: Optional[List[FunctionDescription]] = None
    """The list of functions"""

    test: Optional['TestFileMessage'] = None
    """The unit test file"""


class FunctionMessage(BaseMessage):
    """Message to keep information about a function"""

    type: Literal['code'] = 'function'
    """The type of the message (used for deserialization). Defaults to "function"."""

    file_message: FileMessage
    """Link to the file message"""

    generated_code: Optional[str] = None
    """The generated unit test code"""


class TestFileMessage(BaseMessage):
    """Message to keep information about a test file"""

    type: Literal['test_file'] = 'test_file'
    """The type of the message (used for deserialization). Defaults to "test_file"."""
