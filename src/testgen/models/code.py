from enum import Enum
from typing import List, Literal, Optional, Any
from uuid import uuid4

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, PrivateAttr


class CodeBlockType(Enum):
    method = 'method'
    function = 'function'


class CodeBlockDescription(BaseModel):
    """Description of code block like function or method"""
    _type: CodeBlockType = PrivateAttr(default=None)
    _file: 'FileDescription' = PrivateAttr(default=None)
    _class: 'ClassDescription' = PrivateAttr(default=None)
    name: str = Field(description='Name of the function or method')
    signature: str = Field(description='The function or method signature')
    docstring: str = Field(description='Full dockstring of the function or method if explicitly defined')
    body: str = Field(description='Source code of the function or method')

    def __init__(
            self,
            _type: CodeBlockType = None,
            _file: 'FileDescription' = None,
            _class: 'ClassDescription' = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._type = _type
        self._file = _file
        self._class = _class


class ClassDescription(BaseModel):
    """The class description"""
    name: str = Field(description='Name of the class')
    docstring: str = Field(description='Dockstring of the class if explicitly defined')
    signature: str = Field(description='Source code of the class signature')
    constructor: str = Field(description='source code of the class constructor if explicitly defined')
    methods: List[CodeBlockDescription] = Field(description='List of class methods')


class FileDescription(BaseModel):
    """The Python file code description"""
    file_path: str = Field(description='Full path of the Python code file')
    imports: str = Field(description='Source code of the Python imports')
    classes: List[ClassDescription] = Field(description='List of class descriptions')
    functions: List[CodeBlockDescription] = Field(description='List of functions')


class FileMessage(BaseMessage):
    """Message to keep information about a file"""

    type: Literal['file'] = 'file'
    """The type of the message (used for deserialization). Defaults to "file"."""

    description: Optional[FileDescription] = None
    """The code description"""

    code_blocks: Optional[CodeBlockDescription] = None
    """The list of code blocks descriptions"""

    generated_code: Optional[str] = None
    """The generated unit test code"""


class CodeBlockMessage(BaseMessage):
    """Message to keep information about a code block"""

    type: Literal['code'] = 'code'
    """The type of the message (used for deserialization). Defaults to "code"."""

    description: CodeBlockDescription
    """The code block description"""

    generated_code: Optional[str] = None
    """The generated unit test code"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id=str(uuid4()), content='placeholder', **kwargs)
