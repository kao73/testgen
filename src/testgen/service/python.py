import ast
from typing import List

from testgen.models.code import FunctionDescription
from testgen.settings import Settings


class CodeExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.exclude = settings.exclude

    def extract_functions(self, source_code: str) -> List[FunctionDescription]:
        tree = ast.parse(source_code)
        source_code_lines = source_code.splitlines(keepends=True)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Exclude class constructors (usually named __init__)
                if node.name not in self.exclude:
                    # Get the source code of the function including decorators
                    function_source = self.get_function_source(
                        node,
                        source_code_lines
                    )
                    function_description = FunctionDescription(
                        name=node.name,
                        body=function_source
                    )
                    functions.append(function_description)
        return functions

    @staticmethod
    def get_function_source(node: ast.FunctionDef, source_code_lines: List[str]) -> str:
        """Get the full source code of a function node including decorators"""
        start_line = node.lineno - 1  # Zero-indexed
        end_line = node.end_lineno  # This is already zero-indexed

        # Gather lines including decorators
        decorators_lines = []
        for decorator in node.decorator_list:
            decorator_start = decorator.lineno - 1  # Zero-indexed
            decorator_end = decorator.end_lineno  # This is already zero-indexed
            decorators_lines.extend(source_code_lines[decorator_start:decorator_end])

        # Combine decorators and function body
        function_lines = source_code_lines[start_line:end_line]
        return ''.join(decorators_lines + function_lines)
