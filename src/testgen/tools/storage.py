import logging
from pathlib import Path
from typing import List, Optional

from dependency_injector.wiring import Provide, inject
from langchain_core.messages.base import BaseMessage

from testgen.di import DIContainer
from testgen.models import FileMessage
from testgen.settings import Settings

logger = logging.getLogger(__name__)


@inject
def list_files(
        folder: str = None,
        pattern: str = '**/*.py',
        settings: Settings = Provide[DIContainer.settings]
) -> List[BaseMessage]:
    """
    Returns a list of files in the storage folder by pattern with folder validation.
    :param folder: Optional folder relative to the storage folder in settings.
    :param pattern: Optional pattern to filter files, e.g., '*.py'.
    :param settings: Settings object provided by DI containing storage configuration.
    :return: List of FileMessage objects containing file content and file names.
    """
    # Combine the storage folder with the provided relative folder (if any)
    storage_folder = Path(settings.storage_folder).resolve()  # Resolve storage folder to absolute path
    base_folder = storage_folder / folder if folder else storage_folder  # Treat 'folder' as relative to 'storage_folder'
    base_folder = base_folder.resolve()  # Resolve the final path to ensure it's absolute

    # Ensure the base_folder is within the allowed storage folder
    if not base_folder.is_relative_to(storage_folder):
        raise ValueError(f"Access to the folder '{base_folder}' is restricted or invalid.")

    # Check if the folder exists and is a directory
    if not base_folder.exists() or not base_folder.is_dir():
        raise FileNotFoundError(f"The folder '{base_folder}' does not exist or is not a directory.")

    # Initialize an empty list to store FileMessage objects
    file_messages = []

    # Iterate over the files matching the pattern
    for file_path in base_folder.glob(pattern):
        if file_path.is_file():
            # Read the file content
            content = file_path.read_text()

            # Get file path relative to base folder
            relative_path = file_path.relative_to(base_folder)

            # Append the FileMessage to the result list
            file_messages.append(FileMessage(content=content, id=relative_path))

    return file_messages


@inject
def write_files(
        files: List[BaseMessage],
        folder: Optional[str] = None,
        settings: Settings = Provide[DIContainer.settings]
) -> None:
    """
    Write files into the file storage.

    :param files: List of files to be stored into the file storage.
    :param folder: Optional folder relative to the storage folder in settings.
    :param settings: Settings object provided by DI containing storage configuration.
    """
    if not files:
        logger.warning('No files to write')
        return

        # Determine the base storage path from settings
    base_storage_path = Path(settings.storage_folder).resolve()

    # If a folder is provided, append it to the base storage path
    if folder:
        storage_path = base_storage_path / folder
    else:
        storage_path = base_storage_path

    # Create the directory if it doesn't exist
    storage_path.mkdir(parents=True, exist_ok=True)
    logger.debug('Storage path created: %s', storage_path)

    # Iterate over each file in the list and write it to the storage
    for file in files:
        # Construct the full path for the file
        file_path = storage_path / file.id

        try:
            # Make sure the file folder does exist
            file_path.parents[0].mkdir(parents=True, exist_ok=True)

            # Write the file content to disk
            with file_path.open('wb') as f:
                f.write(file.content.encode('utf-8'))
            logger.debug('File written: %s', file_path)
        except Exception as e:
            logger.error('Error writing file %s: %s', file.filename, e)
            raise e

    logger.info('All files have been written to %s', storage_path)
