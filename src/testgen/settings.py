import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings_yaml import YamlBaseSettings

logger = logging.getLogger(__name__)


def get_config_path() -> str:
    home = Path(os.environ.get('TG_HOME', './config')).resolve()
    logger.info('Load config from %s', home)
    return str(home / 'config' / 'config.yaml')


class ModelType(Enum):
    openai = 'openai'
    gigachat = 'gigachat'


class ModelSettings(BaseSettings):
    type: ModelType = Field(description='LLM type')
    params: Dict[str, Any] = Field(default_factory=dict, description='LLM parameters')


class Settings(YamlBaseSettings):
    storage_folder: str
    model: ModelSettings = Field(description='LLM model settings')

    # configure paths to secrets directory and YAML config file
    model_config = SettingsConfigDict(
        secrets_dir='/secrets', yaml_file=get_config_path())
