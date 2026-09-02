"""Task-handler base class and pluggable weight loader."""
import os
from pathlib import Path

import yaml

#: Default location of the task -> checkpoint YAML mapping. Overridable via the
#: ``FUNGIDNA_WEIGHTS_CONFIG`` environment variable.
DEFAULT_WEIGHTS_CONFIG = "~/.config/fungidna/weights.yaml"


class WeightsNotFoundError(Exception):
    def __init__(self, task, message=None):
        super().__init__(message or f"no checkpoint registered for task {task!r}")
        self.task = task


class WeightLoader:
    def __init__(self, registry=None):
        self.registry = dict(registry or {})

    def resolve(self, task):
        if task in self.registry:
            return Path(self.registry[task])
        env_var = f"FUNGIDNA_{task.upper()}_WEIGHTS"
        env_val = os.environ.get(env_var)
        if env_val:
            return Path(env_val)
        path_from_config = self._resolve_from_config_file(task)
        if path_from_config is not None:
            return Path(path_from_config)
        raise WeightsNotFoundError(task)

    def _resolve_from_config_file(self, task):
        config_file = Path(
            os.environ.get("FUNGIDNA_WEIGHTS_CONFIG", DEFAULT_WEIGHTS_CONFIG)
        ).expanduser()
        if not config_file.is_file():
            return None
        try:
            with config_file.open() as fh:
                data = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as exc:
            raise WeightsNotFoundError(
                task,
                f"weights config file {str(config_file)} is unreadable or not "
                f"valid YAML: {exc}; no checkpoint registered for task {task!r}",
            ) from exc
        if isinstance(data, dict):
            path = data.get(task)
            if path:
                return path
        return None


class TaskHandler:
    task = None  # Task enum member, set by subclass

    def run(self, request):
        raise NotImplementedError
