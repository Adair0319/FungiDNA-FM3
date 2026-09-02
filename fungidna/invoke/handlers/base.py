"""Task-handler base class and pluggable weight loader."""
import os
from pathlib import Path


class WeightsNotFoundError(Exception):
    def __init__(self, task):
        super().__init__(f"no checkpoint registered for task {task!r}")
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
        raise WeightsNotFoundError(task)


class TaskHandler:
    task = None  # Task enum member, set by subclass

    def run(self, request):
        raise NotImplementedError
