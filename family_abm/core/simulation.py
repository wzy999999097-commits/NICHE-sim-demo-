from __future__ import annotations
from typing import Any, Callable, Optional
from .environment import Environment
from .scheduler import Scheduler


class Simulation:
    def __init__(
        self,
        environment: Environment,
        scheduler: Optional[Scheduler] = None,
    ):
        self.environment = environment
        self.scheduler = scheduler or Scheduler()
        self.recorders: list[Any] = []
        self.hooks: dict[str, list[Callable]] = {"pre_step": [], "post_step": []}
        self.current_step = 0

    def add_recorder(self, recorder: Any) -> None:
        self.recorders.append(recorder)

    def add_hook(self, stage: str, func: Callable) -> None:
        if stage in self.hooks:
            self.hooks[stage].append(func)

    def _run_hooks(self, stage: str) -> None:
        for hook in self.hooks.get(stage, []):
            hook(self)

    def step(self) -> None:
        self._run_hooks("pre_step")
        self.scheduler.step(self.environment)
        for recorder in self.recorders:
            recorder.record(self.environment)
        self._run_hooks("post_step")
        self.current_step += 1

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def run_until(self, condition: Callable[[Environment], bool], max_steps: int = 1000) -> None:
        for _ in range(max_steps):
            if condition(self.environment):
                break
            self.step()

    def reset(self) -> None:
        self.environment.time = 0
        self.current_step = 0
        for recorder in self.recorders:
            recorder.reset()

    def get_results(self) -> dict[str, Any]:
        return {
            "steps": self.current_step,
            "environment": self.environment.get_global_state(),
            "recordings": [r.get_data() for r in self.recorders],
        }
