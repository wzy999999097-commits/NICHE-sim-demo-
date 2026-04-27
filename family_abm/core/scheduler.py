from __future__ import annotations
import random
from .agent import Agent
from .environment import Environment


class Scheduler:
    def __init__(self, method: str = "sequential"):
        self.method = method
        self.time = 0

    def schedule(self, environment: Environment) -> list[Agent]:
        agents = environment.get_agents()
        if self.method == "sequential":
            return agents
        elif self.method == "random":
            shuffled = list(agents)
            random.shuffle(shuffled)
            return shuffled
        elif self.method == "random_activation":
            shuffled = list(agents)
            random.shuffle(shuffled)
            n = random.randint(1, max(1, len(shuffled)))
            return shuffled[:n]
        else:
            raise ValueError(f"Unknown scheduler method: {self.method}")

    def step(self, environment: Environment) -> None:
        for agent in self.schedule(environment):
            if agent.alive:
                agent.step()
        self.time += 1
        environment.time = self.time
