from __future__ import annotations
from typing import Any, Optional
from .agent import Agent


class Environment:
    def __init__(self, width: int = 100, height: int = 100):
        self.agents: dict[str, Agent] = {}
        self.time: int = 0
        self.width = width
        self.height = height
        self.properties: dict[str, Any] = {}

    def add_agent(self, agent: Agent, x: Optional[float] = None, y: Optional[float] = None) -> None:
        self.agents[agent.id] = agent
        agent.environment = self
        if x is not None:
            agent.set_attribute("x", x)
        if y is not None:
            agent.set_attribute("y", y)

    def remove_agent(self, agent_id: str) -> None:
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.environment = None
            del self.agents[agent_id]

    def get_agents(self) -> list[Agent]:
        return list(self.agents.values())

    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    def get_agents_by_type(self, agent_type: type) -> list[Agent]:
        return [a for a in self.agents.values() if isinstance(a, agent_type)]

    def step(self) -> None:
        for agent in self.get_agents():
            if agent.alive:
                agent.step()
        self.time += 1

    def get_global_state(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "population": len(self.agents),
            "properties": dict(self.properties),
        }
