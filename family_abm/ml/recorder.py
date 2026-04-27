from __future__ import annotations
from typing import Any
import pandas as pd
from ..core.environment import Environment


class StateRecorder:
    def __init__(self, record_agents: bool = True, record_environment: bool = True):
        self.record_agents = record_agents
        self.record_environment = record_environment
        self.history: list[dict[str, Any]] = []
        self.agent_history: list[dict[str, Any]] = []

    def record(self, env: Environment) -> None:
        step_data: dict[str, Any] = {
            "time": env.time,
            "population": len(env.agents),
        }
        if self.record_environment:
            step_data["environment"] = env.get_global_state()
        self.history.append(step_data)

        if self.record_agents:
            for agent in env.get_agents():
                agent_state = agent.get_state()
                record: dict[str, Any] = {
                    "time": env.time,
                    "agent_id": agent.id,
                    "agent_type": agent_state["type"],
                    "alive": agent_state["alive"],
                }
                for attr_key, attr_val in agent_state.get("attributes", {}).items():
                    if isinstance(attr_val, (str, int, float, bool)):
                        record[f"attr_{attr_key}"] = attr_val
                for state_key, state_val in agent_state.get("state", {}).items():
                    if isinstance(state_val, (int, float)):
                        record[f"state_{state_key}"] = state_val
                self.agent_history.append(record)

    def get_data(self) -> dict[str, Any]:
        return {"environment": self.history, "agents": self.agent_history}

    def to_dataframe(self) -> pd.DataFrame:
        if self.agent_history:
            return pd.DataFrame(self.agent_history)
        return pd.DataFrame()

    def to_csv(self, filepath: str) -> None:
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def reset(self) -> None:
        self.history.clear()
        self.agent_history.clear()
