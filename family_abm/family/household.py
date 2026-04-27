from __future__ import annotations
from typing import Any, Optional
from ..core.agent import Agent
from .family_member import FamilyMember
from .relationships import Relationship


class Household(Agent):
    def __init__(self, household_id: Optional[str] = None, name: str = "Household", **kwargs):
        super().__init__(agent_id=household_id, **kwargs)
        self.set_attribute("name", name)
        self.members: dict[str, FamilyMember] = {}
        self.relationships: dict[tuple[str, str], Relationship] = {}

        self.set_state_value("total_income", 0.0)
        self.set_state_value("savings", 0.0)
        self.set_state_value("housing_quality", 0.5)
        self.set_state_value("neighborhood_quality", 0.5)
        self.set_state_value("cultural_level", 0.5)
        self.set_state_value("social_capital", 0.5)

    def add_member(self, member: FamilyMember, relation_to_head: str = "member") -> None:
        self.members[member.id] = member
        for existing_id in self.members:
            if existing_id != member.id:
                rel = Relationship(
                    agent_a_id=member.id,
                    agent_b_id=existing_id,
                    relation_type=relation_to_head,
                )
                self.relationships[(member.id, existing_id)] = rel
                self.relationships[(existing_id, member.id)] = rel
        if self.environment is not None:
            self.environment.add_agent(member)

    def remove_member(self, member_id: str) -> None:
        if member_id in self.members:
            del self.members[member_id]
            keys_to_remove = [k for k in self.relationships if member_id in k]
            for k in keys_to_remove:
                del self.relationships[k]
            if self.environment is not None:
                self.environment.remove_agent(member_id)

    def get_relationship(self, agent_a: str, agent_b: str) -> Optional[Relationship]:
        return self.relationships.get((agent_a, agent_b))

    def get_member_relationships(self, member_id: str) -> list[Relationship]:
        return [
            rel for (a, _), rel in self.relationships.items()
            if a == member_id
        ]

    def step(self) -> None:
        for rel in set(self.relationships.values()):
            rel.update_dynamics()

        total_income = sum(
            m.get_state_value("income", 0.0)
            for m in self.members.values()
        )
        self.set_state_value("total_income", total_income)

    def get_state(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": type(self).__name__,
            "attributes": dict(self.attributes),
            "state": dict(self.state),
            "alive": self.alive,
            "member_count": len(self.members),
        }

    def __repr__(self) -> str:
        return f"Household({self.get_attribute('name')}, members={len(self.members)})"
