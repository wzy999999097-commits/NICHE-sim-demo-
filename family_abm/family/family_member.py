from __future__ import annotations
from typing import Any, Optional
import math
import random
from ..core.agent import Agent

PERSONALITY_DIMENSIONS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# ── Default simulation parameters ──────────────────────────────────────────
DEFAULT_PARAMS: dict[str, float] = {
    "education_rate": 0.008,
    "income_base": 0.10,
    "income_edu_boost": 0.40,
    "income_age_peak": 45,
    "income_age_spread": 18,
    "health_decay_base": 0.002,
    "health_decay_age": 0.00015,
    "health_edu_protection": 0.30,
    "stress_base": 0.03,
    "stress_work_add": 0.02,
    "stress_decay": 0.06,
    "stress_neuro_sensitivity": 0.30,
    "happiness_baseline": 0.35,
    "happiness_health_weight": 0.20,
    "happiness_income_weight": 0.25,
    "happiness_edu_weight": 0.10,
    "happiness_stress_penalty": 0.30,
    "happiness_recovery": 0.08,
    "randomness": 0.02,
}


class FamilyMember(Agent):
    def __init__(
        self,
        name: str = "",
        age: float = 25.0,
        gender: str = "other",
        personality: Optional[dict[str, float]] = None,
        role_name: str = "adult",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_attribute("name", name)
        self.set_attribute("age", age)
        self.set_attribute("gender", gender)
        self.set_attribute("personality", personality or {
            k: max(0.0, min(1.0, random.gauss(0.5, 0.15))) for k in PERSONALITY_DIMENSIONS
        })
        self.set_attribute("role", role_name)

        self.set_state_value("health", max(0.3, min(1.0, 1.0 - age * 0.003 + random.gauss(0, 0.05))))
        self.set_state_value("happiness", random.uniform(0.4, 0.7))
        self.set_state_value("stress", random.uniform(0.15, 0.4))
        self.set_state_value("energy", random.uniform(0.6, 1.0))
        self.set_state_value("education", max(0.0, min(1.0, age * 0.012 + random.gauss(0, 0.05))))
        self.set_state_value("income", 0.0)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _p(self, key: str, default: float = 0.0) -> float:
        """Read a parameter from environment config, with fallback."""
        if self.environment is not None:
            env_params = getattr(self.environment, "params", None) or {}
            return float(env_params.get(key, DEFAULT_PARAMS.get(key, default)))
        return float(DEFAULT_PARAMS.get(key, default))

    def _role_at_age(self, age: float) -> str:
        if age < 6:     return "preschool"
        elif age < 18:   return "child"
        elif age < 22:   return "student"
        elif age < 65:   return "adult"
        else:           return "elder"

    def age_increment(self, years: float = 1 / 12) -> None:
        new_age = self.get_attribute("age") + years
        self.set_attribute("age", new_age)
        self.set_attribute("role", self._role_at_age(new_age))

    # ── Main step ────────────────────────────────────────────────────────

    def step(self) -> None:
        age = self.get_attribute("age")
        role = self.get_attribute("role")
        personality = self.get_attribute("personality")
        rng = random.gauss

        dt_months = self._p("dt_months", 1.0)
        noise = self._p("randomness", 0.02)

        # ── Education (peaks early, plateaus) ──
        edu = self.get_state_value("education")
        edu_rate = self._p("education_rate", 0.008)
        age_edu_factor = max(0.05, 1.0 - age / 60)
        edu_gain = edu_rate * age_edu_factor * (1 - edu)
        if age < 6:
            edu_gain *= 0.3
        edu += (edu_gain + rng(0, noise * 0.3)) * dt_months
        self.set_state_value("education", max(0.0, min(1.0, edu)))

        # ── Income (education × age curve × role) ──
        base = self._p("income_base", 0.10)
        edu_boost = self._p("income_edu_boost", 0.4) * edu
        peak = self._p("income_age_peak", 45)
        spread = self._p("income_age_spread", 18)
        age_factor = math.exp(-((age - peak) ** 2) / (2 * spread ** 2))
        role_mul = {"preschool": 0.0, "child": 0.0, "student": 0.15,
                    "adult": 1.0, "elder": 0.35}.get(role, 0.2)
        income = base * (1 + edu_boost) * age_factor * role_mul
        income += rng(0, noise * base)
        self.set_state_value("income", max(0.0, income))

        # ── Health (decays with age, buffered by education) ──
        hp = self.get_state_value("health")
        base_decay = self._p("health_decay_base", 0.002)
        age_decay = self._p("health_decay_age", 0.00015) * age
        edu_protect = self._p("health_edu_protection", 0.30) * edu
        hp_change = -(base_decay + age_decay * (1 - edu_protect)) * dt_months
        hp_change += rng(0, noise * 0.3)
        self.set_state_value("health", max(0.01, min(1.0, hp + hp_change)))

        # ── Stress (base + role burden, decay) ──
        st = self.get_state_value("stress")
        st_base = self._p("stress_base", 0.03)
        st_work = self._p("stress_work_add", 0.02) if role in ("adult", "student") else 0.005
        st_decay = self._p("stress_decay", 0.06)
        neuro = personality.get("neuroticism", 0.5)
        neuro_sens = self._p("stress_neuro_sensitivity", 0.30)
        st_change = (st_base + st_work) * (1 + neuro_sens * neuro) - st_decay * st
        st_change += rng(0, noise * 0.5) * (st + 0.1)
        self.set_state_value("stress", max(0.0, min(1.0, st + st_change)))

        # ── Happiness (target-based, mean-reverting) ──
        ha = self.get_state_value("happiness")
        h_base = self._p("happiness_baseline", 0.35)
        h_health = self._p("happiness_health_weight", 0.20) * hp
        h_income = self._p("happiness_income_weight", 0.25) * min(1.0, income * 3)
        h_edu = self._p("happiness_edu_weight", 0.10) * edu
        h_stress_pen = self._p("happiness_stress_penalty", 0.30) * st
        target = max(0.0, min(1.0, h_base + h_health + h_income + h_edu - h_stress_pen))
        recovery = self._p("happiness_recovery", 0.08)
        ha += (target - ha) * recovery + rng(0, noise * 0.5)
        self.set_state_value("happiness", max(0.0, min(1.0, ha)))

        # ── Energy (simple cycle) ──
        en = self.get_state_value("energy")
        en = en * 0.92 + 0.06 + rng(0, noise * 0.2)
        self.set_state_value("energy", max(0.0, min(1.0, en)))

        # ── Age ──
        self.age_increment(years=dt_months / 12)

    def __repr__(self) -> str:
        return f"FamilyMember({self.get_attribute('name')}, age={self.get_attribute('age'):.1f}, role={self.get_attribute('role')})"
