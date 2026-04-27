"""
Simple example: Simulate a single family household over time.
"""
from family_abm import (
    Environment, Simulation, Scheduler,
    FamilyMember, Household, StateRecorder,
)


def main():
    env = Environment()

    household = Household(name="Smith Family")
    env.add_agent(household)

    father = FamilyMember(name="Dad", age=40, gender="male", role_name="parent")
    mother = FamilyMember(name="Mom", age=38, gender="female", role_name="parent")
    son = FamilyMember(name="Son", age=10, gender="male", role_name="child")
    daughter = FamilyMember(name="Daughter", age=8, gender="female", role_name="child")

    household.add_member(father)
    household.add_member(mother)
    household.add_member(son)
    household.add_member(daughter)

    recorder = StateRecorder(record_agents=True)
    sim = Simulation(env, scheduler=Scheduler("sequential"))
    sim.add_recorder(recorder)

    sim.run(steps=120)

    df = recorder.to_dataframe()
    print("Simulation finished. Recorded", len(df), "observations.")
    print("\nColumns:", list(df.columns))
    print("\nSample data:")
    print(df.head(10))

    print("\n--- Final states ---")
    for agent in env.get_agents():
        if isinstance(agent, FamilyMember):
            s = agent.get_state()
            print(f"  {s['attributes']['name']:>10} | "
                  f"age={s['attributes']['age']:.1f} | "
                  f"role={s['attributes']['role']:>6} | "
                  f"happiness={s['state']['happiness']:.2f} | "
                  f"stress={s['state']['stress']:.2f} | "
                  f"health={s['state']['health']:.2f}")


if __name__ == "__main__":
    main()
