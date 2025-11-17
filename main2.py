import json
from kgrag_agent import KGRAGAgent
from llm_client import LLMClient
from nudge_agent import NudgeAgent


if __name__ == "__main__":

    with open("models/ideal_profile.json", "r") as f:
        IDEAL_PROFILE = json.load(f)

    with open("models/behavior.json", "r") as f:
        METRIC_BEHAVIOR = json.load(f)

    PROJECT_ID = "hybrid-ridge-476409-c2"
    REGION = "us-central1"

    llm_client = LLMClient(
        project_id=PROJECT_ID,
        location=REGION,
    )

    sample_user_profile = IDEAL_PROFILE.copy()

    sample_user_profile["weekend_ratio"] = 0.05
    sample_user_profile["procrastination_ratio"] = 0.35
    sample_user_profile["avg_inactive_days"] = 1.0

    nudge_agent = NudgeAgent(
        ideal_profile=IDEAL_PROFILE,
        metric_behavior=METRIC_BEHAVIOR,
        llm_client=llm_client,
        tolerance_good=0.8,
        tolerance_bad=1.2,
    )

    krag_agent = KGRAGAgent(
        forward_graph="database/forward_graph.json",
        remedial_graph="database/remedial_graph.json",
        content_data="database/content_data.json",
        llm_client=llm_client,
    )

    last_item = "50"

    print(f"--- Executing KG-RAG for item: '{last_item}' ---")
    recs = krag_agent.get_recommendations(last_item, max_recommendations=3)

    print("\n--- Final Output (JSON) ---")
    print(json.dumps(recs, indent=2, ensure_ascii=False))

    print("--- Generating Nudge (Test 1: User with high procrastination) ---")
    nudge = nudge_agent.generate_nudge(sample_user_profile)

    if nudge:
        print("\n--- Nudge JSON Generated ---")
        print(json.dumps(nudge, indent=2))

    print("\n--- Generating Nudge (Test 2: User within tolerance) ---")
    good_user_profile = IDEAL_PROFILE.copy()
    good_user_profile["weekend_ratio"] = 0.20
    good_user_profile["procrastination_ratio"] = 0.06

    nudge_good = nudge_agent.generate_nudge(good_user_profile)

    if nudge_good:
        print("\n--- Nudge JSON Generated ---")
        print(json.dumps(nudge_good, indent=2))

    llm_client.stop()
