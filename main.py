import json
from kgrag_agent import KGRAGAgent
from llm_client import LLMClient
from nudge_agent import NudgeAgent
from recommender import RecommenderAgent


if __name__ == "__main__":

    with open("models/ideal_profile.json", "r") as f:
        IDEAL_PROFILE = json.load(f)

    with open("models/behavior.json", "r") as f:
        METRIC_BEHAVIOR = json.load(f)

    PROJECT_ID = ""
    REGION = "us-central1"

    ideal_user_profile = IDEAL_PROFILE.copy()

    llm_client = LLMClient(
        project_id=PROJECT_ID,
        location=REGION,
    )

    nudge_agent = NudgeAgent(
        ideal_profile=IDEAL_PROFILE,
        metric_behavior=METRIC_BEHAVIOR,
        llm_client=llm_client,
        tolerance_good=0.8,
        tolerance_bad=1.1,
    )

    krag_agent = KGRAGAgent(
        forward_graph="database/forward_graph.json",
        remedial_graph="database/remedial_graph.json",
        content_data="database/content_data.json",
        llm_client=llm_client,
    )

    recommender = RecommenderAgent(
        nudge_agent=nudge_agent,
        kgrag_agent=krag_agent,
        llm_client=llm_client,
    )

    results = {}

    print(f"\n--- Generating Recommendation for Scenario 1 - The ideal student ---")
    
    last_item = "54"
    last_item_name = "C6: El modo framebuffer de la NDS"
    
    user_profile = ideal_user_profile.copy()
    
    recommendation = recommender.recommend(
        user_profile=user_profile,
        last_visited_item=last_item,
        max_recommendations=3,
        last_item_name=last_item_name,
        generate="both",
    )
    
    results['scenario_1'] = recommendation["message"]
    
    print(f"\n--- Generating Recommendation for Scenario 2 - The Procrastinator ---")
    
    last_item = "54"
    last_item_name = "C6: El modo framebuffer de la NDS"
    
    user_profile = ideal_user_profile.copy()
    user_profile["procrastination_ratio"] = 1.5
    
    recommendation = recommender.recommend(
        user_profile=user_profile,
        last_visited_item=last_item,
        max_recommendations=3,
        last_item_name=last_item_name,
        generate="both",
    )
    
    results['scenario_2'] = recommendation["message"]
    
    print(f"\n--- Generating Recommendation for Scenario 3 - The confused ---")

    last_item = "55"
    last_item_name = "Notas: Insertar imágenes en modo framebuffer de la NDS"

    user_profile = ideal_user_profile.copy()
    user_profile["backward_prereq_ratio"] = 0.1

    recommendation = recommender.recommend(
        user_profile=user_profile,
        last_visited_item=last_item,
        max_recommendations=3,
        last_item_name=last_item_name,
        generate="both",
    )

    results["scenario_3"] = recommendation["message"]
    
    print(f"\n--- Generating Recommendation for Scenario 4 - The weekend player ---")
    
    last_item = "120"
    last_item_name = "Examen: Juego NDS"
    
    user_profile = ideal_user_profile.copy()
    user_profile["weekend_ratio"] = 0.3
    user_profile["regularity_ratio"] = 0.15
    
    recommendation = recommender.recommend(
        user_profile=user_profile,
        last_visited_item=last_item,
        max_recommendations=3,
        last_item_name=last_item_name,
        generate="both",
    )
    
    results["scenario_4"] = recommendation["message"]
    
    print(f"\n--- Generating Recommendation for Scenario 5 - The positive streak ---")
    
    last_item = "54"
    last_item_name = "C6: El modo framebuffer de la NDS"
    
    user_profile = ideal_user_profile.copy()
    user_profile["max_consecutive_days"] = 12
    
    recommendation = recommender.recommend(
        user_profile=user_profile,
        last_visited_item=last_item,
        max_recommendations=3,
        last_item_name=last_item_name,
        generate="both",
    )
    
    results["scenario_5"] = recommendation["message"]
    
    print(f"\n--- Generating Recommendation for Scenario 6 - The zapper ---")
    
    last_item = "54"
    last_item_name = "C6: El modo framebuffer de la NDS"
    
    user_profile = ideal_user_profile.copy()
    user_profile["zapping_ratio"] = 0.95
    user_profile["avg_delta_t_sec"] = 75
    
    recommendation = recommender.recommend(
        user_profile=user_profile,
        last_visited_item=last_item,
        max_recommendations=3,
        last_item_name=last_item_name,
        generate="both",
    )
    
    results["scenario_6"] = recommendation["message"]
    
    llm_client.stop()
    
    with open("recommendations.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
