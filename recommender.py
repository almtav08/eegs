import json
from typing import List, Literal, Optional, Dict
from kgrag_agent import KGRAGAgent
from llm_client import LLMClient
from nudge_agent import NudgeAgent


class RecommenderAgent:
    def __init__(
        self, nudge_agent: NudgeAgent, kgrag_agent: KGRAGAgent, llm_client: LLMClient
    ):
        self.nudge_agent = nudge_agent
        self.kgrag_agent = kgrag_agent
        self.llm_client = llm_client

    def _build_llm_prompt(
        self,
        content_recs: Optional[List[Dict]],
        nudge: Optional[Dict],
        last_item_name: str,
    ) -> str:
        """
        Constructs the prompt for the LLM
        """

        # Convert to string
        if content_recs:
            content_json = json.dumps(content_recs, indent=2, ensure_ascii=False)
        else:
            content_json = "[]"

        if nudge:
            nudge_json = json.dumps(nudge, indent=2, ensure_ascii=False)
        else:
            nudge_json = "null"

        prompt = f"""
            ### ROLE
            You are an expert conversational tutor for a course. Your personality is:
            - **Supportive and encouraging** (like a friendly coach).
            - **Clear and concise** (never overwhelming).
            - **Action-oriented** (always proposing a next step).

            ### TASK
            Your goal is to synthesize structured JSON data into a single, natural, conversational message for the student.

            ### OUTPUT LANGUAGE
            The final response *MUST* be in **Spanish**.

            ### INPUTS
            You will receive two JSON inputs and the name of the completed resource. One or both JSON may be null/empty. If so, ignore that input.

            ### RULES
            Always start by congratulating them on completing the resource.
            Use a smooth transition to connect the different parts of the response.
            Always end with a positive motivational statement to encourage the student to continue learning.

            **1. Completed Resource Name:**
            "{last_item_name}"
            **2. Content Recommendations (from ContentKGRAgent):**
            A list of "what" to study next.
            ```json
            {content_json}
            ```
            **3. Behavioral Nudge (from NudgeAgent):**
            A single object describing "how" to study better.
            ```json
            {nudge_json}
            ```

            ### OUTPUT FORMAT
            Respond ONLY with the JSON object. Do not include any other text, pre-amble, or markdown backticks.

            {{
            "message": "string"
            }}
            """
        return prompt

    def _call_llm(
        self,
        content_recs: Optional[List[Dict]],
        nudge: Optional[Dict],
        last_item_name: str,
    ) -> Dict[str, str]:
        """
        Prepares and calls the LLM to generate the response.
        """

        prompt = self._build_llm_prompt(content_recs, nudge, last_item_name)

        print(f"[Recommender Log] Calling LLM for content recommendations")

        return self.llm_client.generate_content(prompt)

    def recommend(
        self,
        user_profile: dict,
        last_visited_item: str,
        max_recommendations: int,
        last_item_name: str,
        generate: Literal["both", "nudge", "content"] = "both",
    ) -> Dict[str, str]:
        if generate == "nudge" or generate == "both":
            nudge_response = self.nudge_agent.generate_nudge(user_profile)
        else:
            nudge_response = None

        if generate == "content" or generate == "both":
            kgrag_response = self.kgrag_agent.get_recommendations(
                last_visited_item, max_recommendations
            )
        else:
            kgrag_response = None

        llm_response = self._call_llm(
            content_recs=kgrag_response,
            nudge=nudge_response,
            last_item_name=last_item_name,
        )

        return llm_response
