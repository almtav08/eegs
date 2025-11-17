from typing import Dict, List, Any

import json
from llm_client import LLMClient


class KGRAGAgent:
    """
    Agent that combines Knowledge Graph (KG) retrieval with
    Retrieval-Augmented Generation (RAG) to provide recommendations
    with explanations.
    """

    def __init__(
        self,
        forward_graph: str,
        remedial_graph: str,
        content_data: str,
        llm_client: LLMClient,
    ):
        with open(forward_graph, "r") as f:
            self.forward_graph = json.load(f)

        with open(remedial_graph, "r") as f:
            self.remedial_graph = json.load(f)

        with open(content_data, "r") as f:
            self.content_data = json.load(f)

        self.llm_client = llm_client

    def _retrieve_candidates(self, last_visited_item: str) -> List[Dict[str, str]]:
        """
        (KG-Retrieval) Retrieve candidates from knowledge graphs.
        """
        print(
            f"[KG-RAG] Step 1: Retrieving candidates from graphs for item '{last_visited_item}'"
        )
        next_steps_ids = self.forward_graph.get(last_visited_item, [])
        remedial_steps_ids = self.remedial_graph.get(last_visited_item, [])

        candidates = []
        for id in next_steps_ids:
            candidates.append({"id": id, "type": "next_step"})
        for id in remedial_steps_ids:
            candidates.append({"id": id, "type": "remedial"})

        print(f"[KG-RAG] -> Found {len(candidates)} candidates: {candidates}")
        return candidates

    def _generate_prompt(
        self, last_visited_item: Dict[str, str], candidates: List[Dict[str, str]]
    ):
        """
        Generates the prompt for the LLM based on the last item content and augmented candidates.
        """
        prompt = f"""
        ### ROLE
        You are an expert Educational Tutor and curriculum designer.

        ### CONTEXT: COMPLETED RESOURCE
        The student has just finished this resource:
        - **ID:** "{last_visited_item["id"]}"
        - **Name:** "{last_visited_item["name"]}"
        - **Summary:** "{last_visited_item["summary"]}"
        - **Keywords:** "{", ".join(last_visited_item["keywords"])}"

        ### CANDIDATE RESOURCES TO EXPLAIN
        You must generate one explanation for each of the following candidates:
        """
        # Add candidates to the prompt
        for i, cand in enumerate(candidates):
            prompt += f"""

        **Candidate {i+1}:**
        - **ID:** "{cand["item_id"]}"
        - **Type:** "{cand["type"]}" (e.g., "next_step" or "remedial")
        - **Name:** "{cand["name"]}"
        - **Summary:** "{cand["summary"]}"
        - **Keywords:** "{", ".join(cand["keywords"])}"
        """

        # Task
        prompt += """
        ### TASK
        Your goal is to generate a brief, encouraging, one-sentence explanation for *why* a student should look at each "Candidate Resource" next. You must generate 
        this explanation by logically connecting the "Completed Resource" (the context) to each candidate.

        ### RULES
        1.  **Connect the Concepts:** The explanation *must* link a concept from the "Completed Resource" (e.g., from its keywords or summary) to a concept in the "Candidate Resource".
        2.  **Use the "Type":**
            - If `Type` is "next_step", frame it as a logical progression (e.g., "Now that you understand [Concept A], the next logical step is to explore [Concept B]...").
            - If `Type` is "remedial", frame it as a helpful review (e.g., "To help solidify your understanding of [Concept A], it might be useful to review [Concept C]...").
        3.  **Language:** English.
        4.  **Tone:** Supportive, clear, and concise.

        ### OUTPUT FORMAT
        Respond ONLY with a valid JSON object mapping each Candidate ID to its 
        generated explanation. Do not include markdown backticks or any other text 
        outside the JSON.

        Example:
        {
        "{candidate_1_id}": "Your generated explanation for candidate 1...",
        "{candidate_2_id}": "Your generated explanation for candidate 2..."
        }
        """
        return prompt

    def _augment_and_generate(
        self, last_visited_item: str, candidates: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        (RAG) Augments content and generates explanations.
        """
        print(
            f"[KG-RAG] Step 2: Augmenting candidates with content (summaries/keywords)."
        )

        # (R) Augment: Obtain base item content
        last_item_content = self.content_data.get(last_visited_item)
        if not last_item_content:
            print(
                f"[KG-RAG] Error: No content found for base item '{last_visited_item}'"
            )
            return []

        # (R) Augment: Obtain content for candidates
        augmented_candidates = []
        for cand in candidates:
            content = self.content_data.get(str(cand["id"]))
            if content:
                augmented_candidates.append(
                    {
                        "item_id": cand["id"],
                        "name": content["name"],
                        "type": cand["type"],
                        "summary": str(content["summary"]).replace("\n", " "),
                        "keywords": content["keywords"],
                    }
                )

        if not augmented_candidates:
            print(f"[KG-RAG] No content found for any candidates.")
            return []

        print(f"[KG-RAG] Step 3: Generating explanations with LLM.")

        # (G) Generate: Call LLM with all context
        last_visited_content = {
            "id": last_visited_item,
            "name": self.content_data[last_visited_item]["name"],
            "summary": str(self.content_data[last_visited_item]["summary"]).replace("\n", " "),
            "keywords": self.content_data[last_visited_item]["keywords"],
        }

        prompt = self._generate_prompt(last_visited_content, augmented_candidates)
        explanations_map = self.llm_client.generate_content(prompt)

        # 4. Combine explanations
        final_recommendations = []
        for cand in augmented_candidates:
            item_id = str(cand["item_id"])
            if item_id in explanations_map:
                final_recommendations.append(
                    {
                        "item_id": item_id,
                        "type": cand["type"],
                        "explanation": explanations_map[item_id],
                        "name": cand["name"],
                        "summary": cand["summary"],
                        "keywords": cand["keywords"],
                    }
                )

        return final_recommendations

    def get_recommendations(
        self, last_visited_item: str, max_recommendations: int = 3
    ) -> List[Dict[str, str]]:

        all_candidates = self._retrieve_candidates(last_visited_item)

        top_candidates = all_candidates[:max_recommendations]

        if not top_candidates:
            print(
                f"[KG-RAG] No recommendations found for '{last_visited_item}'."
            )
            return []

        explained_recommendations = self._augment_and_generate(
            last_visited_item, top_candidates
        )

        return explained_recommendations
