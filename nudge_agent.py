from typing import Any, Dict, Optional
from google import genai
from google.genai.types import GenerateContentConfig

from llm_client import LLMClient


class NudgeAgent:
    """
    Agent responsible for generating nudges based on user performance metrics
    compared to an ideal profile. Utilizes a large language model (LLM) to create
    personalized nudge structures in JSON format.
    """

    def __init__(
        self,
        ideal_profile: Dict[str, float],
        metric_behavior: Dict[str, Dict[str, Any]],
        llm_client: LLMClient,
        tolerance_good: float = 0.8,
        tolerance_bad: float = 1.1,
    ):
        """
        Initializes the agent with the central knowledge.

        Args:
            ideal_profile: The metric profile of the "passing student".
            metric_behavior: The dictionary that defines the direction and glossary of each metric.
            llm_client: An instance of an LLM client (e.g. OpenAI, Gemini) that has a method for generating the nudge.
            tolerance_good: Factor for metrics to maximize (e.g. 0.8 = 80%).
            tolerance_bad: Factor for metrics to minimize (e.g. 1.1 = 110%).
        """
        self.ideal_profile = ideal_profile
        self.metric_behavior = metric_behavior
        self.llm_client = llm_client
        self.tolerance_good = tolerance_good
        self.tolerance_bad = tolerance_bad
        self.epsilon = 1e-6  # To avoid division by zero

    def _calculate_deviation_scores(
        self, user_profile: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculates the proportional deviation score for all metrics.
        A higher score means a worse and more urgent deviation.
        """
        deviation_scores = {}
        improvement_scores = {} # Para almacenar las puntuaciones "buenas"
        has_any_deviation = False # Flag para eficiencia

        for metric, behavior in self.metric_behavior.items():

            ideal_val = self.ideal_profile.get(metric)
            user_val = user_profile.get(metric)

            # If any data is missing, we cannot calculate the deviation
            if ideal_val is None or user_val is None:
                deviation_scores[metric] = 0.0
                improvement_scores[metric] = 0.0
                continue

            direction = behavior.get("direction", 0)

            if direction == 1:
                # --- Metric to MAXIMIZE ---
                # The threshold is *below* the ideal
                if ideal_val >= 0:
                    # Normal: 100 (ideal) -> 80 (threshold)
                    threshold = ideal_val * self.tolerance_good
                else:
                    # Negativo: -100 (ideal) -> -120 (threshold)
                    # (Usamos la tolerancia "externa" para ir MÁS negativo)
                    threshold = ideal_val * self.tolerance_bad

                if user_val < threshold:
                    # --- DESVIACIÓN (Mal) ---
                    # Qué tan lejos está el usuario del umbral (proporcional)
                    score = (threshold - user_val) / (threshold + self.epsilon)
                    deviation_scores[metric] = score
                    improvement_scores[metric] = 0.0
                    has_any_deviation = True # Marcamos que encontramos una desviación
                else:
                    # --- SIN DESVIACIÓN (Bien) ---
                    deviation_scores[metric] = 0.0 
                    # Calculamos la *mejora* potencial sobre el ideal
                    improvement = (user_val - ideal_val) / (ideal_val + self.epsilon)
                    improvement_scores[metric] = max(0.0, improvement) # Solo > 0

            elif direction == -1:
                # --- Metric to MINIMIZE ---
                # The threshold is *above* the ideal
                if ideal_val >= 0:
                    # Normal: 100 (ideal) -> 120 (threshold)
                    threshold = ideal_val * self.tolerance_bad
                else:
                    # Negativo: -100 (ideal) -> -80 (threshold)
                    # (Usamos la tolerancia "interna" para ir MÁS cerca de 0)
                    threshold = ideal_val * self.tolerance_good
                if user_val > threshold:
                    # --- DESVIACIÓN (Mal) ---
                    # Qué tan lejos está el usuario del umbral (proporcional)
                    score = (user_val - threshold) / (threshold + self.epsilon)
                    deviation_scores[metric] = score
                    improvement_scores[metric] = 0.0
                    has_any_deviation = True # Marcamos que encontramos una desviación
                else:
                    # --- SIN DESVIACIÓN (Bien) ---
                    deviation_scores[metric] = 0.0
                    # Calculamos la *mejora* potencial sobre el ideal
                    improvement = (ideal_val - user_val) / (ideal_val + self.epsilon)
                    improvement_scores[metric] = max(0.0, improvement) # Solo > 0

            else:
                deviation_scores[metric] = 0.0  # Metric not considered
                improvement_scores[metric] = 0.0

        # --- Comprobación Final ---
        # Si encontramos *alguna* desviación mala, devolvemos esas puntuaciones.
        if has_any_deviation:
            print("Deviations found.")
            return deviation_scores
        else:
            # Si no hubo desviaciones, devolvemos las puntuaciones de mejora.
            return improvement_scores

    def _find_worst_metric(self, deviation_scores: Dict[str, float]) -> Optional[str]:
        """Finds the metric with the highest deviation score."""
        if not deviation_scores:
            return None

        # Find the metric with the maximum deviation score
        worst_metric = max(deviation_scores, key=deviation_scores.get)

        # If the worst score is 0, it means there are no deviations
        if deviation_scores[worst_metric] == 0.0:
            return None

        return worst_metric

    def _build_llm_prompt(self, metric: str, user_val: float, ideal_val: float, glossary: str, direction: int) -> str:
        """
        Constructs the prompt for the LLM
        """

        if direction == 1:
            direction_text = "Higher is better. The student's value is too low and should be increased."
        else:
            direction_text = "Lower is better. The student's value is too high and should be decreased."

        prompt = f"""
            ### ROLE
            You are an expert Educational Psychologist and motivational coach. Your goal is to 
            help a student succeed by providing a supportive, data-driven "nudge".

            ### TASK
            Generate a JSON object for a single behavioral nudge. The nudge must be 
            supportive, empathetic, and actionable. If the student's metric improves
            the desired direction, acknowledge the progress.

            ### CONTEXT
            You must base your nudge on the following metric, which has been identified as
            the highest priority for this student. 

            * **Metric to Address:** "{metric}"
            * **Metric Definition:** "{glossary}"
            * **Desired Direction:** "{direction_text}"
            * **Student's Value:** {user_val:.4f}
            * **Ideal Value (Pass Group):** {ideal_val:.4f}

            ### RULES
            1.  **Tone:** Must be "supportive". NEVER be critical, shaming, or alarming.
            2.  **Observation:** A neutral, data-informed statement. (e.g., "I noticed..."). Do not include the data value.
            3.  **Insight:** The "why". Explain *why* this metric matters, connecting it positively to success. (e.g., "Students who pass find that...")
            4.  **Suggestion:** A single, concrete, and *small* actionable step. Make it feel easy to do. (e.g., "How about trying...")
            5.  **Call to Action:** An open-ended, low-pressure question that invites collaboration. (e.g., "Would you like to...?", "What do you think...?")

            ### OUTPUT FORMAT
            Respond ONLY with the JSON object. Do not include any other text, pre-amble, or markdown backticks.

            {{
            "metric_name": "{metric}",
            "tone": "string",
            "observation": "string",
            "insight": "string",
            "suggestion": "string",
            "call_to_action": "string"
            }}
            """
        return prompt

    def _call_llm_for_nudge(
        self, metric: str, user_val: float, ideal_val: float
    ) -> Dict[str, str]:
        """
        Prepares and calls the LLM to generate the nudge structure.
        """
        behavior = self.metric_behavior[metric]
        glossary = behavior["glossary"]
        direction = behavior["direction"]

        prompt = self._build_llm_prompt(metric, user_val, ideal_val, glossary, direction)

        print(f"[NudgeAgent Log] Calling LLM for metric: {metric}")

        return self.llm_client.generate_content(prompt)

    def generate_nudge(
        self, user_profile: Dict[str, float]
    ) -> Optional[Dict[str, str]]:
        """
        Creates a nudge structure if the user is deviating significantly from the ideal profile.

        Args:
            user_profile: The user's current metric profile.

        Returns:
            A dictionary (JSON) with the nudge structure, or None if no nudge is needed.
        """
        deviation_scores = self._calculate_deviation_scores(user_profile)

        metric_to_nudge = self._find_worst_metric(deviation_scores)

        if metric_to_nudge is None:
            print("[NudgeAgent Log] No nudge needed. User is within tolerance.")
            return None

        user_val = user_profile[metric_to_nudge]
        ideal_val = self.ideal_profile[metric_to_nudge]

        nudge_structure = self._call_llm_for_nudge(
            metric=metric_to_nudge, user_val=user_val, ideal_val=ideal_val
        )

        return nudge_structure
