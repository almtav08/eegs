import json
import argparse
import sys
from google import genai
from google.genai.types import GenerateContentConfig


class DocumentGenerator:
    """
    Class for generating document summaries and keywords
    using the Vertex AI (Gemini) API.
    """

    def __init__(
        self, project_id: str, location: str, model_name: str = "gemini-2.5-flash"
    ):
        """
        Initialize the Vertex AI client and the model.

        Args:
            project_id (str): Your Google Cloud project ID.
            location (str): The region where you will use Vertex AI (e.g. "us-central1").
            model_name (str): The name of the model to use.
        """
        try:
            self.client = genai.Client(
                vertexai=True, project=project_id, location=location
            )
            self.model = model_name

            # Configuration to force JSON output
            self.json_config = GenerateContentConfig(
                response_mime_type="application/json",
            )

        except Exception as e:
            print(f"Error initializing Vertex AI: {e}", file=sys.stderr)
            sys.exit(1)

    def stop(self):
        """Cleans up resources if needed."""
        self.client.close()

    def _get_prompt_base(self) -> str:
        """Returns the base instructions for the prompt."""
        return """
        You are: An Instructional Designer and educational metadata specialist.
        Your task: Generate high-quality metadata to catalog a learning resource based solely and exclusively on the provided title.

        STRICT RULES:
        1. **Mandatory Output Format**: Your response MUST be a single, valid JSON object. Do not include any explanatory text, greetings, notes, or markdown (like ```json) before or after the JSON block.
        2. **Required JSON Schema**: The JSON must strictly adhere to the following schema:
            {
            "summary": "string",
            "keywords": ["string"]
            }
        3. **Summary Quality**:
            Inference: The summary must infer the most likely learning objective (what the student will learn or be able to do) based only on the title.
            Language: Must be written in English.
            Content: Do not mention the title directly.
            Tone: Objective, impersonal (written in the third person), and information-dense.
            Length: Should capture the essence of the document in 2 or 3 paragraphs.
        4. **Keyword Quality**:
            Quantity: Generate between 5 and 7 keywords.
            Relevance: Must be specific, relevant, and high-value (avoid generic terms).
            Format: Compound terms are preferred (e.g., "Artificial Intelligence" instead of "Intelligence").
        """

    def generate_from_title(self, title: str) -> dict:
        """
        Generate a hypothetical summary and keywords based solely on a title.
        """
        prompt = f"""
        {self._get_prompt_base()}

        SPECIFIC TASK:
        Based *exclusively* on the following educational resource title, generate the summary and keywords.

        Title: "{title}"
        """
        return self._call_api(prompt)

    def _call_api(self, prompt: str) -> dict:
        """
        Private method to send the prompt to the Gemini model and handle the response.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self.json_config
            )

            # The JSON mode ensures that response.text is a valid JSON string
            return json.loads(response.text)

        except Exception as e:
            print(f"Error calling Vertex AI API: {e}", file=sys.stderr)
            return {"error": str(e)}


# -----------------------------------------------------------------
# --- EXECUTABLE BLOCK (for use as a script) ---
# -----------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Summary Generator with Vertex AI (Gemini).",
        epilog="Example: python doc_generator.py --project 'my-project' --title 'AI in Logistics'",
    )

    parser.add_argument(
        "-p", "--project", required=True, help="Google Cloud project ID."
    )
    parser.add_argument(
        "-l",
        "--location",
        default="us-central1",
        help="Vertex AI region (e.g. us-central1).",
    )
    parser.add_argument("-t", "--title", required=True, help="Document title.")

    args = parser.parse_args()

    print("Initializing generator...")
    generador = DocumentGenerator(project_id=args.project, location=args.location)

    result = None

    print(f"Generating from file content: {args.title}...")
    result = generador.generate_from_title(args.title)

    if result:
        print("\n--- RESULT ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
