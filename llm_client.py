import json
import argparse
import sys
from google import genai
from google.genai.types import GenerateContentConfig


class LLMClient:
    """
    Class for generating content using the Vertex AI (Gemini) API.
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

    def generate_content(self, prompt: str) -> dict:
        """
        Generates content based on the provided prompt.
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