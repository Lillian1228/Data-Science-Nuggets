from openai import OpenAI
import os

class HelloAgentsLLM:
    """
    A simple OpenAI-compatible client used by the ReAct agent.
    Handles reasoning (Thought + Action) steps via chat completions.
    """

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        # Use environment variables by default
        self.model = model or os.getenv("MODEL_ID")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("BASE_URL")
        )

    def think(self, messages):
        """
        Sends a message sequence to the model and returns the text output.
        The messages parameter must be a list of dictionaries with roles and content.
        Example:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"}
        ]
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return None
