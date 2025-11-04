from openai import OpenAI
from tools import available_tools
from agent_system_prompt import AGENT_SYSTEM_PROMPT
import os, re
from dotenv import load_dotenv

load_dotenv()

class OpenAICompatibleClient:
    """
    A client for calling any LLM service that is compatible with the OpenAI API interface.
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Call the LLM API to generate a response."""
        print("Calling the large language model...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("Language model responded successfully.")
            return answer
        except Exception as e:
            print(f"Error while calling the LLM API: {e}")
            return "Error: An issue occurred while calling the language model service."


# --- Set up environment and LLM ---
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_ID = os.getenv("MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = OpenAICompatibleClient(model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL)

# --- Run the reasoning loop ---
user_prompt = "Hi, please check the weather in Beijing today and recommend an attraction."
prompt_history = [f"User request: {user_prompt}"]

for i in range(5):
    print(f"--- Iteration {i+1} ---\n")
    
    # 3.1. Build the full conversation prompt
    full_prompt = "\n".join(prompt_history)
    
    # 3.2. Call the LLM to reason and decide the next action
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    print(f"Model output:\n{llm_output}\n")
    prompt_history.append(llm_output)
    
    # 3.3. Parse and execute the model’s chosen action
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        print("Parsing error: No Action found in model output.")
        break

    action_str = action_match.group(1).strip()

    # If the model decides the task is finished
    if action_str.startswith("finish"):
        final_answer = re.search(r'finish\(answer="(.*)"\)', action_str).group(1)
        print(f"Task completed. Final answer: {final_answer}")
        break
    
    # Extract tool name and arguments
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    # Execute the corresponding tool function
    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"Error: Undefined tool '{tool_name}'"

    # 3.4. Record the observation for the next reasoning step
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)
