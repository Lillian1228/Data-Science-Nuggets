import re
from prompts.react_prompt import REACT_PROMPT_TEMPLATE
from tools.executor import ToolExecutor

class ReActAgent:
    """
    A ReAct (Reason + Act) agent that performs step-by-step reasoning
    and executes actions using external tools until it reaches an answer.
    """

    def __init__(self, llm_client, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        Run the ReAct agent to answer a question through iterative reasoning and tool use.
        """
        self.history = []  # reset history for each new run
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- Step {current_step} ---")

            # 1. Format the prompt
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. Let the LLM "think" (generate the next Thought + Action)
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("Error: LLM did not return a valid response.")
                break

            # 3. Parse the LLM output
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"Thought: {thought}")

            if not action:
                print("Warning: Failed to parse a valid Action. Stopping process.")
                break

            # 4. Execute Action
            if action.startswith("Finish"):
                # Finish[...] → final answer
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 Final Answer: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print("Warning: Invalid Action format, skipping step.")
                continue

            print(f"🎬 Action: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"Error: tool '{tool_name}' not found."
            else:
                observation = tool_function(tool_input)

            # 5. Record the observation
            print(f"👀 Observation: {observation}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # End of loop
        print("Reached maximum number of steps. Process terminated.")
        return None

    # --- Helper methods ---
    def _parse_output(self, text: str):
        """Extract Thought and Action from LLM output."""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """Parse the Action string to extract tool name and input."""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None
