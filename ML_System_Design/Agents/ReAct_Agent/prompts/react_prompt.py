REACT_PROMPT_TEMPLATE = """
Note: You are an intelligent assistant capable of invoking external tools.

Available tools:
{tools}

Please strictly follow this response format:

Thought: Your reasoning process to analyze the problem, decompose tasks, and plan the next action.
Action: The action you decide to take, and it must be one of the following formats:
- `{{tool_name}}[{{tool_input}}]` — to call a tool.
- `Finish[final answer]` — when you believe you have the final answer.

Now, start solving the problem below:
Question: {question}
History: {history}
"""
