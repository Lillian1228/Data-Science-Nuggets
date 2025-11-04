import os
from dotenv import load_dotenv
from tools.executor import ToolExecutor
from tools.search_tool import search
from agents.react_agent import ReActAgent
from hello_agents_llm import HelloAgentsLLM

# Load environment variables
load_dotenv()

def main():
    # Initialize the tool executor
    tool_executor = ToolExecutor()

    # Register SerpAPI search tool
    search_description = (
        "A web search engine tool. Use this when you need real-time or factual information "
        "that may not be in your internal knowledge."
    )
    tool_executor.registerTool("Search", search_description, search)

    # Initialize LLM and ReAct Agent
    llm_client = HelloAgentsLLM()
    agent = ReActAgent(llm_client=llm_client, tool_executor=tool_executor, max_steps=5)

    # Ask a test question
    question = "What is NVIDIA's latest GPU model?"
    print(f"\n🤖 Question: {question}\n")
    final_answer = agent.run(question)
    print(f"\n✅ Agent Finished: {final_answer}\n")

if __name__ == "__main__":
    main()
