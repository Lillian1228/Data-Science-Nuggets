import os
from serpapi import SerpApiClient

def search(query: str) -> str:
    """
    A practical web search tool powered by SerpAPI.
    It intelligently parses results and prefers returning direct answers
    or knowledge graph info when available.
    """
    print(f"🔍 Running [SerpAPI] web search: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SERPAPI_API_KEY is not set in the .env file."

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "us",      # country
            "hl": "en",      # language
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # Prefer the most direct answers first
        if "answer_box_list" in results and results["answer_box_list"]:
            # Some responses return a list of concise answers
            return "\n".join(map(str, results["answer_box_list"]))

        if "answer_box" in results and isinstance(results["answer_box"], dict):
            abox = results["answer_box"]
            if "answer" in abox and abox["answer"]:
                return str(abox["answer"])
            if "snippet" in abox and abox["snippet"]:
                return str(abox["snippet"])

        if "knowledge_graph" in results and isinstance(results["knowledge_graph"], dict):
            kg = results["knowledge_graph"]
            if "description" in kg and kg["description"]:
                return str(kg["description"])

        if "organic_results" in results and results["organic_results"]:
            # If no direct answer, return summaries of the top 3 organic results
            snippets = []
            for i, res in enumerate(results["organic_results"][:3]):
                title = res.get("title", "")
                snippet = res.get("snippet", "")
                snippets.append(f"[{i+1}] {title}\n{snippet}")
            return "\n\n".join(snippets)

        return f"Sorry, no information found for '{query}'."

    except Exception as e:
        return f"Error during search: {e}"
