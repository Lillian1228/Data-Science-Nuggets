import requests
import os
from tavily import TavilyClient

def get_weather(city: str) -> str:
    """
    Retrieve real-time weather information by calling the wttr.in API.
    """
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        # Make the network request
        response = requests.get(url)
        response.raise_for_status()  # Check if status code is 200 (OK)
        data = response.json()
        
        # Extract current weather conditions
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        
        # Return a formatted, human-readable description
        return f"Current weather in {city}: {weather_desc}, temperature {temp_c}°C"
        
    except requests.exceptions.RequestException as e:
        # Handle network-related errors
        return f"Error: Network issue occurred while retrieving weather data - {e}"
    except (KeyError, IndexError) as e:
        # Handle data parsing issues
        return f"Error: Failed to parse weather data, possibly due to an invalid city name - {e}"

    
def get_attraction(city: str, weather: str) -> str:
    """
    Recommend tourist attractions based on the city and weather using the Tavily Search API.
    """
    # 1. Load API key from environment variable
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."

    # 2. Initialize the Tavily client
    tavily = TavilyClient(api_key=api_key)
    
    # 3. Construct a precise search query
    query = f"Best tourist attractions to visit in '{city}' under '{weather}' weather, with explanations"
    
    try:
        # 4. Call the API. Setting include_answer=True returns a synthesized answer.
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        # 5. If a summarized answer is available, return it directly
        if response.get("answer"):
            return response["answer"]
        
        # Otherwise, format individual search results
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
            return "Sorry, no relevant tourist attractions were found."

        return "Based on the search, here are the results:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"Error: An issue occurred while performing Tavily search - {e}"


# Collect tools in a dictionary
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

