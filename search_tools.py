import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Your Tavily API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Tavily Search
search = TavilySearchResults(api_key=TAVILY_API_KEY)

def search_documents(query):
    """
    Function to perform search using Tavily's API.
    """
    try:
        results = search.run(query)  # Pass the query to the search
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        return []
