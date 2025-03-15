from swarm import Agent, Swarm
from openai import OpenAI

# Initialize OpenAI client
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Placeholder; not used in this context
)

# Define the model to be used
model = "qwen2.5:7b"

# -------------------------------
# Define Direct Tools (Functions)
# -------------------------------

def meaning_of_life():
    """Returns the answer to the meaning of life"""
    return "651654116"

def web_search():
    """Performs a web search and returns results"""
    return "3 kittens"

def youtube_search():
    """Searches YouTube for videos"""
    return "Choufli hal 2005 episode 10"

# -------------------------------
# Define Agent with Direct Tools
# -------------------------------

tools_agent = Agent(
    name="ToolsAgent",
    model=model,
    instructions="""
    You are an agent that uses tools to answer user questions.
    
    You have the following tools available:
    - meaning_of_life(): Use this when asked about the meaning of life
    - web_search(): Use this when asked to web search
    - youtube_search(): Use this when asked to search YouTube
    
    IMPORTANT: 
    - When responding, call the appropriate function based on the query
    - DO NOT make up answers - ONLY use the tools to get information
    - Return the exact output from the tool without modification
    """,
    functions=[meaning_of_life, web_search, youtube_search],
    tool_choice="auto",  # Force function calling
    llm=ollama_client
)

# -------------------------------
# Main Execution
# -------------------------------

def main():
    """Run the tools agent system"""
    
    # Initialize Swarm client
    swarm_client = Swarm(ollama_client)
    
    # Test queries
    queries = [
        "What's the meaning of life? only answer based on the TOOL",
          
    ]
    
    for query in queries:
        # Run the agent
        print(f"\n----- Query: {query} -----")
        
        try:
            response = swarm_client.run(
                agent=tools_agent,
                messages=[{"role": "user", "content": query}],
                debug=True  # Enable debug output
            )
            
            # Print the response
            print(f"Response: {response.messages[-1]['content']}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
if __name__ == "__main__":
    main()