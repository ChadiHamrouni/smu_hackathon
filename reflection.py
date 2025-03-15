from swarm import Agent, Swarm
from openai import OpenAI

# Setup
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
model = "qwen2.5:7b"

# Simple tools
def meaning_of_life():
    """Returns the meaning of life"""
    return "651654116"

def web_search():
    """Performs a web search"""
    return "3 kittens"

def youtube_search():
    """Searches YouTube"""
    return "Choufli hal 2005 episode 10"

# Agents
tools_agent = Agent(
    name="ToolsAgent",
    model=model,
    instructions="You use tools to answer questions. ONLY return the tool's output with no other text.",
    functions=[meaning_of_life, web_search, youtube_search],
    tool_choice="auto",
    llm=ollama_client
)

reflection_agent = Agent(
    name="ReflectionAgent",
    model=model,
    instructions="""
You are checking if the answer of the agent only contain the tool outputs and not the agent's opinion nor its explaination of the tool outputs. If the answer contains the agent's opinion or its explaination of the tool outputs, you will say "BAD". If the answer only contain the tool outputs, you will say "GOOD".
    """,
    llm=ollama_client
)


# Main function
def run_with_reflection(query, max_tries=3):
    swarm = Swarm(ollama_client)
    
    for attempt in range(1, max_tries+1):
        print(f"Attempt {attempt}/{max_tries}")
        
        # Get tool response
        response = swarm.run(
            agent=tools_agent,
            messages=[{"role": "user", "content": query}],
            debug=True
        )
        output = response.messages[-1]["content"]
        print(f"Output: {output}")
        
        # Check quality
        reflection = swarm.run(
            agent=reflection_agent,
            messages=[{"role": "user", "content": f"Evaluate: {output}"}]
        )
        verdict = reflection.messages[-1]["content"]
        print(f"Verdict: {verdict}")
        
        # If good, we're done
        if "GOOD" in verdict:
            return output
        
        # Add feedback for next attempt
        query = f"{query}\nPrevious attempt failed. ONLY return the exact tool output with no other text."
    
    return output  # Return last attempt even if not perfect

# Example usage
if __name__ == "__main__":
    query = "What's the meaning of life?"
    max_tries = 3
    final_output = run_with_reflection(query, max_tries)
    print(f"\nFinal output: {final_output}")