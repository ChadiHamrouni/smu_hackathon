import re
from swarm import Agent, Swarm
from openai import OpenAI 

# OLLAMA CLIENT SETUP
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Required but unused
)

model = "qwen2.5:7b"
# -------------------------------
# Task Functions
# -------------------------------

def generate_outline(context_variables):
    """Generate a structured outline for the given topic"""
    topic = context_variables["topic"]
    
    response = ollama_client.chat.completions.create(
        model= model,
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant that generates clear, structured outlines. Respond ONLY with the outline itself."
        }, {
            "role": "user",
            "content": f"Create a detailed 5-section outline for a report on {topic}. Format as a numbered list (1., 2., etc.)."
        }]
    )
    
    outline = response.choices[0].message.content
    context_variables["outline"] = outline
    
    # Return the report agent to hand off to
    return response.choices[0].message.content

def generate_report(context_variables):
    """Generate a full markdown report based on the outline"""
    topic = context_variables["topic"]
    outline = context_variables["outline"]
    
    # Extract chapter titles
    chapters = extract_chapters(outline)
    
    if not chapters or len(chapters) < 2:  # If extraction fails or too few chapters
        # Fallback to simple sections
        chapters = ["Introduction", "Background", "Current Developments", "Applications", "Conclusion"]
    
    # Create report markdown
    report = f"# Comprehensive Report on {topic}\n\n"
    
    # Add outline section
    report += "## Outline\n\n"
    report += outline + "\n\n"
    
    # Generate content for each chapter
    for chapter in chapters:
        report += f"## {chapter}\n\n"
        
        response = ollama_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant that writes informative content. Respond ONLY with the requested content in markdown format."
            }, {
                "role": "user",
                "content": f"Write a small report section for the '{chapter}' chapter of a report about {topic}. Include relevant facts, examples, and analysis. Format in markdown."
            }],
            max_tokens=500,
        )
        
        # Add content and spacing
        chapter_content = response.choices[0].message.content
        report += chapter_content + "\n\n"
    
    # Save report to file
    with open("report.md", "w", encoding="utf-8") as md_file:
        md_file.write(report)
    
    return f"Report generated successfully. Length: {len(report)} characters. Saved to report.md"

def extract_chapters(outline_text: str) -> list:
    """Extract chapter titles from a numbered outline text"""
    # Try to extract numbered sections like "1. Title" or "1) Title"
    chapter_pattern = r'^\s*(\d+[\.\)])\s*(.+)$'
    chapters = []
    
    for line in outline_text.split('\n'):
        match = re.match(chapter_pattern, line.strip())
        if match:
            chapters.append(match.group(2).strip())
    
    return chapters

def handoff_to_report_agent(context_variables):
    """Handoff from OutlineAgent to ReportAgent"""
    return report_agent, context_variables

# -------------------------------
# Define Agents
# -------------------------------

# Agent One: OutlineAgent
outline_agent = Agent(
    name="OutlineAgent",
    model= model,
    instructions="""
    You are a specialized agent that creates structured outlines.
    
    When asked to generate a report, you should:
    1. Use the generate_outline() function to create an outline for the given topic
    2. This function will automatically hand you off to the ReportAgent
    """,
    functions=[generate_outline],
    llm=ollama_client  
)

# Agent Two: ReportAgent
report_agent = Agent(
    name="ReportAgent",
    model= model,
    instructions="""
    You are a specialized agent that creates structured outlines.
    
    When asked to generate a report, you should:
    1. Use the generate_outline() function to create an outline for the given topic.
    2. Hand off control to the ReportAgent using the handoff_to_report_agent() function.
    """,
    functions=[generate_report, handoff_to_report_agent],
    llm=ollama_client  
)

# -------------------------------
# Main Execution
# -------------------------------

def main():
    """Run the complete report generation process"""
    # Topic to generate report about
    topic = "Artificial Intelligence"
    
    # Initialize the Swarm client
    swarm_client = Swarm(ollama_client)  
    
    # User message asking for a report
    user_message = {"role": "user", "content": f"Generate a comprehensive report about {topic}."}
    
    print(f"Starting report generation process for: {topic}\n")
    
    try:
        # Run the swarm - it will start with outline_agent and hand off to report_agent
        response = swarm_client.run(
            agent=outline_agent,
            messages=[user_message],
            context_variables={"topic": topic}
        )
        
        # Display the final response
        print("\nProcess completed!")
        print(response.messages[-1]["content"])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running locally at http://localhost:11434")
        print("2. Verify that the latest model is available")
        print("3. Check network connectivity to the Ollama server")

if __name__ == "__main__":
    main()