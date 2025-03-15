import streamlit as st
import re
from swarm import Agent, Swarm
from openai import OpenAI

# Setup page
st.set_page_config(page_title="Swarm Report Generator", layout="wide")
st.title("Swarm Report Generator")

# OLLAMA CLIENT SETUP
@st.cache_resource
def get_client():
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

ollama_client = get_client()
model = "qwen2.5:7b"

# -------------------------------
# Task Functions with flexible parameters
# -------------------------------

def generate_outline(*args, **kwargs):
    """Generate a structured outline for the given topic"""
    # Get topic from st.session_state
    topic = st.session_state.topic
    
    st.write("🔍 **OutlineAgent:** Generating outline...")
    
    response = ollama_client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant that generates clear, structured outlines. Respond ONLY with the outline itself."
        }, {
            "role": "user",
            "content": f"Create a detailed 2-section outline for a report on {topic}. Format as a numbered list (1., 2., etc.)."
        }]
    )
    
    outline = response.choices[0].message.content
    st.session_state.outline = outline
    
    st.write("✅ **OutlineAgent:** Outline generated!")
    st.markdown("### Generated Outline:")
    st.markdown(outline)
    
    # Return the report agent to hand off to
    return report_agent

def generate_report(*args, **kwargs):
    """Generate a full markdown report based on the outline"""
    # Get data from session state
    topic = st.session_state.topic
    outline = st.session_state.outline
    
    st.write("📝 **ReportAgent:** Beginning report generation...")
    
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
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Generate content for each chapter
    for i, chapter in enumerate(chapters):
        st.write(f"📝 **ReportAgent:** Writing chapter {i+1}/{len(chapters)}: **{chapter}**")
        
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
        report += f"## {chapter}\n\n{chapter_content}\n\n"
        
        # Update progress
        progress_bar.progress((i + 1) / len(chapters))
    
    # Save report to file (keeping this from original implementation)
    with open("report.md", "w", encoding="utf-8") as md_file:
        md_file.write(report)
    
    st.write("✅ **ReportAgent:** Report generation complete!")
    
    # Display the full report
    st.markdown("## Generated Report")
    st.markdown(report)
    
    return f"Report generated successfully. Length: {len(report)} characters. Report displayed above."

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

def handoff_to_report_agent(*args, **kwargs):
    """Handoff from OutlineAgent to ReportAgent"""
    st.write("🔄 Handing off from OutlineAgent to ReportAgent...")
    return report_agent

# -------------------------------
# Define Agents
# -------------------------------

# Agent One: OutlineAgent
@st.cache_resource
def get_outline_agent():
    return Agent(
        name="OutlineAgent",
        model=model,
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
@st.cache_resource
def get_report_agent():
    return Agent(
        name="ReportAgent",
        model=model,
        instructions="""
        You are a specialized agent that creates full reports based on outlines.
        
        Use the generate_report() function to create a complete markdown report.
        """,
        functions=[generate_report, handoff_to_report_agent],
        llm=ollama_client  
    )

# Get the cached agents
outline_agent = get_outline_agent()
report_agent = get_report_agent()

# -------------------------------
# Streamlit UI
# -------------------------------

# Initialize session state for storing variables
if 'topic' not in st.session_state:
    st.session_state.topic = ""
if 'outline' not in st.session_state:
    st.session_state.outline = ""

# Topic input
topic_input = st.text_input("Enter a topic for your report:", "Artificial Intelligence")

if st.button("Generate Report"):
    # Store topic in session state so functions can access it
    st.session_state.topic = topic_input
    
    st.write(f"🚀 Starting report generation process for: **{topic_input}**")
    
    # Initialize the Swarm client
    swarm_client = Swarm(ollama_client)
    
    # User message asking for a report
    user_message = {"role": "user", "content": f"Generate a comprehensive report about {topic_input}."}
    
    try:
        # Run the swarm - it will start with outline_agent and hand off to report_agent
        response = swarm_client.run(
            agent=outline_agent,
            messages=[user_message]
        )
        
        # Final message
        st.success("Report generation process completed!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.warning("""
        Troubleshooting:
        1. Make sure Ollama is running locally at http://localhost:11434
        2. Verify that the llama3.2 model is available
        3. Check network connectivity to the Ollama server
        """)