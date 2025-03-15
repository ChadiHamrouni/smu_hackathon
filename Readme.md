# AI Workshop: Tool Use, Reflection, and Multi-Agent Systems
## Setup Instructions

### Option 1: conda
`conda create -n ai-workshop` 
`conda activate ai-workshop`

# Option 2: virtualenv
`virtualenv ai-workshop`
`cd ai-workshop`
`scripts\activate`

# Install packages
`pip install openai swarm`

# Download & Install Ollama
https://ollama.com/

# Pull Qwen2.5:7b Model
`ollama pull qwen2.5:7b`

# Workshop Files

## `tool_use.py` : Basic agent that uses tools to answer queries.

## `reflection.py` : Agent that evaluates and improves its own responses.

## `multi_agent.py` : Multi-agent system for collaborative report generation.

