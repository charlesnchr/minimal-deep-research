# Minimal Deep Research

A minimal Gemini/OpenAI-inspired implementation of Deep Research using structured outputs with LangChain, an LLM provider, and a web search API.

Any OpenAI-compatible LLM provider can be used in this repository by specifying the corresponding `.env` variables, including running local models with Ollama.

## Overview

Deep Research leverages an LLM and web search (using either Tavily or Perplexity) to generate targeted search queries, gather data from the web, and summarize the findings. The tool is designed to simplify the research process with a clean, class-based structure. The main functionality is encapsulated in the `DeepResearcher` class.

## Steps

- **Automated Query Generation:** Uses an LLM to generate a search query from a research topic.
- **Web Research:** Retrieves relevant web search results using a configurable API (Tavily or Perplexity).
- **Summarization:** Summarizes the gathered sources into a concise report.
- **Reflection:** Generates follow-up queries based on the summary to refine the research.
- **Final Summary:** Consolidates all information and sources into a final formatted summary.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/charlesnchr/minimal-deep-research
   cd minimal-deep-research
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   Create a `.env` file in the project root and add any necessary API keys (e.g., PERPLEXITY_API_KEY).

## Usage

Run the research assistant using the command line. The tool accepts the research topic and optional parameters such as maximum loops, LLM model, and search API.

```bash
python deep_research.py "<research-topic>" [--max-loops <number>] [--model-id <model-name>] [--search-api <tavily|perplexity>]
```

### Example

```bash
python deep_research.py "What are recent advances in quantum computing?" --max-loops 3 --model-id "gpt-4o" --search-api tavily
```

## Configuration

Configuration options can be set via command-line arguments:

- **topic:** The research topic to investigate.
- **--max-loops:** Maximum number of research iterations (default: 3).
- **--model-id:** Name of the LLM model to use (default: llama-3.3-70b).
- **--search-api:** Choose between "tavily" or "perplexity" for web search (default: tavily).

## Acknowledgements

This repository was inspired by:

- [ollama-deep-researcher](https://github.com/langchain-ai/ollama-deep-researcher)

The main selling point of this repository is that it is lighter and simpler to build upon. It has a straightforward main loop in `deep_research.py`, which does not rely on LangGraph, nor is it necessary to run a LangGraph server to start the execution.

Ollama could equally be used in this repository by specifying corresponding `.env` variables.
