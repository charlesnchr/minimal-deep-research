# Deep Researcher

A command-line tool for performing deep research on a topic using a large language model (LLM) and web search APIs.

## Overview

Deep Researcher leverages an LLM and web search (using either Tavily or Perplexity) to generate targeted search queries, gather data from the web, and summarize the findings. The tool is designed to simplify the research process with a clean, class-based structure. The main functionality is encapsulated in the `DeepResearcher` class.

## Features

- **Automated Query Generation:** Uses an LLM to generate a search query from a research topic.
- **Web Research:** Retrieves relevant web search results using a configurable API (Tavily or Perplexity).
- **Summarization:** Summarizes the gathered sources into a concise report.
- **Reflection:** Generates follow-up queries based on the summary to refine the research.
- **Final Summary:** Consolidates all information and sources into a final formatted summary.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/charlesnchr/langgraph-deep-research.git
   cd langgraph-deep-research
   ```

2. Create a virtual environment and activate it:

   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

3. Set up your environment variables:

   Create a `.env` file in the project root and add any necessary API keys (e.g., PERPLEXITY_API_KEY).

## Usage

Run the research assistant using the command line. The tool accepts the research topic and optional parameters such as maximum loops, LLM model, and search API.

```bash
python standalone_deep_researcher.py "<research-topic>" [--max-loops <number>] [--llm-model <model-name>] [--search-api <tavily|perplexity>]
```

### Example

```bash
python standalone_deep_researcher.py "Deep Learning" --max-loops 3 --llm-model "llama-3.3-70b" --search-api tavily
```

## Configuration

Configuration options can be set via command-line arguments:

- **topic:** The research topic to investigate.
- **--max-loops:** Maximum number of research iterations (default: 3).
- **--llm-model:** Name of the LLM model to use (default: llama-3.3-70b).
- **--search-api:** Choose between "tavily" or "perplexity" for web search (default: tavily).
