# Minimal Deep Research

A minimal Gemini/OpenAI-inspired implementation of Deep Research using structured outputs with LangChain, an LLM provider, and a web search API.

Want to try it out? You can use any OpenAI-compatible LLM - whether that's a local model through Ollama or cloud options like Gemini 2.0 (shown in the demo below).

https://github.com/user-attachments/assets/df48ec08-2fea-4131-879d-22a282969f44

## What's This?

Deep Research combines the power of LLMs with web search (via Tavily or Perplexity) to do what researchers do: ask good questions, gather information, and synthesize findings. But instead of taking hours, it takes minutes.

The core functionality resides in the `DeepResearcher` class, which:
- Crafts smart search queries based on your research topic
- Scours the web for relevant information
- Summarizes what it finds
- Identifies gaps and digs deeper
- Pulls everything together into a clear final report

## Getting Started

1. Grab the code:
   ```bash
   git clone https://github.com/charlesnchr/minimal-deep-research
   cd minimal-deep-research
   ```

2. Set up your environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Add your API keys:
   Create a `.env` file in the root directory and add your keys (e.g., PERPLEXITY_API_KEY)

## How to Use It

Basic usage is straightforward:
```bash
python deep_research.py "Your research question here"
```

To use some of the optional command-line arguments:
```bash
python deep_research.py "What are recent advances in quantum computing?" \
    --max-loops 10 \
    --model-id "o3-mini" \
    --search-api tavily
```

### Configuration Options

- **topic:** The research topic to investigate
- **--max-loops:** How many research iterations to run (default: 3)
- **--model-id:** Which LLM to use (default: llama-3.3-70b)
- **--search-api:** Pick your search provider - "tavily" or "perplexity" (default: tavily)

## Acknowledgements
This was inspired by [ollama-deep-researcher](https://github.com/langchain-ai/ollama-deep-researcher), but with a focus on simplicity. The core logic lives in a clean `deep_research.py` file - no LangGraph topology, nor the LangGraph server required to run it, instead relying on a simple main loop and the `deep_research.py` as an entrypoint.