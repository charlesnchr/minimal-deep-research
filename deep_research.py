import argparse
import os
from enum import Enum
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()


class SearchAPI(str, Enum):
    """The search API to use for the research assistant."""

    PERPLEXITY = "perplexity"
    TAVILY = "tavily"


class Configuration(BaseModel):
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = 3
    local_llm: str = "gpt-4o-mini"
    search_api: SearchAPI = SearchAPI.TAVILY

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Initialize the configuration from a runnable config."""
        if config and "configurable" in config:
            return cls(**config["configurable"])
        return cls()


# --- Prompts ---
query_writer_instructions = """As a research query generator, create a targeted web search query to gather information about:

{research_topic}

Your goal is to formulate a precise query that will yield relevant, high-quality information about this topic.

Example output:
{{
    "query": "machine learning transformer architecture explained",
    "aspect": "technical architecture",
    "rationale": "Understanding the fundamental structure of transformer models"
}}

Generate your query now:"""

summarizer_instructions = """You are tasked with generating a high-quality, concise summary of web search results related to the user's topic.

For a new summary:
• Extract and highlight the most relevant information
• Maintain a logical flow of ideas
• Focus on accuracy and clarity

When extending an existing summary:
• Integrate new information seamlessly with existing content
• Add new, relevant details while maintaining coherence
• Skip redundant or irrelevant information
• Ensure the final summary shows clear progression from the previous version

Begin your summary directly, focusing on the most important information:"""

reflection_instructions = """As a research analyst examining our current knowledge about {research_topic}, identify gaps in our understanding and propose targeted follow-up questions.

Focus on:
• Technical details that need clarification
• Implementation specifics that are unclear
• Emerging trends or developments not yet covered
• Practical applications or implications not discussed

Example output:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}

Analyze the current summary and provide your insights:"""


class SearchQuery(BaseModel):
    """Model for the search query generation output."""

    query: str = Field(description="The actual search query string")
    aspect: str = Field(description="The specific aspect of the topic being researched")
    rationale: str = Field(
        description="Brief explanation of why this query is relevant"
    )


class ReflectionOutput(BaseModel):
    """Model for the reflection output."""

    knowledge_gap: str = Field(
        description="Description of what information is missing or needs clarification"
    )
    follow_up_query: str = Field(
        description="Specific question to address the identified knowledge gap"
    )


class DeepResearcher:
    """The main class for the deep researcher."""

    def __init__(self, config: Configuration):
        self.config = config

    def get_llm(self, structured_output=None, streaming: bool = False):
        """Get the LLM client."""
        client = ChatOpenAI(model=self.config.local_llm, streaming=streaming)
        if structured_output:
            client = client.with_structured_output(structured_output)
        return client

    def generate_query(self, research_topic: str) -> str:
        """Generate a query for web search."""
        instructions = query_writer_instructions.format(research_topic=research_topic)
        llm_client = self.get_llm(structured_output=SearchQuery)
        print("Starting query generation...", flush=True)
        result = llm_client.invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(content="Generate a query for web search:"),
            ]
        )
        return result.query

    def web_research(self, query: str, loop_count: int):
        """Perform web research on a query."""
        if self.config.search_api == SearchAPI.TAVILY:
            search_results = self.tavily_search(
                query, include_raw_content=True, max_results=1
            )
            search_summary = self.deduplicate_and_format_sources(
                search_results, max_tokens_per_source=1000, include_raw_content=True
            )
        elif self.config.search_api == SearchAPI.PERPLEXITY:
            search_results = self.perplexity_search(query, loop_count)
            search_summary = self.deduplicate_and_format_sources(
                search_results, max_tokens_per_source=1000, include_raw_content=False
            )
        else:
            raise ValueError(f"Unsupported search API: {self.config.search_api}")
        sources = [self.format_sources(search_results)]
        return search_summary, sources

    def summarize_sources(
        self, research_topic: str, current_summary: Optional[str], search_summary: str
    ) -> str:
        """Summarize the sources."""
        if current_summary:
            message = (
                f"<User Input> \n {research_topic} \n <User Input>\n\n"
                f"<Existing Summary> \n {current_summary} \n <Existing Summary>\n\n"
                f"<New Search Results> \n {search_summary} \n <New Search Results>"
            )
        else:
            message = (
                f"<User Input> \n {research_topic} \n <User Input>\n\n"
                f"<Search Results> \n {search_summary} \n <Search Results>"
            )
        llm_client = self.get_llm()
        result = llm_client.invoke(
            [
                SystemMessage(content=summarizer_instructions),
                HumanMessage(content=message),
            ]
        )
        return result.content

    def reflect_on_summary(self, research_topic: str, current_summary: str) -> str:
        """Reflect on the summary."""
        llm_client = self.get_llm(structured_output=ReflectionOutput, streaming=True)
        message = f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {current_summary}"
        result = llm_client.invoke(
            [
                SystemMessage(
                    content=reflection_instructions.format(
                        research_topic=research_topic
                    )
                ),
                HumanMessage(content=message),
            ]
        )
        return result.follow_up_query

    def finalize_summary(self, running_summary: str, sources_gathered: list) -> str:
        """Finalize the summary."""
        all_sources = "\n".join(sources_gathered)
        final = f"## Summary\n\n{running_summary}\n\n### Sources:\n{all_sources}"
        return final

    def run_research(self, topic: str) -> str:
        """Run the research."""
        research_loop_count = 0
        sources_gathered = []
        running_summary = None

        # Generate initial query
        search_query = self.generate_query(topic)
        print(f"Initial search query: {search_query}", flush=True)

        while research_loop_count < self.config.max_web_research_loops:
            print(
                f"\nResearch Loop: {research_loop_count + 1}\nSearch Query: {search_query}\n",
                flush=True,
            )
            search_summary, sources = self.web_research(
                search_query, research_loop_count
            )
            research_loop_count += 1
            sources_gathered.extend(sources)
            running_summary = self.summarize_sources(
                topic, running_summary, search_summary
            )
            search_query = self.reflect_on_summary(topic, running_summary)

        final_summary = self.finalize_summary(running_summary, sources_gathered)
        return final_summary

    @staticmethod
    def deduplicate_and_format_sources(
        search_response, max_tokens_per_source, include_raw_content=False
    ):
        """Deduplicate and format the sources."""
        if isinstance(search_response, dict):
            sources_list = search_response["results"]
        elif isinstance(search_response, list):
            sources_list = []
            for response in search_response:
                if isinstance(response, dict) and "results" in response:
                    sources_list.extend(response["results"])
                else:
                    sources_list.extend(response)
        else:
            raise ValueError(
                "Input must be either a dict with 'results' or a list of search results"
            )

        unique_sources = {}
        for source in sources_list:
            if source["url"] not in unique_sources:
                unique_sources[source["url"]] = source

        formatted_text = "Sources:\n\n"
        for source in unique_sources.values():
            formatted_text += f"Source {source['title']}:\n===\n"
            formatted_text += f"URL: {source['url']}\n===\n"
            formatted_text += (
                f"Most relevant content from source: {source['content']}\n===\n"
            )
            if include_raw_content:
                char_limit = max_tokens_per_source * 4
                raw_content = source.get("raw_content", "")
                if raw_content:
                    if len(raw_content) > char_limit:
                        raw_content = raw_content[:char_limit] + "... [truncated]"
                    formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        return formatted_text.strip()

    @staticmethod
    def format_sources(search_results):
        """Format the sources."""
        return "\n".join(
            f"* {source['title']} : {source['url']}"
            for source in search_results["results"]
        )

    @staticmethod
    def tavily_search(query, include_raw_content=True, max_results=3):
        """Search with Tavily."""
        tavily_client = TavilyClient()
        return tavily_client.search(
            query, max_results=max_results, include_raw_content=include_raw_content
        )

    @staticmethod
    def perplexity_search(
        query: str, perplexity_search_loop_count: int
    ) -> Dict[str, Any]:
        """Search with Perplexity."""
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources.",
                },
                {"role": "user", "content": query},
            ],
        }
        response = requests.post(
            "https://api.perplexity.ai/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])
        results = [
            {
                "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
                "url": citations[0],
                "content": content,
                "raw_content": content,
            }
        ]
        for i, citation in enumerate(citations[1:], start=2):
            results.append(
                {
                    "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
                    "url": citation,
                    "content": "See above for full content",
                    "raw_content": None,
                }
            )
        return {"results": results}


def main():
    """Main function to run deep research."""
    parser = argparse.ArgumentParser(
        description="Run deep research on a topic using an LLM and web search."
    )
    parser.add_argument("topic", help="The research topic to investigate")
    parser.add_argument(
        "--max-loops",
        type=int,
        help="Maximum number of research iterations (default: 3)",
    )
    parser.add_argument(
        "--llm-model", help="Name of the LLM model to use (default: llama-3.3-70b)"
    )
    parser.add_argument(
        "--search-api",
        choices=["tavily", "perplexity"],
        help="Search API to use (default: tavily)",
    )
    args = parser.parse_args()

    config_dict = {}
    if args.max_loops is not None:
        config_dict["max_web_research_loops"] = args.max_loops
    if args.llm_model is not None:
        config_dict["local_llm"] = args.llm_model
    if args.search_api is not None:
        config_dict["search_api"] = args.search_api

    config = Configuration.from_runnable_config({"configurable": config_dict})
    researcher = DeepResearcher(config)
    try:
        summary = researcher.run_research(args.topic)
        print("\n" + summary)
    except Exception as e:
        print(f"Error running research: {str(e)}")
        raise


if __name__ == "__main__":
    main()
