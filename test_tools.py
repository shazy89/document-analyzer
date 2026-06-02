"""
Quick CLI for manually testing AnalyzerAgentTools methods.

Usage:
    python test_tools.py web_search "Senior Python engineer remote"
    python test_tools.py search_jobs --role "Full Stack Engineer" --skills Python React --locations "New York" Remote
    python test_tools.py search_content --role "Full Stack Engineer" --skills Python React --locations "New York" Remote --query "full stack engineer"
    python test_tools.py scrape "https://example.com"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

# ── Bootstrap ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from document_analyzer.analyzer_agent.config import DocumentAnalyzerConfig
from document_analyzer.analyzer_agent.tools import AnalyzerAgentTools
from document_analyzer.services.together_client import TogetherChatService
from document_analyzer.analyzer_agent.agent import create_job_hunter_agent

def _build_tools() -> AnalyzerAgentTools:
    config = DocumentAnalyzerConfig.from_env()
    service = TogetherChatService(api_key=config.api_key, default_model=config.model_name)
    return AnalyzerAgentTools(config=config, service=service)


# ── Subcommand handlers ───────────────────────────────────────────────────────

def run_web_search(args: argparse.Namespace) -> None:
    tools = _build_tools()
    results = tools.web_search(args.query)
    _print_json(results)


def run_search_jobs(args: argparse.Namespace) -> None:
    tools = _build_tools()
    results = tools.search_jobs(
        role=args.role,
        skills=args.skills,
        locations=args.locations,
        remote_preference=args.remote_preference,
        min_salary=args.min_salary,
        seniority=args.seniority,
    )
    _print_json(results)


def run_search_content(args: argparse.Namespace) -> None:
    tools = _build_tools()
    results = tools.search_content(
        query=args.query,
        role=args.role,
        skills=args.skills,
        locations=args.locations,
        remote_preference=args.remote_preference,
        min_salary=args.min_salary,
        seniority=args.seniority,
    )
    _print_json(results)


def run_scrape(args: argparse.Namespace) -> None:
    tools = _build_tools()
    raw = tools._scrape_web_page(args.url)
    cleaned = tools._clean_scraped_text(raw)
    print(cleaned)


def run_agent(args: argparse.Namespace) -> None:
    from langchain_core.messages import HumanMessage
    import uuid

    config = DocumentAnalyzerConfig.from_env()
    agent = create_job_hunter_agent(config)
    graph = agent.build_graph()

    thread_id = str(uuid.uuid4())

    initial_state = {
        "messages": [HumanMessage(content=args.message)],
        "search_queries": [],
        "found_links": [],
        "scraped_urls": [],
        "skipped_urls": [],
        "normalized_jobs": [],
        "errors": [],
        "step": "start",
    }

    run_config = {"configurable": {"thread_id": thread_id}}

    print(f"\n── Thread {thread_id} ──")

    for step in graph.stream(
        initial_state,
        config=run_config,
        stream_mode="values",
    ):
        last = step["messages"][-1]
        role = last.__class__.__name__
        content = getattr(last, "content", "")
        tool_calls = getattr(last, "tool_calls", [])

        if tool_calls:
            print(f"\n[{role}] → {len(tool_calls)} tool call(s)")
            for tc in tool_calls:
                print(f"  tool : {tc['name']}")
                print(f"  args : {json.dumps(tc.get('args', {}), indent=4, ensure_ascii=False)}")
                print(f"  id   : {tc['id']}")
        elif content:
            print(f"\n[{role}]\n{content}")

        if step.get("found_links"):
            print(f"\nFound links so far : {len(step['found_links'])}")
        if step.get("normalized_jobs"):
            print(f"Normalized jobs so far : {len(step['normalized_jobs'])}")
        if step.get("errors"):
            print(f"Errors : {step['errors']}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test AnalyzerAgentTools methods from the command line."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # web_search
    p_ws = sub.add_parser("web_search", help="Run a raw DuckDuckGo search.")
    p_ws.add_argument("query", help='Search query string, e.g. "Senior Python engineer remote"')
    p_ws.set_defaults(func=run_web_search)

    # search_jobs
    p_sj = sub.add_parser("search_jobs", help="Build targeted job queries and search.")
    p_sj.add_argument("--role", required=True, help='Job role, e.g. "Full Stack Engineer"')
    p_sj.add_argument("--skills", nargs="+", required=True, help="Skill list, e.g. Python React AWS")
    p_sj.add_argument("--locations", nargs="+", required=True, help='Location list, e.g. "New York" Remote')
    p_sj.add_argument("--remote-preference", default="remote OR hybrid", dest="remote_preference")
    p_sj.add_argument("--min-salary", type=int, default=None, dest="min_salary")
    p_sj.add_argument("--seniority", default="senior OR mid-senior")
    p_sj.set_defaults(func=run_search_jobs)

    # search_content (full pipeline: search → scrape → normalize)
    p_sc = sub.add_parser("search_content", help="Full pipeline: search → scrape → LLM normalize.")
    p_sc.add_argument("--query", required=True, help="Freeform query string passed to search_content.")
    p_sc.add_argument("--role", required=True)
    p_sc.add_argument("--skills", nargs="+", required=True)
    p_sc.add_argument("--locations", nargs="+", required=True)
    p_sc.add_argument("--remote-preference", default="remote OR hybrid", dest="remote_preference")
    p_sc.add_argument("--min-salary", type=int, default=None, dest="min_salary")
    p_sc.add_argument("--seniority", default="senior OR mid-senior")
    p_sc.set_defaults(func=run_search_content)

    # scrape (useful for quickly checking a single URL)
    p_scrape = sub.add_parser("scrape", help="Scrape and clean a single URL.")
    p_scrape.add_argument("url", help="URL to scrape")
    p_scrape.set_defaults(func=run_scrape)

    # agent — run the full DocumentAnalyzerAgent graph with a single message
    p_agent = sub.add_parser("agent", help="Run the full agent graph with a message.")
    p_agent.add_argument(
        "message",
        help='Natural-language instruction, e.g. "Find senior Python jobs in New York"',
    )
    p_agent.set_defaults(func=run_agent)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
