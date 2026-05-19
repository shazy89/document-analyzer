from ddgs import results
from ddgs.exceptions import DDGSException
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool
import logging
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from document_analyzer.analyzer_agent.config import DocumentAnalyzerConfig
import requests
import re
from src.document_analyzer.analyzer_agent.prompts import WEB_SEARCH_NORMALIZER_PROMPT

from document_analyzer.services.together_client import TogetherChatService

logger = logging.getLogger(__name__)


class AnalyzerAgentTools:
    """Central registry for tools used by the analyzer agent."""

    def __init__(self, config: DocumentAnalyzerConfig, service: TogetherChatService):
        self.tools: dict[str, BaseTool] = {}
        self.config = config
        self.service = service

    def register_tool(self, name: str, func: callable):
        """Register a tool function under a given name."""
        self.tools[name] = func if isinstance(func, BaseTool) else tool(func)
        logger.info("Registered tool: %s", name)

    def search_content(
        self,
        query: str,
        role: str,
        skills: list[str],
        locations: list[str],
        remote_preference: str = "remote OR hybrid",
        min_salary: int | None = None,
        seniority: str = "senior OR mid-senior",
    ) -> list[dict]:
        """Search job posts, scrape pages, clean content, and return normalized results."""
        logger.debug("search_content called query=%s", query)

        web_results = self.search_jobs(
            role=role,
            skills=skills,
            locations=locations,
            remote_preference=remote_preference,
            min_salary=min_salary,
            seniority=seniority,
        )

        normalized_results = []

        for result in web_results:
            link = result.get("link")

            if not link:
                continue

            raw_content = self._scrape_web_page(link)

            if raw_content.startswith("[skipped:"):
                logger.debug("Skipping scrape result for %s: %s", link, raw_content)
                continue

            cleaned_content = self._clean_scraped_text(raw_content)

            if self._should_skip_content(cleaned_content):
                logger.debug("Skipping low-quality/blocked content for %s", link)
                continue

            prompt = f"""
                Search result metadata:
                Title: {result.get("title", "unknown")}
                Snippet: {result.get("snippet", "")}
                Link: {link}

                Page content:
                {cleaned_content[:12000]}
            """

            normalized = self.service.ask(
                prompt=prompt,
                system_prompt=WEB_SEARCH_NORMALIZER_PROMPT,
            )

            normalized_results.append(
                {
                    "source_query": result.get("query"),
                    "source_title": result.get("title", "unknown"),
                    "source_snippet": result.get("snippet", ""),
                    "source_link": link,
                    "normalized": normalized.answer,
                }
            )

        logger.debug(
            "search_content completed",
            extra={"result_count": len(normalized_results)},
        )

        return normalized_results

    def web_search(self, query: str) -> list[dict]:
        """Search the web and return structured search results."""
        logger.debug("web_search called query=%s", query)

        search_tool = DuckDuckGoSearchResults(max_results=30, output_format="list")

        try:
            results = search_tool.invoke(query)
        except DDGSException as e:
            logger.warning("DuckDuckGo search failed for query %r: %s", query, e)
            return []

        logger.debug("web_search completed", extra={"results": results})
        return results

    def search_jobs(
        self,
        role: str,
        skills: list[str],
        locations: list[str],
        remote_preference: str = "remote OR hybrid",
        min_salary: int | None = None,
        seniority: str = "senior OR mid-senior",
    ) -> list[dict]:
        skill_text = " ".join(f'"{skill}"' for skill in skills)
        location_text = " OR ".join(f'"{location}"' for location in locations)
        salary_text = f'"{min_salary}" OR "${min_salary:,}"' if min_salary else ""
        remote_text = " OR ".join(f'"{x.strip()}"' for x in remote_preference.split("OR"))
        seniority_text = " OR ".join(f'"{x.strip()}"' for x in seniority.split("OR"))

        base_query = (
            f'"{role}" '
            f'{skill_text} '
            f'({location_text}) '
            f'({remote_text}) '
            f'({seniority_text}) '
            f'{salary_text}'
        )

        queries = [
            base_query,
            f"site:greenhouse.io {base_query}",
            f"site:lever.co {base_query}",
            f"site:ashbyhq.com {base_query}",
            f"site:jobs.ashbyhq.com {base_query}",
            f"site:workdayjobs.com {base_query}",
         
        ]

        seen_links = set()
        all_results = []

        for i, query in enumerate(queries):
            if i > 0:
                time.sleep(2)
            results = self.web_search(query)

            for result in results:
                link = result.get("link") or result.get("url")
                if not link or link in seen_links:
                    continue

                seen_links.add(link)

                all_results.append(
                    {
                        "query": query,
                        "title": result.get("title", "unknown"),
                        "snippet": result.get("snippet") or result.get("body", ""),
                        "link": link,
                    }
                )

        return all_results

    def _web_search(self, query: str) -> list[dict]:
        """Search the web and return structured search results."""
        logger.debug("web_search called query=%s", query)

        search_tool = DuckDuckGoSearchResults(max_results=30, output_format="list")

        results = search_tool.invoke(query)

        logger.debug("web_search completed", extra={"results": results})
        return results

    def _scrape_web_page(self, url: str) -> str:
        """Load readable text content from a web page."""
        logger.debug("scrape_web_page called url=%s", url)

        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            logger.debug("scrape_web_page completed")
            return "\n\n".join(doc.page_content for doc in docs)
        except requests.exceptions.SSLError as e:
            logger.warning("SSL error scraping %s: %s", url, e)
            return f"[skipped: SSL error for {url}]"
        except Exception as e:
            logger.warning("Failed to scrape %s: %s", url, e)
            return f"[skipped: {e}]"

    def _clean_scraped_text(self, text: str, max_chars: int = 12000) -> str:
        noise_phrases = [
            "skip to main content",
            "expand search",
            "this button displays",
            "notice: this page displays a fallback",
            "how it works",
            "about us",
            "privacy policy",
            "terms of service",
            "cookie policy",
            "accept cookies",
            "sign in",
            "create job alert",
            "people also viewed",
            "recommended jobs",
            "show more",
            "show less",
            "subscribe",
            "newsletter",
        ]

        seen = set()
        cleaned = []

        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()

            if not line:
                continue

            lower_line = line.lower()

            if any(phrase in lower_line for phrase in noise_phrases):
                continue

            if len(line) < 3:
                continue

            if lower_line in seen:
                continue

            seen.add(lower_line)
            cleaned.append(line)

        result = "\n".join(cleaned)

        return result[:max_chars]

    def _should_skip_content(self, text: str) -> bool:
        lower = text.lower()

        if len(text) < 500:
            return True

        """
        cleaned_content = self._clean_scraped_text(content)

        if self._should_skip_content(cleaned_content):
            continue

        Returns:
            _type_: _description_
        """
        blocked_phrases = [
            "please enable javascript",
            "access denied",
            "captcha",
            "verify you are human",
            "this page is not available",
        ]

        return any(phrase in lower for phrase in blocked_phrases)
