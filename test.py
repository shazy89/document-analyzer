from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
import logging
import requests
from langchain_community.tools import DuckDuckGoSearchResults
logger = logging.getLogger(__name__)
from src.document_analyzer.services.together_client import TogetherChatService
import os

WEB_SEARCH_NORMALIZER_PROMPT = """
You normalize job search results into valid JSON.

Use both the search result metadata and scraped page content.

Extract only information that is present.
Do not invent missing fields.
If a field is missing, use "unknown".
If required skills are missing, use [].

Pay attention to the job post link. We have to make sure the extracted information is consistent with the content of the page in the link. If there is a mismatch, we should trust the content of the page more than the metadata.

Return only valid JSON.

Schema:
{
  "title": "unknown",
  "company": "unknown",
  "location": "unknown",
  "salary": "unknown",
  "remote_type": "unknown",
  "required_skills": [],
  "seniority": "unknown",
  "posted_date": "unknown",
  "is_current_year": false,
  "link": "unknown",
  "fit_score": 0,
  "fit_reason": "..."
}

Fit score rules:
- Start from 0.
- Add up to 2 points if the title matches Full Stack Developer, Software Engineer, Full Stack Engineer, or similar.
- Add up to 3 points for matching skills: Python, JavaScript, TypeScript, React, AWS.
- Add up to 2 points for location: United States, New York, New Jersey, remote, hybrid.
- Add up to 2 points if salary is listed and is $140k or above.
- Add up to 1 point if seniority is senior or mid-senior.
- Subtract 3 points if the role is internship, junior, or entry-level.
- Maximum score is 10.
- Minimum score is 0.

Important:
- If salary is missing, do not guess.
- If skills are missing, return an empty array.
- If remote type is unclear, use "unknown".
- The fit_score must be an integer from 0 to 10.
- Extract posted_date if present.
- Set is_current_year to true only if the result/page clearly indicates it was posted or updated in 2026.
- If the posting date is missing, set posted_date to "unknown" and is_current_year to false.
- Do not assume a job is current just because the page still exists.
"""


@tool
def web_search(query: str) -> list[dict]:
    """Search the web and return structured search results."""
    logger.debug("web_search called query=%s", query)

    search_tool = DuckDuckGoSearchResults(
        max_results=30,
        output_format="list" 
        
    )

    results = search_tool.invoke(query)
    
    logger.debug("web_search completed", extra={"results": results})
    return results


def clean_scraped_text(text: str) -> str:
    noise_phrases = [
        "Skip to main content",
        "Expand search",
        "This button displays",
        "Notice: This page displays a fallback",
        "How It Works",
        "About Us",
    ]

    lines = text.splitlines()

    cleaned = []
    for line in lines:
        line = line.strip()

        if not line:
            continue

        if any(phrase in line for phrase in noise_phrases):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

def scrape_web_page(url: str) -> str:
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

service = TogetherChatService(api_key="tgp_v1_9ji9fOj72WiWsawgnJO8lF6P5GSFYC-5rwGVXJsVETE", default_model="openai/gpt-oss-120b")


# for result in web_search({"query": "Senior Full Stack Engineer Python React remote United States 140k"}):
#     print(result, "\n", "NEW RESULT", "\n")
#     link = result.get("link") or result.get("url", "")
#     if not link:
#         continue
#     content = scrape_web_page(link)
#     cleaned_content = clean_scraped_text(content)
#     normalized = service.ask(
#         prompt=cleaned_content,
#         system_prompt=WEB_SEARCH_NORMALIZER_PROMPT,
#     ).answer
#     print(normalized)  # Print the first 500 characters of the cleaned content
    
CURRENT_YEAR = 2026
def search_jobs(
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

    base_query = f'"{role}" {skill_text} ({location_text}) "{remote_preference}" "{seniority}" {salary_text}'

    queries = [
        base_query,
        f'site:greenhouse.io {base_query}',
        f'site:lever.co {base_query}',
        f'site:ashbyhq.com {base_query}',
        f'site:workdayjobs.com {base_query}',
        f'site:linkedin.com/jobs {base_query}',
    ]

    seen_links = set()
    all_results = []

    for query in queries:
        results = web_search({"query": query})

        for result in results:
            link = result.get("link") or result.get("url")
            if not link or link in seen_links:
                continue

            seen_links.add(link)

            all_results.append({
                "query": query,
                "title": result.get("title", "unknown"),
                "snippet": result.get("snippet") or result.get("body", ""),
                "link": link,
            })

    return all_results



def is_likely_job_result(result: dict) -> bool:
    text = f"{result.get('title', '')} {result.get('snippet', '')} {result.get('link', '')}".lower()

    include_terms = [
        "engineer",
        "developer",
        "full stack",
        "software",
        "python",
        "react",
        "remote",
        "jobs",
        "careers"
    ]

    exclude_terms = [
        "intern",
        "junior",
        "entry level",
        "course",
        "bootcamp"
    ]

    return any(term in text for term in include_terms) and not any(term in text for term in exclude_terms)

results = search_jobs(
    role="Senior Full Stack Engineer",
    skills=["Python", "React"],
    locations=["United States"],
    remote_preference="remote OR hybrid",
    min_salary=140000,
    seniority="senior OR mid-senior"
)

for result in results:
    if not is_likely_job_result(result):
        continue

    link = result["link"]
    content = scrape_web_page(link)
    cleaned_content = clean_scraped_text(content)

    if cleaned_content.startswith("[skipped"):
        continue

    normalized = service.ask(
        prompt=f"""
Search result metadata:
Title: {result["title"]}
Snippet: {result["snippet"]}
Link: {result["link"]}

Page content:
{cleaned_content[:12000]}
""",
        system_prompt=WEB_SEARCH_NORMALIZER_PROMPT,
    ).answer

    print(normalized)