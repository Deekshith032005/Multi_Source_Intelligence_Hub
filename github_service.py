
import os

from github import Github
from dotenv import load_dotenv

load_dotenv()


def get_github_client():
    """Connect to GitHub, using a token if configured."""
    token = os.getenv("GITHUB_TOKEN")

    if token:
        return Github(token)

    # Allows unauthenticated access to public repositories
    return Github()


def get_repository_info(repo_name):
    """Retrieve basic information about a GitHub repository."""
    client = get_github_client()

    try:
        repo = client.get_repo(repo_name)

        return {
            "name": repo.full_name,
            "description": repo.description or "No description available",
            "url": repo.html_url,
            "stars": repo.stargazers_count,
            "language": repo.language or "Not specified",
            "open_issues": repo.open_issues_count,
        }

    finally:
        client.close()


if __name__ == "__main__":
    repository_name = input(
        "Enter a public GitHub repository (owner/repository): "
    ).strip()

    try:
        details = get_repository_info(repository_name)

        for key, value in details.items():
            print(f"{key}: {value}")

    except Exception as error:
        print("GitHub request failed:", error)
