
import os

from dotenv import load_dotenv
from github import Auth, Github

load_dotenv()


def get_github_client():
    """Connect to GitHub using a token if configured."""
    token = os.getenv("GITHUB_TOKEN")

    if token:
        return Github(auth=Auth.Token(token))

    # Allows access to public repositories without a token.
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


def get_repository_issues(repo_name):
    """Retrieve open issues, excluding pull requests."""
    client = get_github_client()

    try:
        repo = client.get_repo(repo_name)
        issues = []

        for issue in repo.get_issues(state="open"):
            if issue.pull_request is not None:
                continue

            issues.append({
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "url": issue.html_url,
                "created_at": issue.created_at.isoformat(),
                "body": issue.body or "No description provided",
            })

        return issues

    finally:
        client.close()


if __name__ == "__main__":
    repository_name = input(
        "Enter a GitHub repository (owner/repository): "
    ).strip()

    try:
        details = get_repository_info(repository_name)

        print("\n--- Repository Information ---")
        for key, value in details.items():
            print(f"{key}: {value}")

        issues = get_repository_issues(repository_name)

        print(f"\n--- Open Issues: {len(issues)} ---")

        for issue in issues:
            print(f"\n#{issue['number']}: {issue['title']}")
            print(f"URL: {issue['url']}")
            print(f"Created: {issue['created_at']}")

    except Exception as error:
        print("GitHub request failed:", error)
