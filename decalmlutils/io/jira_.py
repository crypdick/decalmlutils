import logging
from datetime import datetime

from beartype import beartype
from beartype.typing import Dict, Optional
from jira import JIRA

from decalmlutils.io.aws.secretsmanager import get_secret

logger = logging.getLogger(__name__)

JIRA_URL = "https://jira.foo.com"
JIRA_TOKEN_SECRET_NAME = "jira-token"


@beartype
def add_issues_to_active_sprint(
    jira: JIRA, board_id: int, issue_keys: list[str]
) -> Optional[str]:
    now = datetime.now()
    started_sprints = list()
    for sprint in jira.sprints(board_id, state="active"):
        # account for sprints with no start date
        if start_date := getattr(sprint, "startDate", None):
            start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%fZ")
            if start_date < now:
                started_sprints.append(sprint)

    # Exit early and warn if no active sprints found
    if not started_sprints:
        logger.warning("No active sprints found")
        return

    # some sprints do not have an end date
    sprint_with_latest_end_date = max(
        started_sprints, key=lambda x: getattr(x, "endDate", "0")
    )
    logger.info(
        f"Started sprints: {started_sprints} Now: {now} Latest: {sprint_with_latest_end_date} Board ID: {board_id} Client: {jira.client_info()}"
    )
    logger.info(f"Adding issues to sprint `{sprint_with_latest_end_date.name}`")
    jira.add_issues_to_sprint(
        sprint_id=sprint_with_latest_end_date.id, issue_keys=issue_keys
    )
    return sprint_with_latest_end_date.name


@beartype
def get_or_create_epic(
    jira_project_key: str, summary: str, jira: JIRA = None, **kwargs
):
    """

    Args:
        jira_project_key: Jira project key. ex "MLDC"
        summary:
        jira:
        **kwargs:

    Returns:

    """
    creds = get_secret(JIRA_TOKEN_SECRET_NAME, jsonify=True)
    jira = jira or JIRA(JIRA_URL, basic_auth=tuple(creds.values()))

    kwargs.pop("issuetype", None)

    for epic in jira.search_issues(f"project={jira_project_key} and issueType=Epic"):
        if epic.fields.summary == summary:
            logger.info(f"Found existing epic `{summary}`")
            return epic

    epic = jira.create_issue(
        fields={
            "issuetype": {"name": "Epic"},
            "project": jira_project_key,
            "summary": summary,
            **kwargs,
        }
    )
    logger.info(f"Created epic `{summary}`")

    return epic


@beartype
def create_basic_table(fields_dict: Dict):
    rows = ["||*Item*||*Value*||"]
    rows.extend([f"|{k} | {v} |" for k, v in fields_dict.items()])
    table = "\n".join(rows)
    return table


@beartype
def layer_id_to_labeller_url(base_url: str, layer_id: int) -> str:
    return f"{base_url.rstrip('/')}/internal/labeller/freestyle/{layer_id}"
