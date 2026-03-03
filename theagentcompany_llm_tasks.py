"""
List of TheAgentCompany tasks whose evaluator uses an LLM (for judge variance).
Derived from evaluators that use evaluate_with_llm, evaluate_chat_history_with_llm,
llm_complete, or compare_images_with_llm from common.
"""

# Task folder names (under workspaces/tasks/) that use LLM in evaluator
LLM_TASK_NAMES = [
    "admin-ask-for-upgrade-reimbursement",
    "admin-check-employees-budget-and-reply-2",
    # "admin-check-employees-budget-and-reply-and-record",
    # "ds-answer-numerical-data-question",
    # "ds-answer-spreadsheet-questions",
    # "ds-merge-multiple-sheets",
    # "ds-stock-analysis-slides",
    # "ds-visualize-data-in-pie-and-bar-chart",
    # "finance-apply-tax-credit",
    # "finance-create-10k-income-report",
    # "finance-find-signatories",
    # "finance-r-d-activities",
    # "finance-revenue-reconciliation",
    # "hr-analyze-outing-bills",
    # "hr-collect-feedbacks",
    # "hr-create-career-ladder",
    # "hr-create-employee-manual",
    # "hr-green-card-consultation",
    # "hr-internal-tooling-slides",
    # "hr-mass-survey",
    # "hr-massive-resume-screening",
    # "hr-new-grad-job-description",
    # "hr-new-grad-job-description-2",
    # "hr-new-grad-job-description-3",
    # "hr-pick-interviewer-1",
    # "hr-pick-interviewer-2",
    # "hr-pick-interviewer-3",
    # "hr-resume-screening",
    # "hr-transfer-group",
    # "ml-generate-gradcam",
    # "pm-create-teammate-channel-from-spreadsheet",
    # "pm-plan-personnel-for-new-project",
    # "pm-present-engineer-group-members",
    # "pm-schedule-meeting-1",
    # "pm-schedule-meeting-2",
    # "pm-send-notification-to-corresponding-user",
    # "research-answer-questions-on-paper",
    # "research-reproduce-figures",
    # "sde-add-wiki-page",
    # "sde-create-new-gitlab-project-logo",
    # "sde-create-new-repo",
    # "sde-find-api",
    # "sde-find-answer-in-codebase-1",
    # "sde-find-answer-in-codebase-3",
    # "sde-move-page-to-cloud",
    # "sde-pitch-idea-to-manager",
    # "sde-reply-community-issue-by-asking-npc",
    # "sde-reply-community-issue-with-fixed-reply",
    # "sde-report-agent-repos",
    # "sde-sotopia-dev-container",
    # "sde-sotopia-update-ci",
    # "sde-update-readme",
    # "sde-write-a-unit-test-for-scroll_down-function",
]

TASK_IMAGE_TAG = "1.0.0"


def task_name_to_image(task_name: str) -> str:
    return f"ghcr.io/theagentcompany/{task_name}-image:{TASK_IMAGE_TAG}"
