"""
Curated subset of TheAgentCompany tasks for studying LLM-judge variance.

Background
----------
48 of the 175 TheAgentCompany tasks use an LLM somewhere in their evaluator
(via evaluate_with_llm, evaluate_chat_history_with_llm, compare_images_with_llm,
or llm_complete).  However, many of those 48 use the LLM only for a small
fallback check or a near-deterministic fact look-up, so the judge's identity
has little practical effect on the final score.

This file keeps the subset where the LLM evaluator is a *significant* part of
the result—i.e. where swapping one judge model for another could plausibly
change the score—so we can properly measure judge bias and variance.

Inclusion criteria
------------------
A task is included when EITHER:

  * ≥ 50 % of the total checkpoint score is LLM-determined, OR
  * the LLM judgment is highly subjective (open-ended quality / intent) AND
    accounts for ≥ 40 % of the score.

27 of the 48 LLM-using tasks pass these thresholds.

Excluded tasks (21) and reasons
-------------------------------
Low LLM weight (< 33 % of score):
  admin-ask-for-upgrade-reimbursement      0–1 / 4  (LLM is a fallback only)
  hr-analyze-outing-bills                  1 / 7
  hr-mass-survey                           2 / 7
  pm-plan-personnel-for-new-project        1 / 7
  finance-create-10k-income-report         1 / 6
  hr-new-grad-job-description-3            1 / 5
  sde-pitch-idea-to-manager                1 / 5
  finance-find-signatories                 1 / 5
  hr-new-grad-job-description-2            1 / 4
  finance-revenue-reconciliation           1 / 4

LLM weight 33–39 % AND predicates are mostly objective / factual:
  admin-check-employees-budget-and-reply-and-record  2 / 6  – specific item removal list
  finance-r-d-activities                   2 / 6  – "asked about R&D hours" per employee
  finance-apply-tax-credit                 3 / 8  – whether two specific questions were asked
  ds-merge-multiple-sheets                 1 / 3  – "a proposal of merging the data"
  hr-green-card-consultation               1 / 3  – checks a specific date (01AUG23)
  hr-new-grad-job-description              1 / 3  – template-vs-requirement match
  hr-transfer-group                        1 / 3  – "the person works in the AI team"
  pm-present-engineer-group-members        1 / 3  – "introductory slides are finished"
  sde-move-page-to-cloud                   1 / 3  – near-exact text equivalence
  sde-reply-community-issue-with-fixed-reply  1 / 3  – reply vs fixed expected text
  sde-sotopia-update-ci                    1 / 3  – "CI for amd and x86 is added"

Ordering
--------
Included tasks are sorted by expected judge-bias potential: highly subjective
first, then moderately subjective (visual, then conversational / content),
then mostly-objective-but-high-LLM-weight (which serve as a low-bias control).
"""

# ---------------------------------------------------------------------------
# Tasks where the LLM evaluator is a significant part of the result.
#
# For each entry the comment shows:
#   (LLM pts / total pts, subjectivity level)
#
# Subjectivity levels
#   high     – open-ended quality / intent judgment (chart aesthetics, resume
#              qualification, "average performance", agreement interpretation)
#   moderate – semantic match or content-coverage check that still admits
#              reasonable disagreement between judges
#   low      – LLM is essentially doing fuzzy string / fact look-up, but
#              the score is ≥ 50 % LLM-determined so any variance matters
# ---------------------------------------------------------------------------

LLM_TASK_NAMES = [
    # -- highly subjective (text-only) -------------------------------------
    "hr-massive-resume-screening",  # 3/5,  high  – "Alex Chen is a qualified candidate"
    "hr-collect-feedbacks",  # 2/5,  high  – "average job performance"
    "hr-pick-interviewer-3",  # 4/4,  high  – "Emily Zhou agrees to interview"
    # -- moderately subjective (conversation / content) --------------------
    "hr-pick-interviewer-1",  # 6/6,  mod   – chat about interviewer selection
    "hr-pick-interviewer-2",  # 4/6,  mod   – availability conversation
    "pm-send-notification-to-corresponding-user",  # 4/4,  mod   – kickoff-meeting plan conveyed
    "sde-create-new-repo",  # 2/3,  mod   – "tasks to start a new project"
    "hr-internal-tooling-slides",  # 5/10, mod   – slide content coverage
    "sde-update-readme",  # 1/2,  mod   – "contents regarding contributing"
    "admin-check-employees-budget-and-reply-2",  # 2/4,  mod   – budget amounts + status in chat
    "pm-create-teammate-channel-from-spreadsheet",  # 2/5,  mod   – instruction messages to teammates
    "sde-find-answer-in-codebase-3",  # 2/5,  mod   – "conversation about llama3.1 context"
    "sde-reply-community-issue-by-asking-npc",  # 2/5,  mod   – reply matches issue content
    # -- mostly objective but ≥ 50 % LLM-determined -----------------------
    "sde-find-api",  # 4/4,  low   – URL pattern + query params
    "sde-find-answer-in-codebase-1",  # 3/3,  low   – "mentions PR 8676"
    "research-answer-questions-on-paper",  # 11/12, low  – factual Q&A vs reference
    "ds-answer-numerical-data-question",  # 6/6,  low   – specific numeric answers
    "pm-schedule-meeting-1",  # 3/5,  low   – "no meeting scheduled this week"
    "pm-schedule-meeting-2",  # 3/5,  low   – "meeting is scheduled on Friday"
    "ds-answer-spreadsheet-questions",  # 3/5,  low   – keyword presence
    "sde-sotopia-dev-container",  # 4/7,  low   – config content checks
    "hr-resume-screening",  # 2/4,  low   – visa-requirement conclusion
    "sde-report-agent-repos",  # 1/2,  low   – specific repo list
    # -- require VISION support from the judge (comment out for text-only judges)
    "ds-stock-analysis-slides",  # 7/8,  high  – slide images: chart quality + script
    "ds-visualize-data-in-pie-and-bar-chart",  # 4/4,  mod   – pie/bar chart images
    "ml-generate-gradcam",  # 2/4,  mod   – GradCAM heatmap image comparison
    "sde-create-new-gitlab-project-logo",  # 2/3,  mod   – project logo image: "letter S"
]

TASK_IMAGE_TAG = "1.0.0"


def task_name_to_image(task_name: str) -> str:
    return f"ghcr.io/theagentcompany/{task_name}-image:{TASK_IMAGE_TAG}"
