#!/usr/bin/env python3
"""
Run Mind-Eval (simulated clinician–member conversations) and cache results
in the same format/structure as run_experiment.py so analyze_results.py can
run the same variance decomposition analysis.

Uses OpenRouter for all LLM calls and the same MODELS / JUDGES as run_experiment.
Writes to data/mind_eval/generations and data/mind_eval/judgements (same layout as
mt_bench so analyze_results can run with --data-dir data/mind_eval).
"""

import json
import os
import sys
from pathlib import Path

# Allow importing mind-eval from subfolder
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "mind-eval"))

# Suppress litellm "Provider List" warning (non-critical) before any mind-eval import
import litellm

litellm.suppress_debug_info = True

from run_experiment import DATA_DIR, JUDGES, MODELS

# Mind-eval cache: separate from mt_bench
GENERATIONS_ROOT = DATA_DIR / "mind_eval" / "generations"
JUDGEMENTS_ROOT = DATA_DIR / "mind_eval" / "judgements"

# Mind-eval config (match mind-eval/run_benchmark.sh: data/profiles.jsonl, custom clinician, v0_2 member, 10 turns)
# We evaluate the clinician; member (client) is fixed (see run_benchmark.sh: clinician=$1, member=fixed).
NUM_GENERATIONS = 5
N_TURNS = 10
PROFILES_PATH = (
    DATA_DIR / "profiles.jsonl"
)  # same as run_benchmark.sh --profiles_path data/profiles.jsonl
CLINICIAN_TEMPLATE_VERSION = "custom"
MEMBER_TEMPLATE_VERSION = "v0_2"
JUDGE_TEMPLATE_VERSION = "v0_1"
# Fixed model for simulated member (client); clinician varies per MODELS (evaluated).
# OpenRouter model ID (no "openrouter/" prefix).
MEMBER_MODEL_OPENROUTER_ID = "anthropic/claude-haiku-4.5"
MAX_WORKERS_INTERACTION = 5
MAX_WORKERS_JUDGE = 5


def openrouter_api_params(model_key: str, is_judge: bool = False) -> dict:
    """Build litellm api_params for OpenRouter. Relies on OPENROUTER_API_KEY in env."""
    mapping = JUDGES if is_judge else MODELS
    openrouter_id = mapping.get(model_key)
    if not openrouter_id:
        raise ValueError(f"Unknown model key: {model_key}")
    return {
        "model": f"openrouter/{openrouter_id}",
        "max_tokens": 2048,
    }


def ensure_mind_eval_path():
    """Ensure mind-eval is on path and can be imported."""
    if "mindeval" not in sys.modules:
        import mindeval  # noqa: F401


def load_profiles():
    """Load profiles from data/profiles.jsonl (same as run_benchmark.sh --profiles_path data/profiles.jsonl)."""
    if not PROFILES_PATH.exists():
        print(f"Error: profiles not found at {PROFILES_PATH}", flush=True)
        print(
            "  Use data/profiles.jsonl from mind-eval (same as run_benchmark.sh).",
            flush=True,
        )
        sys.exit(1)
    ensure_mind_eval_path()
    from mindeval.utils import load_jsonl

    rows = load_jsonl(str(PROFILES_PATH))
    return [
        {
            "member_attributes": r["member_attributes"],
            "member_narrative": r["member_narrative"],
        }
        for r in rows
    ]


def run_one_interaction(
    profile_dict,
    clinician_engine,
    member_engine,
    clinician_template,
    member_template,
):
    """Run one multi-turn interaction; return interaction messages (list of dicts with role/content)."""
    ensure_mind_eval_path()
    from mindeval.scripts.generate_interactions import run_interactions

    profile = {
        **profile_dict["member_attributes"],
        "member_narrative": profile_dict["member_narrative"],
    }
    clinician_messages, _, _ = run_interactions(
        profile,
        clinician_template,
        clinician_engine,
        member_template,
        member_engine,
        N_TURNS,
    )
    return clinician_messages


def generation_cache_path(model_name, question_id, index):
    return GENERATIONS_ROOT / model_name / str(question_id) / f"{index}.json"


def evaluation_cache_path(model_name, question_id, index, judge_model):
    return (
        JUDGEMENTS_ROOT
        / model_name
        / str(question_id)
        / str(index)
        / f"{judge_model}.json"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Mind-Eval and cache in mt_bench format for analyze_results.py"
    )
    parser.add_argument(
        "--profiles",
        type=int,
        default=None,
        help="Use first N profiles from data/profiles.jsonl (default: all)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=NUM_GENERATIONS,
        help="Generations per model/profile (default 5)",
    )
    parser.add_argument(
        "--skip-interactions",
        action="store_true",
        help="Skip interactions (use existing generations)",
    )
    parser.add_argument(
        "--skip-judgments", action="store_true", help="Skip judgments (use existing)"
    )
    args = parser.parse_args()

    num_generations = args.generations

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: set OPENROUTER_API_KEY")
        sys.exit(1)
    # So litellm uses its OpenRouter provider (avoids "Provider List" / provider not found)
    os.environ.setdefault("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

    ensure_mind_eval_path()
    from mindeval.inference import InferenceEngine
    from mindeval.judge_prompts import JUDGE_PROMPT_TEMPLATE_VERSION_DICT
    from mindeval.prompts import (
        INTERACTION_CLINICIAN_VERSION_DICT,
        INTERACTION_MEMBER_VERSION_DICT,
    )
    from mindeval.utils import messages_to_convo_str, parse_judge_scores
    from tqdm import tqdm

    DATA_DIR.mkdir(exist_ok=True)
    GENERATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    JUDGEMENTS_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) Profiles (same as run_benchmark.sh: data/profiles.jsonl)
    print("=== Profiles ===", flush=True)
    print(f"  Loading from {PROFILES_PATH} ...", flush=True)
    profiles = load_profiles()
    if args.profiles is not None:
        profiles = profiles[: args.profiles]
        print(
            f"  Using first {len(profiles)} profiles (--profiles={args.profiles})",
            flush=True,
        )
    else:
        print(f"  Loaded {len(profiles)} profiles", flush=True)

    # Fixed member (client) engine; clinician varies per model under evaluation.
    member_engine = InferenceEngine(
        api_params={
            "model": f"openrouter/{MEMBER_MODEL_OPENROUTER_ID}",
            "max_tokens": 2048,
        }
    )
    clinician_template = INTERACTION_CLINICIAN_VERSION_DICT[CLINICIAN_TEMPLATE_VERSION]
    member_template = INTERACTION_MEMBER_VERSION_DICT[MEMBER_TEMPLATE_VERSION]

    # 2) Interactions (generations)
    if not args.skip_interactions:
        print("\n=== Generations (simulated conversations) ===", flush=True)
        # Build flat task list for one progress bar
        gen_tasks = [
            (mk, pid, g)
            for mk in MODELS
            for pid in range(len(profiles))
            for g in range(num_generations)
            if not generation_cache_path(mk, pid, g).exists()
        ]
        if not gen_tasks:
            print("  All generations already cached.", flush=True)
        else:
            print(f"  {len(gen_tasks)} conversations to run", flush=True)
        pbar = tqdm(gen_tasks, desc="  Generations", unit="conv")
        current_model = None
        clinician_engine = None
        for model_key, profile_id, g in pbar:
            pbar.set_postfix_str(f"{model_key} p{profile_id} g{g}")
            path = generation_cache_path(model_key, profile_id, g)
            if path.exists():
                continue
            if model_key != current_model:
                current_model = model_key
                clinician_engine = InferenceEngine(
                    api_params=openrouter_api_params(model_key, is_judge=False)
                )
            profile_dict = profiles[profile_id]
            try:
                interaction = run_one_interaction(
                    profile_dict,
                    clinician_engine,
                    member_engine,
                    clinician_template,
                    member_template,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "model_id": model_key,
                    "question_id": profile_id,
                    "index": g,
                    "interaction": interaction,
                    "member_profile": {
                        **profile_dict["member_attributes"],
                        "member_narrative": profile_dict["member_narrative"],
                    },
                }
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                tqdm.write(f"  Error {model_key} profile {profile_id} gen {g}: {e}")
        print("  Generations done.", flush=True)
    else:
        print("\n=== Generations (skipped) ===")

    # 3) Judgments
    if not args.skip_judgments:
        print("\n=== Judgments ===", flush=True)
        judge_prompt_template = JUDGE_PROMPT_TEMPLATE_VERSION_DICT[
            JUDGE_TEMPLATE_VERSION
        ]
        task_list = []
        for model_key in MODELS:
            for profile_id in range(len(profiles)):
                for g in range(num_generations):
                    path = generation_cache_path(model_key, profile_id, g)
                    if not path.exists():
                        continue
                    for judge_key in JUDGES:
                        out_path = evaluation_cache_path(
                            model_key, profile_id, g, judge_key
                        )
                        if out_path.exists():
                            continue
                        task_list.append((model_key, profile_id, g, judge_key, path))
        if not task_list:
            print("  All judgments already cached.", flush=True)
        else:
            print(f"  {len(task_list)} judgment tasks", flush=True)
        for model_key, profile_id, g, judge_key, path in tqdm(
            task_list, desc="  Judging"
        ):
            try:
                with open(path) as f:
                    gen = json.load(f)
                interaction = gen["interaction"]
                profile = gen["member_profile"]
                convo_str = messages_to_convo_str(interaction)
                user_prompt = judge_prompt_template.substitute(
                    conversation_str=convo_str, **profile
                )
                judge_engine = InferenceEngine(
                    api_params=openrouter_api_params(judge_key, is_judge=True)
                )
                messages = [{"role": "user", "content": user_prompt}]
                unparsed, _ = judge_engine.generate_with_thinking(messages)
                parsed = parse_judge_scores(unparsed)[0]
                overall = parsed.get("Overall score") or parsed.get("Average score")
                if overall is None:
                    criteria = [
                        "Clinical Accuracy & Competence",
                        "Ethical & Professional Conduct",
                        "Assessment & Response",
                        "Therapeutic Relationship & Alliance",
                        "AI-Specific Communication Quality",
                    ]
                    overall = sum(parsed.get(c, 3) for c in criteria) / len(criteria)
                # Full result: overall score, all dimension scores, raw reasoning
                out_obj = {
                    "score": float(overall),
                    "reasoning": unparsed or "",
                    "scores": {
                        "Clinical Accuracy & Competence": parsed.get(
                            "Clinical Accuracy & Competence", 3
                        ),
                        "Ethical & Professional Conduct": parsed.get(
                            "Ethical & Professional Conduct", 3
                        ),
                        "Assessment & Response": parsed.get(
                            "Assessment & Response", 3
                        ),
                        "Therapeutic Relationship & Alliance": parsed.get(
                            "Therapeutic Relationship & Alliance", 3
                        ),
                        "AI-Specific Communication Quality": parsed.get(
                            "AI-Specific Communication Quality", 3
                        ),
                    },
                }
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(out_obj, f, indent=2)
            except Exception as e:
                print(f"  Error judge {judge_key} {model_key} p{profile_id} g{g}: {e}")
        print("  Judgments done.", flush=True)
    else:
        print("\n=== Judgments (skipped) ===")

    print("\nDone. Cache: data/mind_eval/generations and data/mind_eval/judgements")
    print("Run analysis: python analyze_results.py --benchmark mind_eval")


if __name__ == "__main__":
    main()
