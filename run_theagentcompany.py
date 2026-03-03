#!/usr/bin/env python3
"""
Run TheAgentCompany tasks that use LLM as judge, with the same MODELS and JUDGES
as run_experiment.py (OpenRouter). Saves results in a format compatible with
analyze_results.py: one score per (model, task_id, run_idx, judge).

Layout:
  - Generations: data/theagentcompany/generations/{model}/{task_id}/{run_idx}/traj.json
  - Judgements:  data/theagentcompany/judgements/{model}/{task_id}/{run_idx}/{judge}.json
    Each judgement JSON has top-level "score" (0–1 from result/total) for analyze_results.

Use .venv and set OPENROUTER_API_KEY. For generations you need TheAgentCompany services
and OpenHands (see docs). For judgements-only, provide existing trajectories.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_experiment import JUDGES, MODELS
from theagentcompany_llm_tasks import (
    LLM_TASK_NAMES,
    TASK_IMAGE_TAG,
    task_name_to_image,
)

# Same data root as run_experiment / run_mind_eval
DATA_DIR = Path("data")
TAC_ROOT = DATA_DIR / "theagentcompany"
GENERATIONS_ROOT = TAC_ROOT / "generations"
JUDGEMENTS_ROOT = TAC_ROOT / "judgements"

NUM_RUNS = 2  # generations per (model, task)
DECRYPTION_KEY = "theagentcompany is all you need"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _model_to_config_key(name: str) -> str:
    """TOML section [llm.X] treats dots as nested keys; use hyphens so lookup works."""
    return name.replace(".", "-")


def task_id_from_name(task_name: str) -> int:
    """Stable integer ID for directory layout (analyze_results expects int question_id)."""
    try:
        return LLM_TASK_NAMES.index(task_name)
    except ValueError:
        raise ValueError(f"Unknown LLM task: {task_name}")


def task_name_from_id(task_id: int) -> str:
    return LLM_TASK_NAMES[task_id]


def generation_dir(model: str, task_id: int, run_idx: int) -> Path:
    return GENERATIONS_ROOT / model / str(task_id) / str(run_idx)


def judgement_path(model: str, task_id: int, run_idx: int, judge: str) -> Path:
    return JUDGEMENTS_ROOT / model / str(task_id) / str(run_idx) / f"{judge}.json"


def openrouter_env_for_judge(judge_key: str) -> dict:
    """LiteLLM/OpenRouter env vars for a judge key from run_experiment JUDGES."""
    model_id = JUDGES.get(judge_key)
    if not model_id:
        raise ValueError(f"Unknown judge: {judge_key}")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    return {
        "LITELLM_API_KEY": api_key,
        "LITELLM_BASE_URL": OPENROUTER_BASE,
        "LITELLM_MODEL": model_id,
    }


def run_evaluator_docker(
    task_name: str,
    trajectory_path: Path,
    judge_key: str,
    result_path: Path,
    server_hostname: str = "localhost",
) -> bool:
    """Run the task's evaluator in Docker with the given judge LLM; write result to result_path."""
    image = task_name_to_image(task_name)
    env = openrouter_env_for_judge(judge_key)
    # Mount dirs so we can read trajectory and write result
    traj_dir = trajectory_path.parent
    out_dir = result_path.parent
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{traj_dir.resolve()}:/data:ro",
        "-v",
        f"{out_dir.resolve()}:/out",
        "-e",
        f"LITELLM_API_KEY={env['LITELLM_API_KEY']}",
        "-e",
        f"LITELLM_BASE_URL={env['LITELLM_BASE_URL']}",
        "-e",
        f"LITELLM_MODEL={env['LITELLM_MODEL']}",
        "-e",
        f"DECRYPTION_KEY={DECRYPTION_KEY}",
        "-e",
        f"SERVER_HOSTNAME={server_hostname}",
        "--network",
        "host",  # so container can reach RocketChat etc. on host
        image,
        "python_default",
        "/utils/eval.py",
        "--trajectory_path",
        "/data/" + trajectory_path.name,
        "--result_path",
        "/out/" + result_path.name,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  Evaluator failed: {e}", file=sys.stderr)
        return False


def score_from_result(result_path: Path) -> float | None:
    """Normalize result/total to 0–1; return None if missing."""
    if not result_path.exists():
        return None
    with open(result_path) as f:
        data = json.load(f)
    fs = data.get("final_score")
    if not isinstance(fs, dict):
        return None
    total = fs.get("total")
    result = fs.get("result")
    if total is None or result is None or total == 0:
        return None
    return result / total


def run_judgements(
    models: list[str],
    judges: list[str],
    task_ids: list[int],
    num_runs: int,
    server_hostname: str,
) -> None:
    """Run evaluator in Docker for each (model, task_id, run_idx, judge); save score per generation."""
    JUDGEMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    todo = []
    for model in models:
        for task_id in task_ids:
            task_name = task_name_from_id(task_id)
            for run_idx in range(num_runs):
                gen_dir = generation_dir(model, task_id, run_idx)
                traj_path = gen_dir / "traj.json"
                if not traj_path.exists():
                    continue
                for judge in judges:
                    jpath = judgement_path(model, task_id, run_idx, judge)
                    if jpath.exists():
                        continue
                    todo.append((model, task_id, run_idx, task_name, traj_path, judge, jpath))
    if not todo:
        n_traj = sum(
            1
            for m in models
            for t in task_ids
            for r in range(num_runs)
            if generation_dir(m, t, r).joinpath("traj.json").exists()
        )
        if n_traj == 0:
            print("Judgements: no trajectories found. Run --phase generations first or place traj.json under data/theagentcompany/generations/{model}/{task_id}/{run_idx}/")
        else:
            print("Judgements: all cached.")
        return
    print(f"Judgements: {len(todo)} to run")
    done = 0
    for model, task_id, run_idx, task_name, traj_path, judge, jpath in todo:
        jpath.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp result file then move, so Docker can write into mounted /out
        result_file = jpath.parent / "result.json"
        ok = run_evaluator_docker(
            task_name, traj_path, judge, result_file, server_hostname=server_hostname
        )
        if ok and result_file.exists():
            score = score_from_result(result_file)
            if score is not None:
                payload = {"score": score, "model": model, "task_id": task_id, "run_idx": run_idx, "judge": judge}
                with open(jpath, "w") as f:
                    json.dump(payload, f, indent=2)
                result_file.unlink(missing_ok=True)
                done += 1
        print(f"  {model} task={task_id} run={run_idx} judge={judge} -> {'ok' if ok else 'fail'}")
    print(f"  Done: {done} new judgements.")


def _write_config_toml(path: Path, models: list[str], judges: list[str]) -> None:
    """Write OpenRouter config.toml for [llm.<key>] sections. Keys have dots replaced by - for TOML."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required")
    lines = ["[core]", "run_as_openhands = false", ""]
    seen = set()
    for k in models + list(judges):
        if k in seen:
            continue
        seen.add(k)
        model_id = JUDGES.get(k) or MODELS.get(k)
        if not model_id:
            continue
        config_key = _model_to_config_key(k)
        lines.append(f'[llm.{config_key}]')
        lines.append(f'model = "openrouter/{model_id}"')
        lines.append(f'base_url = "{OPENROUTER_BASE}"')
        lines.append(f'api_key = "{api_key}"')
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def _find_poetry() -> str:
    """Return absolute path to poetry so it works under sudo."""
    import shutil
    p = shutil.which("poetry")
    if p:
        return p
    for candidate in [Path.home() / ".local/bin/poetry", Path("/usr/local/bin/poetry")]:
        if candidate.exists():
            return str(candidate)
    return "poetry"


def _fix_ownership(path: Path) -> None:
    """Best-effort chown of *path* tree back to the real (non-root) user."""
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if uid and gid:
        subprocess.run(
            ["chown", "-R", f"{uid}:{gid}", str(path)],
            capture_output=True,
        )
    elif os.getuid() == 0:
        pass  # already root with no SUDO context; leave as root
    else:
        subprocess.run(
            ["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(path)],
            capture_output=True,
        )


def run_generations_via_openhands(
    models: list[str],
    task_ids: list[int],
    num_runs: int,
    server_hostname: str,
    evaluation_dir: Path,
    config_toml: Path,
) -> None:
    """
    Run agent + evaluator once per (model, task_id, run_idx) using TheAgentCompany/evaluation/run_eval.py.
    Requires: poetry in evaluation_dir, config.toml with [llm.<model>] and [llm.<judge>], and services.

    OpenHands Docker runtime runs as root and changes mounted-directory ownership,
    so the subprocess must run as root (see TheAgentCompany/evaluation/README.md).
    We use sudo when not already root and chown results back afterward.
    """
    _write_config_toml(config_toml, models, list(JUDGES.keys()))

    repo_root = evaluation_dir.parent
    poetry_bin = _find_poetry()
    need_sudo = os.getuid() != 0
    first_judge = list(JUDGES.keys())[0]
    for model in models:
        for task_id in task_ids:
            task_name = task_name_from_id(task_id)
            image_name = task_name_to_image(task_name)
            for run_idx in range(num_runs):
                gen_dir = generation_dir(model, task_id, run_idx)
                if gen_dir.exists() and (gen_dir / "traj.json").exists():
                    continue
                gen_dir.mkdir(parents=True, exist_ok=True)
                inner_cmd = [
                    poetry_bin,
                    "run",
                    "python",
                    "evaluation/run_eval.py",
                    "--config-file",
                    str(config_toml.resolve()),
                    "--agent-llm-config",
                    _model_to_config_key(model),
                    "--env-llm-config",
                    _model_to_config_key(first_judge),
                    "--outputs-path",
                    str(gen_dir.resolve()),
                    "--server-hostname",
                    server_hostname,
                    "--task-image-name",
                    image_name,
                ]
                if need_sudo:
                    cmd = ["sudo", "-E", "env", f"PATH={os.environ.get('PATH', '')}"] + inner_cmd
                else:
                    cmd = inner_cmd
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        timeout=3600,
                    )
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(
                            result.returncode, cmd, result.stdout, result.stderr
                        )
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                    print(f"  Generation failed {model} {task_name} run={run_idx}: {e}", file=sys.stderr)
                    if isinstance(e, subprocess.CalledProcessError) and (e.stdout or e.stderr):
                        out = (e.stdout or "") + (e.stderr or "")
                        lines = out.strip().splitlines()
                        if len(lines) > 180:
                            head = "\n".join(lines[:60])
                            tail = "\n".join(lines[-120:])
                            print(
                                f"  First 60 lines:\n{head}\n  ... [{len(lines) - 180} lines omitted] ...\n  Last 120 lines:\n{tail}",
                                file=sys.stderr,
                            )
                        else:
                            print(f"  Output:\n{out.strip()}", file=sys.stderr)
                    if "Image build failed" in str(e) or "docker" in str(e).lower():
                        print(
                            f"  Hint: to see the Docker build error, run:\n"
                            f"    docker pull {image_name}\n"
                            f"  then from TheAgentCompany/: poetry run python evaluation/run_eval.py --config-file <path> --task-image-name {image_name} --build-image-only True",
                            file=sys.stderr,
                        )
                    out_for_hint = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
                    if out_for_hint and (
                        "Failed to connect" in out_for_hint
                        or "the-agent-company.com" in out_for_hint
                        or "Task environment init failed" in out_for_hint
                        or "Couldn't connect to server" in out_for_hint
                    ):
                        print(
                            "  Hint: TheAgentCompany services (RocketChat, GitLab, etc.) must be running on --server-hostname. See TheAgentCompany/docs/EVALUATION.md.",
                            file=sys.stderr,
                        )
                    continue
                finally:
                    _fix_ownership(gen_dir)
                # Normalize: copy traj_*.json to traj.json for judgements phase
                for f in gen_dir.glob("traj_*.json"):
                    (gen_dir / "traj.json").write_bytes(f.read_bytes())
                    break
    print("Generations (OpenHands) done. Run judgements phase to score with all judges.")


def main():
    parser = argparse.ArgumentParser(
        description="Run TheAgentCompany (LLM-judge subset) and save judgements for analyze_results"
    )
    parser.add_argument(
        "--phase",
        choices=["judgements", "generations", "both"],
        default="judgements",
        help="judgements: run evaluator only (need existing trajectories). generations: run agent via OpenHands. both: generations then judgements.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS.keys()),
        help="Model keys (default: all from run_experiment)",
    )
    parser.add_argument(
        "--judges",
        nargs="*",
        default=list(JUDGES.keys()),
        help="Judge keys (default: all from run_experiment)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help="Task IDs to run (default: all LLM tasks). Example: 0 1 2",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=NUM_RUNS,
        help=f"Runs per (model, task) (default: {NUM_RUNS})",
    )
    parser.add_argument(
        "--server-hostname",
        default="localhost",
        help="Hostname for RocketChat/GitLab etc. (default: localhost)",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "TheAgentCompany" / "evaluation",
        help="Path to TheAgentCompany/evaluation (poetry run from its parent)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.toml path (default: <evaluation-dir>/config.toml)",
    )
    args = parser.parse_args()

    task_ids = list(range(len(LLM_TASK_NAMES))) if args.tasks is None else args.tasks
    for tid in task_ids:
        if tid < 0 or tid >= len(LLM_TASK_NAMES):
            parser.error(f"Task ID {tid} out of range [0, {len(LLM_TASK_NAMES)-1}]")
    config_toml = args.config or (args.evaluation_dir / "config.toml")

    TAC_ROOT.mkdir(parents=True, exist_ok=True)

    if args.phase in ("generations", "both"):
        run_generations_via_openhands(
            args.models,
            task_ids,
            args.num_runs,
            args.server_hostname,
            args.evaluation_dir,
            config_toml,
        )
    if args.phase in ("judgements", "both"):
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("Warning: OPENROUTER_API_KEY not set; evaluator LLM calls may fail.")
        run_judgements(
            args.models,
            args.judges,
            task_ids,
            args.num_runs,
            args.server_hostname,
        )


if __name__ == "__main__":
    main()
