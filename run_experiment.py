#!/usr/bin/env python3
"""
MT-Bench Variance Decomposition Experiment
==========================================
Proper two-turn MT-Bench evaluation via OpenRouter.
Follows the official FastChat/llm_judge protocol:
  - Two-turn generation (conversation context preserved across turns)
  - Per-turn judging with official single-v1 and single-v1-multi-turn prompts
  - Answers saved in FastChat JSONL format
  - Final MT-Bench score = average of turn 1 and turn 2 scores
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import aiohttp

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "gpt-5.2-chat": "openai/gpt-5.2-chat",
}

JUDGES = {
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "gpt-5.2": "openai/gpt-5.2",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "kimi-k2": "moonshotai/kimi-k2",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
}

NUM_GENERATIONS = 10
GEN_TEMPERATURE = 1.0
GEN_MAX_TOKENS = 2048
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 2048

CONCURRENCY = 10
MAX_RETRIES = 6

DATA_DIR = Path("data")
ANSWER_DIR = DATA_DIR / "mt_bench" / "model_answer"

# ── Official MT-Bench Judge Prompts (from FastChat judge_prompts.jsonl) ──────
# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl

JUDGE_PROMPTS = {
    "single-v1": {
        "system_prompt": "You are a helpful assistant.",
        "prompt_template": (
            "[Instruction]\n"
            "Please act as an impartial judge and evaluate the quality of the "
            "response provided by an AI assistant to the user question displayed "
            "below. Your evaluation should consider factors such as the "
            "helpfulness, relevance, accuracy, depth, creativity, and level of "
            "detail of the response. Begin your evaluation by providing a short "
            "explanation. Be as objective as possible. After providing your "
            "explanation, you must rate the response on a scale of 1 to 10 by "
            'strictly following this format: "[[rating]]", for example: '
            '"Rating: [[5]]".\n\n'
            "[Question]\n{question}\n\n"
            "[The Start of Assistant's Answer]\n{answer}\n"
            "[The End of Assistant's Answer]"
        ),
    },
    "single-v1-multi-turn": {
        "system_prompt": (
            "Please act as an impartial judge and evaluate the quality of the "
            "response provided by an AI assistant to the user question displayed "
            "below. Your evaluation should consider factors such as the "
            "helpfulness, relevance, accuracy, depth, creativity, and level of "
            "detail of the response. You evaluation should focus on the "
            "assistant's answer to the second user question. Begin your "
            "evaluation by providing a short explanation. Be as objective as "
            "possible. After providing your explanation, you must rate the "
            "response on a scale of 1 to 10 by strictly following this format: "
            '"[[rating]]", for example: "Rating: [[5]]".\n\n'
        ),
        "prompt_template": (
            "<|The Start of Assistant A's Conversation with User|>\n\n"
            "### User:\n{question_1}\n\n"
            "### Assistant A:\n{answer_1}\n\n"
            "### User:\n{question_2}\n\n"
            "### Assistant A:\n{answer_2}\n\n"
            "<|The End of Assistant A's Conversation with User|>"
        ),
    },
}

# ── Helpers ──────────────────────────────────────────────────────────────────


def extract_score(text):
    """Extract numeric score from judge response."""
    if not text or len(text.strip()) < 3:
        return None
    match = re.search(r"\[\[(\d+\.?\d*)\]\]", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:Rating|Score)\s*[:=]\s*(\d+\.?\d*)", text, re.I)
    if match:
        return float(match.group(1))
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*10\b", text)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:rating|score|grade)[:\s]+(?:is\s+)?(\d+\.?\d*)", text, re.I)
    if match:
        return float(match.group(1))
    match = re.search(r"\*\*(\d+(?:\.\d+)?)\*\*\s*/\s*10", text)
    if match:
        return float(match.group(1))
    last_line = text.strip().split("\n")[-1]
    match = re.search(r"\b([1-9]|10)(?:\.\d+)?\b", last_line)
    if match and len(last_line) < 50:
        return float(match.group(0))
    return None


def load_questions():
    q_path = DATA_DIR / "mt_bench_questions.jsonl"
    questions = []
    with open(q_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


class Progress:
    def __init__(self, total, label):
        self.total = total
        self.label = label
        self.done = 0
        self.failed = 0
        self.t0 = time.time()

    def tick(self, ok=True):
        if ok:
            self.done += 1
        else:
            self.failed += 1
        elapsed = time.time() - self.t0
        rate = (self.done + self.failed) / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done - self.failed) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r  [{self.label}] {self.done + self.failed}/{self.total}  "
            f"ok={self.done} err={self.failed}  "
            f"{elapsed:.0f}s elapsed  ~{eta:.0f}s left   "
        )
        sys.stdout.flush()

    def finish(self):
        elapsed = time.time() - self.t0
        print(
            f"\n  [{self.label}] Done: {self.done} ok, "
            f"{self.failed} failed in {elapsed:.1f}s"
        )


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── API ──────────────────────────────────────────────────────────────────────


async def api_call(session, sem, model, messages, temperature, max_tokens):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://variance-decomposition.research",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                timeout = aiohttp.ClientTimeout(total=180)
                async with session.post(
                    API_URL, json=payload, headers=headers, timeout=timeout
                ) as resp:
                    if resp.status == 429:
                        wait = float(
                            resp.headers.get("Retry-After", 2 ** (attempt + 1))
                        )
                        await asyncio.sleep(min(wait, 60))
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
                continue
            raise RuntimeError(f"Network error after {MAX_RETRIES} retries: {exc}")
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


# ── Phase 1: Two-Turn Generation ────────────────────────────────────────────


async def run_generations(session, sem, questions, gen_cache):
    """Generate two-turn conversations. Each entry is a list [turn1, turn2]."""
    task_list = []
    for mname, mid in MODELS.items():
        for q in questions:
            for g in range(NUM_GENERATIONS):
                gkey = f"{mname}|{q['question_id']}|{g}"
                if gkey in gen_cache:
                    continue
                task_list.append((mname, mid, q, g, gkey))

    if not task_list:
        print("  Generations: all cached")
        return

    print(f"  Generations: {len(task_list)} conversations "
          f"({len(task_list) * 2} API calls)")
    prog = Progress(len(task_list), "Gen")
    queue = asyncio.Queue()
    for t in task_list:
        queue.put_nowait(t)
    save_counter = [0]

    async def worker():
        while True:
            try:
                mname, mid, q, g, gkey = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                msgs = [{"role": "user", "content": q["turns"][0]}]
                t1 = await api_call(session, sem, mid, msgs,
                                    GEN_TEMPERATURE, GEN_MAX_TOKENS)
                msgs.append({"role": "assistant", "content": t1})
                msgs.append({"role": "user", "content": q["turns"][1]})
                t2 = await api_call(session, sem, mid, msgs,
                                    GEN_TEMPERATURE, GEN_MAX_TOKENS)
                gen_cache[gkey] = [t1, t2]
                prog.tick(True)
            except Exception:
                prog.tick(False)
            save_counter[0] += 1
            if save_counter[0] % 100 == 0:
                save_json(gen_cache, DATA_DIR / "generations.json")

    workers = [asyncio.create_task(worker()) for _ in range(CONCURRENCY)]
    await asyncio.gather(*workers)
    prog.finish()
    save_json(gen_cache, DATA_DIR / "generations.json")


# ── Save Answers in FastChat Format ──────────────────────────────────────────


def save_fastchat_answers(questions, gen_cache):
    """Write per-model JSONL files matching FastChat's answer format."""
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)

    for mname in MODELS:
        answer_file = ANSWER_DIR / f"{mname}.jsonl"
        with open(answer_file, "w") as fout:
            for q in questions:
                qid = q["question_id"]
                choices = []
                for g in range(NUM_GENERATIONS):
                    gkey = f"{mname}|{qid}|{g}"
                    turns = gen_cache.get(gkey, ["", ""])
                    choices.append({"index": g, "turns": turns})
                record = {
                    "question_id": qid,
                    "answer_id": str(uuid.uuid4()),
                    "model_id": mname,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(record) + "\n")
        print(f"  Saved {answer_file}")


# ── Phase 2: Per-Turn Judging ────────────────────────────────────────────────


async def run_judgments(session, sem, questions, gen_cache, jdg_cache):
    """Judge each generation on both turns with official MT-Bench prompts."""
    task_list = []
    for mname in MODELS:
        for q in questions:
            qid = q["question_id"]
            for g in range(NUM_GENERATIONS):
                gkey = f"{mname}|{qid}|{g}"
                turns = gen_cache.get(gkey)
                if not turns or len(turns) < 2:
                    continue
                for jname, jid in JUDGES.items():
                    for turn_num in [1, 2]:
                        jkey = f"{mname}|{qid}|{g}|{jname}|t{turn_num}"
                        if jkey in jdg_cache:
                            continue
                        task_list.append((mname, q, g, jname, jid,
                                          turn_num, turns, jkey))

    if not task_list:
        print("  Judgments: all cached")
        return

    print(f"  Judgments: {len(task_list)} API calls")
    prog = Progress(len(task_list), "Judge")
    queue = asyncio.Queue()
    for t in task_list:
        queue.put_nowait(t)
    save_counter = [0]

    async def worker():
        while True:
            try:
                mname, q, g, jname, jid, turn_num, turns, jkey = \
                    queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                if turn_num == 1:
                    pcfg = JUDGE_PROMPTS["single-v1"]
                    user_msg = pcfg["prompt_template"].format(
                        question=q["turns"][0], answer=turns[0])
                else:
                    pcfg = JUDGE_PROMPTS["single-v1-multi-turn"]
                    user_msg = pcfg["prompt_template"].format(
                        question_1=q["turns"][0], answer_1=turns[0],
                        question_2=q["turns"][1], answer_2=turns[1])

                msgs = [
                    {"role": "system", "content": pcfg["system_prompt"]},
                    {"role": "user", "content": user_msg},
                ]
                text = await api_call(session, sem, jid, msgs,
                                      JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
                score = extract_score(text)
                if score is None:
                    msgs += [
                        {"role": "assistant", "content": text},
                        {"role": "user",
                         "content": 'Please provide your rating as "[[X]]".'},
                    ]
                    text2 = await api_call(session, sem, jid, msgs,
                                           JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
                    score = extract_score(text2)
                    text = text + "\n" + text2
                if score is not None:
                    jdg_cache[jkey] = {"score": score, "reasoning": text}
                    prog.tick(True)
                else:
                    prog.tick(False)
            except Exception:
                prog.tick(False)
            save_counter[0] += 1
            if save_counter[0] % 100 == 0:
                save_json(jdg_cache, DATA_DIR / "judgments.json")

    workers = [asyncio.create_task(worker()) for _ in range(CONCURRENCY)]
    await asyncio.gather(*workers)
    prog.finish()
    save_json(jdg_cache, DATA_DIR / "judgments.json")


# ── Main ─────────────────────────────────────────────────────────────────────


def select_questions(questions, n_questions):
    """Pick n_questions, one per category first, then fill randomly."""
    if n_questions >= len(questions):
        return questions
    by_cat = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q)
    selected = []
    for cat in sorted(by_cat):
        selected.append(by_cat[cat][0])
        if len(selected) >= n_questions:
            break
    remaining = [q for q in questions if q not in selected]
    import random

    random.seed(0)
    random.shuffle(remaining)
    selected.extend(remaining[: n_questions - len(selected)])
    return selected[:n_questions]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=int,
        default=80,
        help="Number of MT-Bench questions (default: 80)",
    )
    parser.add_argument(
        "--generations", type=int, default=None, help="Override NUM_GENERATIONS"
    )
    args = parser.parse_args()

    global NUM_GENERATIONS
    if args.generations is not None:
        NUM_GENERATIONS = args.generations

    if not API_KEY:
        print("Error: set OPENROUTER_API_KEY")
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    all_questions = load_questions()
    questions = select_questions(all_questions, args.questions)

    n_gen_calls = len(questions) * len(MODELS) * NUM_GENERATIONS * 2
    n_jdg_calls = len(questions) * len(MODELS) * NUM_GENERATIONS * len(JUDGES) * 2
    print(
        f"Using {len(questions)}/{len(all_questions)} questions, "
        f"{NUM_GENERATIONS} gens, {len(JUDGES)} judges"
    )
    print(
        f"API calls: ~{n_gen_calls} gen (2 per conv) "
        f"+ ~{n_jdg_calls} judge (2 turns each) "
        f"= ~{n_gen_calls + n_jdg_calls} total"
    )

    gen_cache = load_json(DATA_DIR / "generations.json")
    jdg_cache = load_json(DATA_DIR / "judgments.json")
    print(f"Cache: {len(gen_cache)} generations, {len(jdg_cache)} judgments")

    conn = aiohttp.TCPConnector(limit=CONCURRENCY + 5)
    async with aiohttp.ClientSession(connector=conn) as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        print("\n=== Phase 1: Two-Turn Generations ===")
        await run_generations(session, sem, questions, gen_cache)

        print("\n=== Saving answers in FastChat format ===")
        save_fastchat_answers(questions, gen_cache)

        print("\n=== Phase 2: Judge Evaluations (both turns) ===")
        await run_judgments(session, sem, questions, gen_cache, jdg_cache)

    # Build final dataset: score = average(turn1, turn2) per standard MT-Bench
    print("\n=== Building final dataset ===")
    records = []
    for mname in MODELS:
        for q in questions:
            qid = q["question_id"]
            for g in range(NUM_GENERATIONS):
                gkey = f"{mname}|{qid}|{g}"
                for jname in JUDGES:
                    jk1 = f"{mname}|{qid}|{g}|{jname}|t1"
                    jk2 = f"{mname}|{qid}|{g}|{jname}|t2"
                    s1 = jdg_cache.get(jk1, {}).get("score")
                    s2 = jdg_cache.get(jk2, {}).get("score")
                    if s1 is not None and s2 is not None:
                        score = (s1 + s2) / 2.0
                    elif s1 is not None:
                        score = s1
                    elif s2 is not None:
                        score = s2
                    else:
                        score = None
                    records.append(
                        {
                            "model": mname,
                            "question_id": qid,
                            "category": q["category"],
                            "gen_idx": g,
                            "judge": jname,
                            "score": score,
                            "score_turn1": s1,
                            "score_turn2": s2,
                        }
                    )

    save_json(records, DATA_DIR / "all_results.json")
    valid = sum(1 for r in records if r["score"] is not None)
    print(f"  {len(records)} records total, {valid} valid scores")
    print(f"  Saved to {DATA_DIR / 'all_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
