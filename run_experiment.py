#!/usr/bin/env python3
"""
MT-Bench Variance Decomposition Experiment
==========================================
Generates responses from models and collects judge scores via OpenRouter.
Supports checkpointing and resume.
"""
import argparse
import asyncio
import aiohttp
import json
import os
import re
import sys
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "llama-3.3-70b":     "meta-llama/llama-3.3-70b-instruct",
    "gpt-5.2-chat":      "openai/gpt-5.2-chat",
}

JUDGES = {
    "claude-opus-4.6": "anthropic/claude-opus-4.6",
    "gpt-5.2":         "openai/gpt-5.2",
    "gemini-3-pro":    "google/gemini-3-pro-preview",
    "kimi-k2.5":       "moonshotai/kimi-k2.5",
    "glm-5":           "z-ai/glm-5",
}

NUM_GENERATIONS = 10
GEN_TEMPERATURE = 1.0
GEN_MAX_TOKENS = 2048
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 2048

CONCURRENCY = 10
MAX_RETRIES = 6

DATA_DIR = Path("data")

JUDGE_SYSTEM_PROMPT = "You are a helpful assistant."

JUDGE_USER_TEMPLATE = (
    "[Instruction]\n"
    "Please act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your "
    "evaluation should consider factors such as the helpfulness, relevance, "
    "accuracy, depth, creativity, and level of detail of the response. Begin "
    "your evaluation by providing a short explanation. Be as objective as "
    "possible. After providing your explanation, you must rate the response on "
    'a scale of 1 to 10 by strictly following this format: "[[rating]]", for '
    'example: "Rating: [[5]]".\n\n'
    "[Question]\n{question}\n\n"
    "[The Start of Assistant's Answer]\n{answer}\n"
    "[The End of Assistant's Answer]"
)

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
                q = json.loads(line)
                questions.append({
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "question": q["turns"][0],
                })
    return questions


class Progress:
    def __init__(self, total, label):
        self.total = total
        self.label = label
        self.done = 0
        self.failed = 0
        self.t0 = time.time()

    def tick(self, ok=True):
        self.done += 1 if ok else 0
        self.failed += 0 if ok else 1
        elapsed = time.time() - self.t0
        rate = (self.done + self.failed) / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done - self.failed) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r  [{self.label}] {self.done+self.failed}/{self.total}  "
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
                        wait = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                        await asyncio.sleep(min(wait, 60))
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Network error after {MAX_RETRIES} retries: {exc}")
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")

# ── Generation Phase ─────────────────────────────────────────────────────────

async def run_generations(session, sem, questions, cache):
    tasks_info = []
    for mname, mid in MODELS.items():
        for q in questions:
            for g in range(NUM_GENERATIONS):
                key = f"{mname}|{q['question_id']}|{g}"
                if key not in cache:
                    tasks_info.append((mname, mid, q, g, key))

    if not tasks_info:
        print("  Generations: all cached")
        return cache

    print(f"  Generations: {len(tasks_info)} API calls")
    prog = Progress(len(tasks_info), "Gen")

    async def do(mname, mid, q, g, key):
        msgs = [{"role": "user", "content": q["question"]}]
        try:
            resp = await api_call(session, sem, mid, msgs, GEN_TEMPERATURE, GEN_MAX_TOKENS)
            prog.tick(True)
            return key, resp
        except Exception as exc:
            prog.tick(False)
            return key, f"__ERROR__: {exc}"

    results = await asyncio.gather(*(do(*t) for t in tasks_info), return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            continue
        k, v = r
        if not v.startswith("__ERROR__"):
            cache[k] = v

    prog.finish()
    save_json(cache, DATA_DIR / "generations.json")
    return cache

# ── Judging Phase ────────────────────────────────────────────────────────────

async def run_judgments(session, sem, questions, gens, cache):
    tasks_info = []
    qmap = {q["question_id"]: q for q in questions}

    for mname in MODELS:
        for q in questions:
            qid = q["question_id"]
            for g in range(NUM_GENERATIONS):
                gkey = f"{mname}|{qid}|{g}"
                resp = gens.get(gkey, "")
                if not resp or resp.startswith("__ERROR__"):
                    continue
                for jname, jid in JUDGES.items():
                    jkey = f"{mname}|{qid}|{g}|{jname}"
                    if jkey not in cache:
                        tasks_info.append((mname, q, g, jname, jid, jkey, resp))

    if not tasks_info:
        print("  Judgments: all cached")
        return cache

    print(f"  Judgments: {len(tasks_info)} API calls")
    prog = Progress(len(tasks_info), "Judge")

    async def do(mname, q, g, jname, jid, jkey, answer):
        user_msg = JUDGE_USER_TEMPLATE.format(question=q["question"], answer=answer)
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            text = await api_call(session, sem, jid, msgs, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
            score = extract_score(text)
            if score is None:
                msgs += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Provide your rating as [[X]]."},
                ]
                text2 = await api_call(session, sem, jid, msgs, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
                score = extract_score(text2)
                text = text + "\n" + text2
            prog.tick(True)
            return jkey, {"score": score, "reasoning": text}
        except Exception as exc:
            prog.tick(False)
            return jkey, {"score": None, "reasoning": f"__ERROR__: {exc}"}

    results = await asyncio.gather(*(do(*t) for t in tasks_info), return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            continue
        k, v = r
        if v.get("score") is not None:
            cache[k] = v

    prog.finish()
    save_json(cache, DATA_DIR / "judgments.json")
    return cache

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
    import random; random.seed(0)
    random.shuffle(remaining)
    selected.extend(remaining[: n_questions - len(selected)])
    return selected[:n_questions]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=80,
                        help="Number of MT-Bench questions to use (default: 80)")
    parser.add_argument("--generations", type=int, default=None,
                        help="Override NUM_GENERATIONS")
    args = parser.parse_args()

    global NUM_GENERATIONS
    if args.generations is not None:
        NUM_GENERATIONS = args.generations

    if not API_KEY:
        print("Error: set OPENROUTER_API_KEY"); sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    all_questions = load_questions()
    questions = select_questions(all_questions, args.questions)
    n_gen = len(questions) * len(MODELS) * NUM_GENERATIONS
    n_jdg = n_gen * len(JUDGES)
    print(f"Using {len(questions)}/{len(all_questions)} questions, "
          f"{NUM_GENERATIONS} gens, {len(JUDGES)} judges")
    print(f"Estimated API calls: ~{n_gen} gen + ~{n_jdg} judge = ~{n_gen + n_jdg}")

    gen_cache = load_json(DATA_DIR / "generations.json")
    jdg_cache = load_json(DATA_DIR / "judgments.json")
    print(f"Cache: {len(gen_cache)} generations, {len(jdg_cache)} judgments")

    stats = {"gen_ok": 0, "gen_err": 0, "gen_cached": 0,
             "jdg_ok": 0, "jdg_err": 0, "jdg_cached": 0, "api_done": 0}
    t0 = time.time()

    def print_progress():
        elapsed = time.time() - t0
        rate = stats["api_done"] / elapsed if elapsed > 0 else 0
        remaining = (n_gen + n_jdg - stats["gen_cached"] - stats["jdg_cached"]
                     - stats["api_done"])
        eta = remaining / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r  gen={stats['gen_ok']}ok/{stats['gen_err']}err/{stats['gen_cached']}cache  "
            f"jdg={stats['jdg_ok']}ok/{stats['jdg_err']}err/{stats['jdg_cached']}cache  "
            f"api={stats['api_done']}  {elapsed:.0f}s  ~{eta:.0f}s left   "
        )
        sys.stdout.flush()

    save_counter = [0]

    def maybe_checkpoint():
        save_counter[0] += 1
        if save_counter[0] % 100 == 0:
            save_json(gen_cache, DATA_DIR / "generations.json")
            save_json(jdg_cache, DATA_DIR / "judgments.json")

    conn = aiohttp.TCPConnector(limit=CONCURRENCY + 5)
    async with aiohttp.ClientSession(connector=conn) as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def judge_one(mname, q, g, jname, jid, response):
            jkey = f"{mname}|{q['question_id']}|{g}|{jname}"
            if jkey in jdg_cache:
                stats["jdg_cached"] += 1
                return
            user_msg = JUDGE_USER_TEMPLATE.format(
                question=q["question"], answer=response)
            msgs = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            try:
                text = await api_call(session, sem, jid, msgs,
                                      JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
                score = extract_score(text)
                if score is None:
                    msgs += [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": "Provide your rating as [[X]]."},
                    ]
                    text2 = await api_call(session, sem, jid, msgs,
                                           JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
                    score = extract_score(text2)
                    text = text + "\n" + text2
                    stats["api_done"] += 1
                if score is not None:
                    jdg_cache[jkey] = {"score": score, "reasoning": text}
                    stats["jdg_ok"] += 1
                else:
                    stats["jdg_err"] += 1
            except Exception:
                stats["jdg_err"] += 1
            stats["api_done"] += 1
            print_progress()
            maybe_checkpoint()

        async def process_one(mname, mid, q, g):
            gkey = f"{mname}|{q['question_id']}|{g}"
            if gkey in gen_cache:
                response = gen_cache[gkey]
                stats["gen_cached"] += 1
            else:
                msgs = [{"role": "user", "content": q["question"]}]
                try:
                    response = await api_call(session, sem, mid, msgs,
                                              GEN_TEMPERATURE, GEN_MAX_TOKENS)
                    gen_cache[gkey] = response
                    stats["gen_ok"] += 1
                except Exception:
                    stats["gen_err"] += 1
                    stats["api_done"] += 1
                    print_progress()
                    return
                stats["api_done"] += 1
                print_progress()
                maybe_checkpoint()

            if not response or response.startswith("__ERROR__"):
                return

            await asyncio.gather(*(
                judge_one(mname, q, g, jn, ji, response)
                for jn, ji in JUDGES.items()
            ))

        print(f"\nLaunching parallel gen+judge pipeline "
              f"(concurrency={CONCURRENCY})...")
        tasks = []
        for mname, mid in MODELS.items():
            for q in questions:
                for g in range(NUM_GENERATIONS):
                    tasks.append(process_one(mname, mid, q, g))
        await asyncio.gather(*tasks)

    save_json(gen_cache, DATA_DIR / "generations.json")
    save_json(jdg_cache, DATA_DIR / "judgments.json")

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.0f}s! "
          f"gen={stats['gen_ok']}ok/{stats['gen_err']}err  "
          f"jdg={stats['jdg_ok']}ok/{stats['jdg_err']}err")

    print("\n=== Building final dataset ===")
    records = []
    for mname in MODELS:
        for q in questions:
            qid = q["question_id"]
            for g in range(NUM_GENERATIONS):
                gkey = f"{mname}|{qid}|{g}"
                for jname in JUDGES:
                    jkey = f"{mname}|{qid}|{g}|{jname}"
                    jdg = jdg_cache.get(jkey, {})
                    records.append({
                        "model": mname,
                        "question_id": qid,
                        "category": q["category"],
                        "gen_idx": g,
                        "judge": jname,
                        "score": jdg.get("score"),
                        "response": gen_cache.get(gkey, ""),
                        "judge_reasoning": jdg.get("reasoning", ""),
                    })

    save_json(records, DATA_DIR / "all_results.json")
    valid = sum(1 for r in records if r["score"] is not None)
    print(f"  {len(records)} records total, {valid} valid scores")
    print(f"  Saved to {DATA_DIR / 'all_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
