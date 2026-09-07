#!/usr/bin/env python3
"""
Hand-authored training examples teaching IRx-1 about xGAIR (both general
knowledge and the exact intent-parsing task IRx-1 already serves at runtime
for XGAIR's chat router — see irx1-intent-parser.ts in the XGAIR repo).

Grounded directly in verified xGAIR source/docs, not freely generated, to
avoid baking in hallucinated tool names/behavior that could break the real
integration.

Usage:
    python scripts/prepare_xgair_data.py
"""

import json
from pathlib import Path

# Must stay byte-for-byte identical to XGAIR's irx1-intent-parser.ts SYSTEM_PROMPT
# so fine-tuning reinforces exactly the task IRx-1 is invoked for at runtime.
INTENT_SYSTEM_PROMPT = """You are an intent parser for the xGAIR developer tool chat interface. Given a user message, output ONLY a single JSON object (no prose, no markdown fences) describing which tool to call.

Available tools:
- xgair_connect_repo: {url?: string, owner?: string, repo?: string} - register/add a NEW GitHub repo that hasn't been set up yet.
- xgair_discover_repo: {repoId: string} - run the 5-phase discovery/analysis scan on an already-connected repo.
- xgair_get_context: {repoId: string} - load the existing context summary for a repo (discovery already ran).
- xgair_start_task: {repoId: string, taskDescription: string, taskType: feature|fix|refactor|chore|docs} - begin a coding task on a repo.
- xgair_validate: {repoId: string, codeSnippet: string} - check a code snippet against stack rules.
- xgair_list_repos: {} - list all connected repos / show status.
- xgair_get_patterns: {repoId: string, searchQuery?: string} - show known gotchas/patterns for a repo.
- xgair_add_pattern: {repoId: string, gotcha: string} - log a new gotcha (min 20 characters).
- xgair_get_constraints: {repoId: string} - get the constraint block for a repo.

If the user refers to it/this repo/the current one without naming a repo, set repoId to empty string.
If the message does not clearly map to one of these tools, output {"tool": "unknown"}.

Output strict JSON only: {"tool": "<name>", "args": {...}}"""

# (user message, expected JSON response) — varied phrasings per tool.
INTENT_EXAMPLES = [
    ("connect https://github.com/acme/widgets",
     '{"tool": "xgair_connect_repo", "args": {"url": "https://github.com/acme/widgets"}}'),
    ("add my repo https://github.com/acme/widgets",
     '{"tool": "xgair_connect_repo", "args": {"url": "https://github.com/acme/widgets", "owner": "acme", "repo": "widgets"}}'),
    ("register acme/widgets with xgair",
     '{"tool": "xgair_connect_repo", "args": {"owner": "acme", "repo": "widgets"}}'),
    ("can you hook up github.com/acme/widgets",
     '{"tool": "xgair_connect_repo", "args": {"url": "https://github.com/acme/widgets", "owner": "acme", "repo": "widgets"}}'),
    ("discover acme/widgets",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "acme/widgets"}}'),
    ("run discovery on acme/widgets",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "acme/widgets"}}'),
    ("scan acme/widgets for me",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "acme/widgets"}}'),
    ("what stack does acme/widgets use, go find out",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "acme/widgets"}}'),
    ("context for acme/widgets",
     '{"tool": "xgair_get_context", "args": {"repoId": "acme/widgets"}}'),
    ("show me the context summary for acme/widgets",
     '{"tool": "xgair_get_context", "args": {"repoId": "acme/widgets"}}'),
    ("what do we already know about this repo",
     '{"tool": "xgair_get_context", "args": {"repoId": ""}}'),
    ("start task: add a favicon in acme/widgets",
     '{"tool": "xgair_start_task", "args": {"repoId": "acme/widgets", "taskDescription": "add a favicon", "taskType": "feature"}}'),
    ("can you fix the login bug in acme/widgets",
     '{"tool": "xgair_start_task", "args": {"repoId": "acme/widgets", "taskDescription": "fix the login bug", "taskType": "fix"}}'),
    ("refactor the auth module in acme/widgets to use hooks",
     '{"tool": "xgair_start_task", "args": {"repoId": "acme/widgets", "taskDescription": "refactor the auth module to use hooks", "taskType": "refactor"}}'),
    ("update the docs for acme/widgets",
     '{"tool": "xgair_start_task", "args": {"repoId": "acme/widgets", "taskDescription": "update the docs", "taskType": "docs"}}'),
    ("bump the eslint config in acme/widgets, low priority chore",
     '{"tool": "xgair_start_task", "args": {"repoId": "acme/widgets", "taskDescription": "bump the eslint config", "taskType": "chore"}}'),
    ("validate this: const x = eval(userInput)",
     '{"tool": "xgair_validate", "args": {"repoId": "", "codeSnippet": "const x = eval(userInput)"}}'),
    ("check this code against the stack rules: SELECT * FROM users WHERE id = req.query.id",
     '{"tool": "xgair_validate", "args": {"repoId": "", "codeSnippet": "SELECT * FROM users WHERE id = req.query.id"}}'),
    ("list repos",
     '{"tool": "xgair_list_repos", "args": {}}'),
    ("what repos do I have connected",
     '{"tool": "xgair_list_repos", "args": {}}'),
    ("show status",
     '{"tool": "xgair_list_repos", "args": {}}'),
    ("which of my repos are on the acme org",
     '{"tool": "xgair_list_repos", "args": {"filterOwner": "acme"}}'),
    ("patterns for acme/widgets",
     '{"tool": "xgair_get_patterns", "args": {"repoId": "acme/widgets"}}'),
    ("what gotchas have we learned about acme/widgets",
     '{"tool": "xgair_get_patterns", "args": {"repoId": "acme/widgets"}}'),
    ("any known issues with auth in acme/widgets",
     '{"tool": "xgair_get_patterns", "args": {"repoId": "acme/widgets", "searchQuery": "auth"}}'),
    ("add pattern: the auth middleware silently swallows errors on token refresh",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "", "gotcha": "the auth middleware silently swallows errors on token refresh"}}'),
    ("log a gotcha, the build fails silently if the .env file is missing on CI",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "", "gotcha": "the build fails silently if the .env file is missing on CI"}}'),
    ("constraints for acme/widgets",
     '{"tool": "xgair_get_constraints", "args": {"repoId": "acme/widgets"}}'),
    ("what's the scope and ceiling for acme/widgets",
     '{"tool": "xgair_get_constraints", "args": {"repoId": "acme/widgets"}}'),
    ("what's the weather like today",
     '{"tool": "unknown"}'),
    ("tell me a joke",
     '{"tool": "unknown"}'),
    ("what's your favorite color",
     '{"tool": "unknown"}'),
]

# General knowledge about xGAIR, grounded in the actual README/source.
FACTUAL_EXAMPLES = [
    ("What is xGAIR?",
     "xGAIR (AI-Augmented GitHub Intelligence) is a real MCP (Model Context "
     "Protocol) server that plugs into AI coding assistants — Claude Code, VS "
     "Code Copilot, Cursor, Windsurf, ChatGPT, Zed, and others — and turns them "
     "into a disciplined, context-aware development partner for any GitHub "
     "repository. It's not a chatbot wrapper; it's a structured intelligence "
     "layer that discovers a repo via the GitHub API, understands its stack and "
     "conventions, enforces a strict development workflow, remembers gotchas "
     "from past tasks, and validates code against stack-specific anti-pattern "
     "rules before it ships."),
    ("How do I get started with xGAIR?",
     "Run `npx xgair init --github <your-username>`. The interactive wizard "
     "verifies your GitHub token, sets up the data workspace, connects your "
     "first repo, and runs 5-phase AI discovery on it."),
    ("What does xGAIR's discovery process do?",
     "Discovery runs a multi-phase scan of a connected repo: it scans the "
     "codebase, synthesizes findings, extracts conventions and patterns, "
     "writes a structured context summary, and produces a development guide. "
     "An optional 6th phase runs a security scan. The result is used to give "
     "any AI assistant grounded, accurate context about that specific repo."),
    ("What is xGAIR's 4-phase development workflow?",
     "Every task follows: PHASE 1 — INVESTIGATE (load context, patterns, and "
     "constraints; identify at least two implementation approaches; no code "
     "yet), PHASE 2 — PLAN (a mandatory stop — present the exact files to "
     "change, a test strategy, and a branch/PR name, then wait for explicit "
     "approval before continuing), PHASE 3 — IMPLEMENT (only after approval, "
     "following the constraint block exactly), and PHASE 4 — VALIDATE "
     "(summarize what was done and log any new gotchas discovered)."),
    ("What MCP tools does the xGAIR server expose?",
     "Nine tools: xgair_connect_repo (register a repo), xgair_discover_repo "
     "(run the discovery scan), xgair_get_context (load a repo's context "
     "summary), xgair_start_task (begin a coding task), xgair_validate (check "
     "code against stack anti-patterns), xgair_list_repos (list connected "
     "repos), xgair_get_patterns (query known gotchas), xgair_add_pattern (log "
     "a new gotcha), and xgair_get_constraints (get the scope/ceiling "
     "constraint block for a repo)."),
    ("How does xGAIR's chat CLI work?",
     "`npx xgair chat` starts an interactive REPL that talks to the running "
     "xGAIR MCP server over HTTP JSON-RPC. It matches input against known "
     "command patterns like `connect <url>`, `discover <owner/repo>`, "
     "`context for <repo>`, and `start task: <description> in <repo>`. Free-form "
     "phrasing that doesn't match an exact pattern can optionally be "
     "interpreted by a local natural-language model instead of requiring exact "
     "syntax."),
    ("What does xGAIR remember between tasks?",
     "It keeps a per-repo store of learned patterns — gotchas discovered "
     "during past work, along with any constraint that was added to prevent "
     "recurrence and any reusable solution pattern. These are surfaced "
     "automatically at the start of a new task (PHASE 1 — INVESTIGATE) so the "
     "same mistake doesn't get made twice."),
    ("What happens if xGAIR can't validate code cleanly?",
     "xgair_validate checks a code snippet against the anti-pattern rules for "
     "that repo's detected stack and reports violations before the code ships, "
     "as part of PHASE 4 — VALIDATE in the task workflow."),
    ("Does xGAIR require cloning the repository locally?",
     "No — it discovers and analyzes repositories via the GitHub API "
     "directly, without requiring a local clone."),
    ("What is the mandatory PLAN stop in xGAIR's workflow?",
     "After PHASE 1 — INVESTIGATE, xGAIR must present its implementation plan "
     "(files to change, files to create and why, the constraint block, test "
     "strategy, and branch/PR name) and then stop completely, waiting for "
     "explicit developer approval before writing any code in PHASE 3. This "
     "prevents an AI assistant from making unreviewed changes."),
]


def build_examples():
    examples = []
    for user_msg, assistant_json in INTENT_EXAMPLES:
        examples.append({"messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_json},
        ]})
    for question, answer in FACTUAL_EXAMPLES:
        examples.append({"messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]})
    return examples


def main():
    examples = build_examples()
    out_path = Path("data/processed/distilled_xgair.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples "
          f"({len(INTENT_EXAMPLES)} intent-parsing + {len(FACTUAL_EXAMPLES)} factual) "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
