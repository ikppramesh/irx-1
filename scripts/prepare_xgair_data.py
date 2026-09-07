#!/usr/bin/env python3
"""
Hand-authored training examples teaching IRx-1 the intent-parsing task it
serves at runtime for XGAIR's chat router (see irx1-intent-parser.ts in the
XGAIR repo): free-form natural language -> structured MCP tool call.

Grounded directly in verified xGAIR source (exact tool schemas), not freely
generated, to avoid baking in hallucinated tool names/behavior that could
break the real integration.

v2: dropped the "general knowledge about xGAIR" factual Q&A that was here
originally — tested against the trained model and it reliably hallucinated
(confused "xGAIR" with an unrelated real acronym). ~10 examples isn't enough
signal to override a base model's existing prior on a term; that's a fact-
injection problem fine-tuning is bad at, not something more careful prompts
fix. Kept to the intent-parsing task instead: a narrower, more learnable
structured mapping, with much more coverage this round — especially of
messages where the repo reference is embedded mid-sentence rather than a
clean prefix token, which is where the smaller v1 set was weakest.

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

# (user message, expected JSON response). Deliberately varied repo names
# (not just one repeated string) and, for start_task/add_pattern especially,
# the repo reference is often buried mid-sentence rather than a clean prefix
# token — that pattern is where the earlier, smaller set underperformed.
INTENT_EXAMPLES = [
    # connect_repo
    ("connect https://github.com/acme/widgets",
     '{"tool": "xgair_connect_repo", "args": {"url": "https://github.com/acme/widgets"}}'),
    ("add my repo https://github.com/acme/widgets",
     '{"tool": "xgair_connect_repo", "args": {"url": "https://github.com/acme/widgets", "owner": "acme", "repo": "widgets"}}'),
    ("register acme/widgets with xgair",
     '{"tool": "xgair_connect_repo", "args": {"owner": "acme", "repo": "widgets"}}'),
    ("can you hook up github.com/acme/widgets",
     '{"tool": "xgair_connect_repo", "args": {"url": "https://github.com/acme/widgets", "owner": "acme", "repo": "widgets"}}'),
    ("set up ikppramesh/irx-1 in xgair",
     '{"tool": "xgair_connect_repo", "args": {"owner": "ikppramesh", "repo": "irx-1"}}'),
    ("I want to add torvalds/linux",
     '{"tool": "xgair_connect_repo", "args": {"owner": "torvalds", "repo": "linux"}}'),
    ("please connect https://github.com/facebook/react",
     '{"tool": "xgair_connect_repo", "args": {"url": "https://github.com/facebook/react"}}'),
    ("onboard octocat/hello-world",
     '{"tool": "xgair_connect_repo", "args": {"owner": "octocat", "repo": "hello-world"}}'),

    # discover_repo
    ("discover acme/widgets",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "acme/widgets"}}'),
    ("run discovery on acme/widgets",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "acme/widgets"}}'),
    ("scan acme/widgets for me",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "acme/widgets"}}'),
    ("what stack does acme/widgets use, go find out",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "acme/widgets"}}'),
    ("run the 5-phase scan on ikppramesh/irx-1",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "ikppramesh/irx-1"}}'),
    ("analyze facebook/react for me",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "facebook/react"}}'),
    ("kick off discovery for torvalds/linux",
     '{"tool": "xgair_discover_repo", "args": {"repoId": "torvalds/linux"}}'),

    # get_context
    ("context for acme/widgets",
     '{"tool": "xgair_get_context", "args": {"repoId": "acme/widgets"}}'),
    ("show me the context summary for acme/widgets",
     '{"tool": "xgair_get_context", "args": {"repoId": "acme/widgets"}}'),
    ("what do we already know about this repo",
     '{"tool": "xgair_get_context", "args": {"repoId": ""}}'),
    ("load the context for ikppramesh/irx-1",
     '{"tool": "xgair_get_context", "args": {"repoId": "ikppramesh/irx-1"}}'),
    ("give me the summary for facebook/react",
     '{"tool": "xgair_get_context", "args": {"repoId": "facebook/react"}}'),
    ("pull up what xgair knows about it",
     '{"tool": "xgair_get_context", "args": {"repoId": ""}}'),

    # start_task — heavier coverage, repo often mid-sentence
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
    ("in ikppramesh/irx-1, add rate limiting to the API",
     '{"tool": "xgair_start_task", "args": {"repoId": "ikppramesh/irx-1", "taskDescription": "add rate limiting to the API", "taskType": "feature"}}'),
    ("there's a memory leak in facebook/react, can you investigate and fix it",
     '{"tool": "xgair_start_task", "args": {"repoId": "facebook/react", "taskDescription": "investigate and fix the memory leak", "taskType": "fix"}}'),
    ("torvalds/linux needs a new driver for the xyz chipset, build it",
     '{"tool": "xgair_start_task", "args": {"repoId": "torvalds/linux", "taskDescription": "build a new driver for the xyz chipset", "taskType": "feature"}}'),
    ("please add dark mode support to octocat/hello-world",
     '{"tool": "xgair_start_task", "args": {"repoId": "octocat/hello-world", "taskDescription": "add dark mode support", "taskType": "feature"}}'),
    ("can you look into the crash on startup in acme/widgets",
     '{"tool": "xgair_start_task", "args": {"repoId": "acme/widgets", "taskDescription": "investigate the crash on startup", "taskType": "fix"}}'),
    ("clean up the dead code in ikppramesh/irx-1, just a chore",
     '{"tool": "xgair_start_task", "args": {"repoId": "ikppramesh/irx-1", "taskDescription": "clean up dead code", "taskType": "chore"}}'),
    ("write API docs for the endpoints in facebook/react",
     '{"tool": "xgair_start_task", "args": {"repoId": "facebook/react", "taskDescription": "write API docs for the endpoints", "taskType": "docs"}}'),

    # validate
    ("validate this: const x = eval(userInput)",
     '{"tool": "xgair_validate", "args": {"repoId": "", "codeSnippet": "const x = eval(userInput)"}}'),
    ("check this code against the stack rules: SELECT * FROM users WHERE id = req.query.id",
     '{"tool": "xgair_validate", "args": {"repoId": "", "codeSnippet": "SELECT * FROM users WHERE id = req.query.id"}}'),
    ('does this pass the rules for acme/widgets: password = "hunter2"',
     '{"tool": "xgair_validate", "args": {"repoId": "acme/widgets", "codeSnippet": "password = \\"hunter2\\""}}'),
    ("run a stack check on this snippet from ikppramesh/irx-1: os.system(cmd)",
     '{"tool": "xgair_validate", "args": {"repoId": "ikppramesh/irx-1", "codeSnippet": "os.system(cmd)"}}'),
    ("is this safe: innerHTML = userInput",
     '{"tool": "xgair_validate", "args": {"repoId": "", "codeSnippet": "innerHTML = userInput"}}'),

    # list_repos
    ("list repos",
     '{"tool": "xgair_list_repos", "args": {}}'),
    ("what repos do I have connected",
     '{"tool": "xgair_list_repos", "args": {}}'),
    ("show status",
     '{"tool": "xgair_list_repos", "args": {}}'),
    ("which of my repos are on the acme org",
     '{"tool": "xgair_list_repos", "args": {"filterOwner": "acme"}}'),
    ("give me my repo list",
     '{"tool": "xgair_list_repos", "args": {}}'),
    ("what's connected right now",
     '{"tool": "xgair_list_repos", "args": {}}'),

    # get_patterns
    ("patterns for acme/widgets",
     '{"tool": "xgair_get_patterns", "args": {"repoId": "acme/widgets"}}'),
    ("what gotchas have we learned about acme/widgets",
     '{"tool": "xgair_get_patterns", "args": {"repoId": "acme/widgets"}}'),
    ("any known issues with auth in acme/widgets",
     '{"tool": "xgair_get_patterns", "args": {"repoId": "acme/widgets", "searchQuery": "auth"}}'),
    ("show me the learned patterns for ikppramesh/irx-1",
     '{"tool": "xgair_get_patterns", "args": {"repoId": "ikppramesh/irx-1"}}'),
    ("search facebook/react's gotchas for anything about caching",
     '{"tool": "xgair_get_patterns", "args": {"repoId": "facebook/react", "searchQuery": "caching"}}'),
    ("what mistakes have we made before on torvalds/linux",
     '{"tool": "xgair_get_patterns", "args": {"repoId": "torvalds/linux"}}'),

    # add_pattern — heavier coverage, repo often buried inside the gotcha text
    ("add pattern: the auth middleware silently swallows errors on token refresh",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "", "gotcha": "the auth middleware silently swallows errors on token refresh"}}'),
    ("log a gotcha, the build fails silently if the .env file is missing on CI",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "", "gotcha": "the build fails silently if the .env file is missing on CI"}}'),
    ("I just found out that the auth middleware silently swallows errors on token refresh in acme/widgets, add that as a gotcha",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "acme/widgets", "gotcha": "the auth middleware silently swallows errors on token refresh"}}'),
    ("note for ikppramesh/irx-1: the migration script deletes rows before checking foreign keys, log that",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "ikppramesh/irx-1", "gotcha": "the migration script deletes rows before checking foreign keys"}}'),
    ("add a pattern for facebook/react — useEffect cleanup functions were being skipped on fast refresh",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "facebook/react", "gotcha": "useEffect cleanup functions were being skipped on fast refresh"}}'),
    ("here's a gotcha for torvalds/linux: the driver panics if initialized before the bus is ready, please save it",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "torvalds/linux", "gotcha": "the driver panics if initialized before the bus is ready"}}'),
    ("log this issue against acme/widgets: pagination silently drops the last page when the count is exact",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "acme/widgets", "gotcha": "pagination silently drops the last page when the count is exact"}}'),
    ("remember this for next time on octocat/hello-world: the CI cache key doesn't include the lockfile hash",
     '{"tool": "xgair_add_pattern", "args": {"repoId": "octocat/hello-world", "gotcha": "the CI cache key does not include the lockfile hash"}}'),

    # get_constraints
    ("constraints for acme/widgets",
     '{"tool": "xgair_get_constraints", "args": {"repoId": "acme/widgets"}}'),
    ("what's the scope and ceiling for acme/widgets",
     '{"tool": "xgair_get_constraints", "args": {"repoId": "acme/widgets"}}'),
    ("show me the constraint block for ikppramesh/irx-1",
     '{"tool": "xgair_get_constraints", "args": {"repoId": "ikppramesh/irx-1"}}'),
    ("what are the forbidden patterns for facebook/react",
     '{"tool": "xgair_get_constraints", "args": {"repoId": "facebook/react"}}'),
    ("give me the rules before I start on torvalds/linux",
     '{"tool": "xgair_get_constraints", "args": {"repoId": "torvalds/linux"}}'),

    # unknown — off-topic negatives
    ("what's the weather like today",
     '{"tool": "unknown"}'),
    ("tell me a joke",
     '{"tool": "unknown"}'),
    ("what's your favorite color",
     '{"tool": "unknown"}'),
    ("how do I cook pasta",
     '{"tool": "unknown"}'),
    ("who won the game last night",
     '{"tool": "unknown"}'),
    ("sing me a song",
     '{"tool": "unknown"}'),
]


def build_examples():
    examples = []
    for user_msg, assistant_json in INTENT_EXAMPLES:
        examples.append({"messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_json},
        ]})
    return examples


def main():
    examples = build_examples()
    out_path = Path("data/processed/distilled_xgair.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} intent-parsing examples -> {out_path}")


if __name__ == "__main__":
    main()
