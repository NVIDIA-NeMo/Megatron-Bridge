# Issue Triage Workflow

Four-phase workflow with two maintainer review gates:

**Phase 1:** Fetch & Summarize → **Review #1** → **Phase 2:** Execute PRs +
Draft Responses → **Review #2** → **Phase 3:** Post Responses + Relabel

---

## Phase 1: Fetch & Summarize

### 1a: Fetch `needs-triage` issues

Collect all open issues labeled `needs-triage`.

### 1b: Read each issue

Fetch the full body and comments for each issue.

### 1c: Classify each issue

1. **Type** — `bug`, `feature`, `support`, `docs`, `ci`
2. **Area** — `area:model`, `area:recipe`, `area:training`, `area:data`,
   `area:ckpt`, `area:peft`, `area:perf`, `area:build`, `area:misc`
3. **Author affiliation** — if external contributor, note `community-request`
4. **Existing PR check** — always check if the author (or anyone) already
   opened a PR. Prefer reviewing author PRs over creating duplicates.
5. **Actionability** — categorize into one of:
   - **Known fix (existing PR)** — author already opened a PR; review and merge
   - **Known fix (new PR needed)** — root cause is clear, no existing PR
   - **Needs more info** — need repro script, env details, or clarification
   - **Other response** — support question, feature request, or redirect

### 1d: Generate triage report

The report includes:

1. **Issues with Known Fixes** — metadata, fix description, existing or planned
   PR, proposed GitHub response
2. **Issues Needing More Info** — metadata, assessment, proposed question
3. **Issues with Other Responses** — support questions, feature requests
4. **Overview table** — all issues with columns: #, Title, Type, Area, New
   State, Action
5. **PR Plan** — table of new and existing PRs with status

### 1e: Maintainer review gate

Present the report to the maintainer. **Stop and wait for approval** before
executing any PRs or posting responses.

---

## Phase 2: Execute PRs + Draft Responses

After the maintainer approves the triage plan:

### 2a: Create PRs for known fixes

For each issue needing a new PR (not covered by an existing author PR):

1. Create a branch from `main`
2. Apply the fix
3. Commit with signed-off conventional commit message
4. Push and create the PR

### 2b: Finalize responses

Update the triage report with actual PR links (replacing placeholders).

### 2c: Second maintainer review gate

Present the finalized draft responses. **Stop and wait for approval** before
posting any comments.

---

## Phase 3: Post Responses + Relabel

After the maintainer approves the responses:

### 3a: Post responses on issues

Comment on each issue with the approved response.

### 3b: Relabel issues

Common state transitions:

- **Known fix with PR** → remove `needs-triage`, add `needs-follow-up`
- **Needs more info** → add `needs-author`
- **Support answered** → remove `needs-triage`, add `needs-follow-up`

---

## Decision Tree

```
NEEDS-TRIAGE ISSUE
├── Read issue body + comments
├── Classify: type + area
├── Check for existing author PRs (ALWAYS do this)
├── Determine actionability:
│   ├── Known fix (author PR exists) → link existing PR, plan review
│   ├── Known fix (no PR) → plan new PR
│   ├── Needs info → draft question for author
│   └── Other → draft response (support, feature, redirect)
├── Write to triage report
└── REVIEW GATE #1: maintainer approves plan
    └── On "go":
        ├── Create new PRs (known fixes without author PR)
        ├── Finalize draft responses with PR links
        └── REVIEW GATE #2: maintainer approves responses
            └── On "reply":
                ├── Post responses (all non-skipped)
                └── Relabel issues
```
