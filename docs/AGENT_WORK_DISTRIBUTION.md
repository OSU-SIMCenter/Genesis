# Agent Work Distribution — routing, protocol, and the live board

**Status:** living document. Written 2026-08-14 from session `bcbbd0dc` (workstream B-316L-7).
**This file is the coordination channel.** Sibling sessions on other branches read it without
checking anything out:

```bash
git show origin/agforge/v2/thermal-st-invariance:docs/AGENT_WORK_DISTRIBUTION.md
```

> It is a file in the repo and **not** an artifact, deliberately. Artifacts are account-scoped —
> from the other Claude account they are invisible and the failure looks like deletion. The repo
> is the only channel between these sessions that has ever actually worked.

---

## 1. Why this exists — the constraint is verification, not throughput

The bottleneck in this project has never been how much work gets produced. It is how much of what
gets produced turns out to be **true**. A single day (2026-08-13/14) produced:

- two of my own headline claims overturned by my own later measurement,
- a tool the docs pointed at as "already exists for this" that had **never worked**
  (`benchmark_force_balance.py` calls `args.visualize`, never defined in its parser),
- a doc item stale for weeks that said a measurement was impossible using data already on disk,
- a sibling session independently working the item this session had labelled "the single most
  important open item", discovered only by reading their transcript.

Adding agents multiplies claim *generation*. It does not multiply claim *verification*. So the
routing question is not "what can run in parallel" but **"what can be parallelised without
multiplying the failure mode."**

### 🚨 The one rule that matters most

**Never let the same tier write both an implementation and its test.**

This project has already paid for this once: **sixteen mirror tests all passed** while
re-implementing the same 1000× units error (see coupling doc §5 and
[[thermal-solver-explicit-instability]]). A model asked to write code *and* verify it produces
**consistency, not correctness**.

⇒ **The Claude tier writes the acceptance criterion, before the work starts and before seeing any
code. The Cursor tier implements. The Claude tier checks the result against the criterion it
wrote.** If that ordering is broken, the delegation is worthless regardless of model quality.

---

## 2. Routing — what goes where

Calibration is **task-dependent**, and measured rather than assumed (see
[[cursor-cli-capabilities]]): Cursor models perform at roughly Sonnet-high→Opus tier **on code**
(review, generation, refactor) and at mid-tier, shallower, **on general reasoning and research**.
`cursor-grok-4.5-high` scored **5/5 on three independent calibration gauges** on this very
project — verifying arithmetic it was handed rather than accepting it, confirming GPU-vs-CPU
backend before reporting numbers, and giving honest "this weakens but does not settle it" reads.

| ✅ Delegate to Cursor / Grok | ❌ Keep on Claude (Opus 5) |
|---|---|
| Implementing to a **written spec with a pre-stated test** | Deciding whether a finding is real |
| Mechanical refactors, plumbing, porting | Anything touching `base_mpm_solver.py` / `strike_controller.py` |
| Scripts whose output is a **table or pass/fail** | Physics judgment; model selection |
| Repo-wide staleness sweeps — **report, never decide** | Cross-workstream reconciliation |
| Code review / bug hunt (their strongest suit) | Interpreting a disagreement between two measurements |
| Literature retrieval returning **URLs + verbatim quotes** | Turning literature into a recommendation |
| Inventories, git archaeology, measure-and-report | Anything whose deliverable is a *claim*, not an *artifact* |

⚠️ **On literature specifically:** running N engines validates **structure, never specifics**, and
confidence has run **anti-correlated** with accuracy ([[ai-research-engine-fabrication]]). Cursor
agents may *fetch* sources. They may not *summarise into claims we then act on*.

⚠️ **On the shared solver:** this worktree is shared with sibling sessions that commit to it. A
Cursor agent editing `agforge/` can silently break another session's work — including, as
happened on 08-13, invalidating line-number citations in a pushed document. Solver edits stay on
Claude and stay coordinated.

---

## 3. Protocol — four rules, each earned from a specific failure

1. **One writer per file, assigned not assumed.** *Earned: B-3's two comment-only commits shifted
   `options.py` by +12 lines and `strike_controller.py` by +10, silently invalidating ten
   `file.py:NNN` citations in a pushed document — including the one the headline force-limit
   finding rested on.* Corollary: **cite the symbol, not the line** (`grep -n "cond_force = "`),
   and if you must give a line, stamp the commit it was valid at.

2. **Push or it didn't happen.** *Earned in both directions: B-3 read a pushed finding and fixed
   two diagnostics within hours — the first time a sibling acted on our work. Meanwhile A-7/A-8's
   toolchain sat `[ahead 3]` and unpushed on one machine, exactly the single-point-of-failure
   shape that made the 06-15 mcap look "blocked on the user" for two weeks while it sat on disk.*

3. **The board lives in the repo.** See the header. Artifacts are account-scoped.

4. **Every delegated task returns an artifact — a commit, a script, a table — never a summary.**
   A summary cannot be checked. A table can. This is also what makes rule 1 of §1 enforceable.

### The calibration gauge — cheap, and it converts trust into evidence

Attach to every dispatch **5 questions whose answers you have already verified independently**,
mixed in with the real work. Score them. A run that misses gauge questions has its real output
treated as unverified regardless of how confident it reads. Three grok dispatches on this project
scored 5/5, and one **corrected the orchestrator's line number** — which is the outcome that
makes the gauge worth its cost.

### 🚨 Contamination: give every dispatch a unique workspace

Cursor's own system prompt tells each agent where past chat transcripts for a workspace live, and
the slug is derived from the workspace **path**. A second run on the same `--workspace` can read
the first run's full transcript. **For any A/B, replication, or independent-corroboration test,
every arm needs a unique workspace path** — and note this also means ordinary reruns are not
truly independent.

---

## 4. Verified dispatch recipe

Checked live 2026-08-14.

```bash
# WSL binary — the repo lives in WSL. Never dispatch the Windows binary at a WSL path:
# writes through \\wsl.localhost zero-pad to 4096-byte boundaries (silent corruption).
~/.local/bin/cursor-agent -p \
  --trust \
  --model 'cursor-grok-4.6-xhigh' \
  --workspace /home/timothy/GitHub/Genesis/aims-genesis/thermal-st-invariance \
  --output-format json \
  "Read /home/timothy/briefs/<brief>.md and follow it."
```

- **`--trust` is REQUIRED** for headless `-p`. Without it the run fast-fails in ~1 s with exit 1
  and no stdout — it looks like a model error and is not.
- **Fast mode is a distinct model id, not a flag.** `cursor-grok-4.6-xhigh` is the non-fast
  extra-high variant; `cursor-grok-4.6-xhigh-fast` is the fast one. **Selecting the id without
  the suffix is fast-off by construction.** Available 4.6 ladder: `-low`, `-medium`, `-high`,
  `-xhigh` (each with a `-fast` twin).
- **`-p` has full write + shell access by default.** Read-only is opt-in via `--mode ask` or
  `--mode plan`. Use `--mode ask` for anything that should not touch the tree.
- 🚨 **The whole dispatch must live in a shell script file. Never inline the prompt.**
  Passing it through PowerShell → `wsl.exe` → `bash -lc` **strips the double quotes**, so
  `cursor-agent` receives only the **first token** of the prompt and the remaining words become
  stray argv entries. **Verified 2026-08-14:** `"Reply with exactly: PILOT_OK"` arrived as
  `Reply`. ⚠️ **The run still exits 0 and returns a well-formed success JSON** — the agent
  cheerfully answers the truncated prompt — so this fails *silently* and looks like a bad model
  response rather than a broken dispatch. The first pilot dispatch was lost to exactly this.
  Stage the script on Windows, `cp` it in from `/mnt/c`, then `bash -lc /path/to/script.sh`.
- **Long briefs go in a file**; the dispatch prompt stays one line pointing at it.
- Auth is per-account and has changed before — run `cursor-agent status` rather than assuming.
- Every run is auditable afterwards at `~/.cursor/chats/<hash>/<uuid>/store.db` (SQLite, plaintext
  JSON blobs). Open it **read-only** (`file:...?immutable=1`); never copy a live DB with its
  `-wal`/`-shm`.

---

## 5. The live board

Sessions go stale fast — **A-7 → A-8, B-3 → B-4 inside a week**, while handoffs kept naming the
dead ids. **Re-derive rather than trust this table**: transcript `customTitle` + mtime under
`~/.claude/projects/<slug>/`.

| workstream | owner | scope | state |
|---|---|---|---|
| **W-A · Instrumentation & coupling** | Claude, unassigned | clamp telemetry, A1 KE/IE monitor, A2 energy audit, A3 temperature telemetry, A6 coupled path | open |
| **W-B · Scaling validation** | Claude, unassigned | **D1a N-sweep at gain 5e-5**, physical-rate default, `rate_max` guard | **briefed — best Tier-2 pilot** |
| **W-C · Materials & extrapolation** | Claude, unassigned | 4340 restoration + consistency test, Q/n creep check, out-of-band rate corroboration | open |
| Contact / toolchain | A-8 `81414d01` | contact engine, force-stop stability | active, `[ahead 3]` unpushed |
| Billet geometry / IC | A-7-billet `40379be1` | IC comparison, cloud-derived closure | active |
| Thermal gather / FLIP | B-4 `4a4afd66` | Phase C, gather replacement | active |
| Coupling / scaling / metrics / mcap | this line | the canonical doc | `aceb404a` |

**Why three new workstreams and not eight.** Six sessions already collided three times this week
and there is still no channel between them beyond this file. Coordination cost is superlinear;
the marginal agent is only worth it if its output can be verified without a Claude session
reading everything it did.

### Not delegable — deliberately

- **§4.7.1's terminal force spike.** Survives a 30× gain reduction with the dies tracking to
  0.04%, so it is not the balance controller. Least tractable item in the backlog and the worst
  delegation candidate; it needs whoever holds the most context.
- **The A-8 closure reconciliation.** Two workstreams measuring the same quantity 2 mm apart is
  exactly the "interpreting a disagreement" row of §2.

---

## 6. Pilot, and how it gets judged

**W-B runs first, alone**, because it is the best-specified work we have: its acceptance criteria
were written into the coupling doc's §9.4 *before* any of this and are numeric.

Judge the pilot on one question: **did the Tier-2 output survive Claude-tier verification without
a Claude session having to redo the work?** If yes, scale to W-A and W-C. If no, we have learned
that for the price of one workstream instead of three.

⚠️ **Record the answer here either way.** An unrecorded negative result is how
`benchmark_force_balance.py` stayed broken and cited for weeks.

### ✅ Pilot 1 result, 2026-08-14 — `clamp_probe.py`, `cursor-grok-4.6-xhigh`

**Verdict: pass, and the output improved the Claude-tier work rather than merely matching it.**
579 s, 157k in / 37k out.

- **Calibration gauge 5/5**, with three genuine additions the orchestrator did not have: that the
  `JohnsonCookPlasticity` class docstring still advertises the `(1 + C ln ε̇*)` term the kernel
  never evaluates; that Genesis's own `SimOptions` default is 1 substep, not this worktree's 8;
  and the mirroring comment in `base_mpm_solver.py` that names the consistency test.
- **Acceptance criteria 2–5 met**, including reproducing the known 2.33× clamp error exactly
  (181.1 vs 77.8 MPa) — the criterion it was invited to fail honestly, and did not need to.
- **Criterion 1 was mis-specified by the orchestrator** ("without pixi") — numpy is not in the
  system interpreter, so no script in this repo can meet it. The intent (no GPU, seconds) is met:
  ~1 s under pixi. **The criterion was wrong, not the work.**
- **It respected the shared-source constraint**: `git status` showed exactly one untracked file.
- 🎯 **It pushed back on the brief, correctly.** It found that the brief's criterion 4 targeted
  the *former* temperature wall while part (3) specified measured medians — inconsistent as
  written — and printed both. And it found the result recorded in §8.3.1 of the material doc:
  the clamp that actually binds is the temperature **floor**, not the rate ceiling. That
  reversed the priority ordering in a document the orchestrator had written the day before.

**What this says about routing.** The task was well-suited by construction — mechanically
checkable output, no shared-source edits, no physics judgment required, and acceptance criteria
fixed before dispatch. The pushback in §4 of its report is the part that earned trust, and it
only exists because the brief explicitly invited it. **Keep that invitation in every brief.**

⚠️ **The failure mode this pilot did *not* test:** every number it produced was independently
re-derivable, and the orchestrator re-ran the script and checked the envelope figures against a
separate prior derivation. A task whose output *cannot* be checked that cheaply has not been
shown to be safe to delegate.
