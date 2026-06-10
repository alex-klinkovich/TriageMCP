# TriageMCP

[![CI](https://github.com/alex-klinkovich/TriageMCP/actions/workflows/ci.yml/badge.svg)](https://github.com/alex-klinkovich/TriageMCP/actions/workflows/ci.yml) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Typing: mypy strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

An agentic security-alert triage engine, exposed over the **Model Context Protocol (MCP)**.

TriageMCP ingests security alerts, runs an LLM agent that investigates each one with
deterministic, fully-tested tools, and returns a **schema-validated verdict**: severity,
confidence, MITRE ATT&CK technique, recommended action, and an evidence-based rationale.
It triages batches concurrently with per-alert error isolation, ships a self-scoring eval
harness, and serves the `triage_alert` capability to any MCP client.

## Eval headline

<!-- EVAL:START -->
**Overall triage accuracy: 54.4%** (severity 52.6% exact / 100.0% within one level, MITRE technique 65.8%, action 44.7%) over 38 scored alerts with 0 errors.
<!-- EVAL:END -->

The number above is written **by the eval harness itself** (`triagemcp eval --update-readme`),
so it is always reproducible from the code, not hand-edited.

**Eval-driven iteration:** a controlled prompt experiment lifted MITRE technique accuracy on
Sonnet from **73.7% to 91.4%** (temperature 0, over the 35 scored alerts — or **84.2%** if the 3
large-prompt timeouts are counted as misses rather than excluded) by adding an ATT&CK technique
catalog to the system prompt — with an honestly-documented latency cost (those timeouts) and a
negative result (few-shot examples hurt the target metric). Full write-up, including the Haiku
variant ladder and Wilson confidence intervals, in
[`docs/experiments/mitre-accuracy.md`](docs/experiments/mitre-accuracy.md).

## Quickstart (< 2 minutes)

Requires **Python 3.12+** (this project targets 3.12 specifically) and `git`.

```bash
# 1. Create a 3.12 virtual environment and install the project + dev tools.
py -3.12 -m venv .venv            # Windows;  on macOS/Linux: python3.12 -m venv .venv
.venv\Scripts\activate            # Windows;  on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# 2. Everything below works WITHOUT a key:
triagemcp sample                  # print the 38 bundled labeled alerts as JSON
pytest                            # 128 tests, fully offline, ~8s
ruff check . && mypy              # zero lint / type errors

# 3. The agent itself needs a key (this is your Anthropic spend):
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # Windows PowerShell
# export ANTHROPIC_API_KEY="sk-ant-..."   # macOS/Linux

triagemcp triage                  # triage the bundled sample alerts -> rich table
triagemcp triage --format json --out report.json
triagemcp eval --update-readme    # score the agent on the labeled set, record the headline
triagemcp serve                   # start the MCP server over stdio
```

`triagemcp triage <file.json>` accepts either a JSON array of alerts or of labeled alerts
(the `sample` output works directly).

## Architecture

The design isolates every source of non-determinism behind an injected interface, so the
entire test suite is offline, deterministic, and free, and `mypy --strict` stays clean.

```
ingest ─► TriageAgent (tool-use loop) ─► TriageResult (schema-validated)
              │  every model call behind the LLMClient seam
              │  investigates via the ToolRegistry, finishes via submit_triage
              ▼
        ┌── deterministic tools ──────────────────────────────┐
        │ map_to_mitre · lookup_ip_reputation                  │
        │ query_recent_alerts (aiosqlite) · enrich_hash        │
        └──────────────────────────────────────────────────────┘

batch:   triage_batch  = TaskGroup + Semaphore, per-alert error isolation
serve:   FastMCP        = exposes triage_alert(alert) -> TriageResult
eval:    run_eval       = triage the labeled set, score severity + MITRE + action
```

### Key seams

- **`LLMClient` protocol** (`agent/llm.py`) is the only thing that talks to Anthropic.
  `AnthropicLLMClient` adapts the SDK and maps transient SDK errors onto a retryable
  `TransientLLMError`; `FakeLLMClient` (`testing.py`) replays scripted turns in tests. No SDK
  types leak into the agent loop.
- **Structured output via a terminal tool.** The agent doesn't parse JSON from prose — it
  finishes by calling a `submit_triage` tool whose input schema *is* `TriageResult`. Invalid
  submissions are rejected with a corrective tool result and retried in-loop. This is the
  "output MUST validate or be rejected and retried" requirement, enforced at the boundary.
- **Tools as injected dependencies** (`tools/`). Each tool owns a Pydantic input model
  (its JSON schema is what the model sees) and a backing store passed in via the constructor
  (aiosqlite connection, local JSON store, or an httpx client). No global mutable state.
- **Config from the environment only** (`config.py`, `pydantic-settings`). The key is read
  from `ANTHROPIC_API_KEY` and held as a `SecretStr`; everything else uses the `TRIAGEMCP_`
  prefix.

### Module layout

```
src/triagemcp/
  models.py         Alert, TriageResult, LabeledAlert, TriageOutcome, enums
  config.py         Settings (pydantic-settings); secret as SecretStr
  errors.py         exception hierarchy (TransientLLMError, AgentError, ...)
  datasets.py       load the bundled sample alerts
  tools/            base (BaseTool/ToolResult), registry, and the 4 tools
  agent/            llm (seam + adapter), prompts, retry, loop (TriageAgent)
  pipeline.py       triage_batch: TaskGroup + Semaphore + isolation
  server.py         FastMCP server + production runtime builder
  eval/             metrics (scoring) + harness (run_eval, README writer)
  report.py         render outcomes/eval as JSON / Markdown / rich table
  cli.py            Typer CLI: triage | eval | serve | sample
  testing.py        FakeLLMClient + turn builders (offline test doubles)
  data/             sample_alerts.json, mitre_techniques.json, ip/hash stores
```

## Connecting the MCP server

`triagemcp serve` speaks MCP over stdio. To use it from **Claude Desktop**, add to its
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "triagemcp": {
      "command": "triagemcp",
      "args": ["serve"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

To inspect it interactively with the **MCP Inspector**:

```bash
npx @modelcontextprotocol/inspector triagemcp serve
```

It exposes one tool, `triage_alert(alert) -> TriageResult`.

### Docker

```bash
docker build -t triagemcp .
docker run --rm -i -e ANTHROPIC_API_KEY=sk-ant-... triagemcp   # serves over stdio
```

## Design decisions

- **MCP via the official SDK's `FastMCP`** (`mcp.server.fastmcp`), not a third-party layer —
  one first-party dependency, the most credible choice for a security audience.
- **Default model `claude-sonnet-4-6`**, configurable via `TRIAGEMCP_MODEL` or `--model`:
  the best accuracy/cost balance for high-volume triage.
- **A terminal `submit_triage` tool** rather than JSON-in-prose parsing — the model literally
  cannot return a structurally-invalid verdict and have it accepted.
- **`asyncio.TaskGroup` workers swallow their own exceptions.** TaskGroup is fail-fast (one
  raising child cancels its siblings), so the per-alert `try/except` inside each worker *is*
  the isolation mechanism. Outcomes are returned in input order regardless of completion order.
- **Deterministic exponential backoff (no jitter)** in the retry helper, with an injectable
  `sleep`, so retry behaviour is reproducible under test. (Production at real scale would add
  jitter to avoid thundering herds — a deliberate, documented trade-off.)
- **Benign alerts still carry a MITRE technique** (the technique the activity *resembles*);
  benign-ness is expressed through `severity` + `recommended_action` (e.g.
  `close_false_positive`). This keeps MITRE accuracy meaningful across the whole sample set.
- **Severity scored two ways** — exact and "within one level" — because being one rung off is
  materially different from a critical/informational confusion.

## Testing & quality

- `pytest` — 128 tests, **fully mocked, offline, and free**. `pytest-asyncio` for the async
  paths; `respx` to exercise the httpx threat-intel client without a network.
- One **opt-in live smoke test** (`tests/test_live.py`, marked `live`) proves the real wiring:
  skipped by default, runs only with `pytest --run-live` and a real key.
- `ruff check` + `ruff format --check` — zero lint/format errors.
- `mypy --strict` over `src` and `tests` — zero errors.
- **CI** (`.github/workflows/ci.yml`) runs ruff, `mypy --strict`, and pytest on Python 3.12.

```bash
pytest                 # offline suite
pytest --run-live      # also hit the real API (needs ANTHROPIC_API_KEY)
ruff check . && ruff format --check . && mypy
```

## Known simplifications

This is a portfolio project; the threat-intel and history stores are bundled offline fixtures,
not live feeds. The `HttpxIpReputationClient` shows (and tests) the real HTTP seam but the
default wiring uses the offline store. `query_recent_alerts` is seeded from genuine cross-alert
overlap in the sample set — an observable that recurs across two or more alerts gets one
prior-sighting row, while observables unique to a single alert stay novel. The eval and
experiment runs use a clock anchored to the dataset (just after the newest alert), so this tool's
results are deterministic and independent of the calendar date; the live server uses real time.
Swapping in real feeds is a matter of providing
different `IpReputationClient` / `AlertHistoryStore` / `HashIntelStore` implementations — the
injection points already exist.

## License

MIT.
