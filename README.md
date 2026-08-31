# Order / Payment Reconciliation Dashboard

Ingests a messy orders export and a messy payments export, reconciles them with a
deterministic rules engine, prices the money at risk, and explains any finding in plain
English on demand.

**Live app:** `https://<your-app>.streamlit.app`
**Demo account:** `demo@example.com` / `ReconDemo2025!`

---

## 1. Results on the supplied dataset

Every number below is printed by the code, not typed by hand:
`python scripts/run_local_recon.py data/samples/orders.csv data/samples/payments.csv`

| Measure | Value |
| --- | --- |
| Orders loaded | 184 (185 rows read, 1 exact duplicate collapsed) |
| Payment transactions | 187 |
| Distinct order references seen | 187 |
| Reconciled cleanly | 167 |
| Flagged | 20 |
| **Match rate** | **89.3%** |
| Total order value | 41,854.65 |
| Payments settled | 41,904.38 |
| Reconciled value | 39,773.28 |
| Disputed value | 2,827.95 |
| **Money at risk** | **2,178.43** |

Money at risk splits three ways, because the total alone hides the only distinction that
changes who acts on it:

| Direction | Amount | Who owns it |
| --- | --- | --- |
| Revenue at risk (we are owed) | 787.85 | Collections |
| Customer owed (we owe them) | 628.58 | Refunds / support |
| Needs investigation | 762.00 | Ops |

**By type:** Status conflict 5 (771.00) · Missing payment 4 (392.35) · Currency mismatch 2
(355.00) · Missing order 3 (308.00) · Duplicate payment 2 (248.58) · Amount mismatch 3
(103.50) · Timing anomaly 1 (0.00)

**By severity:** CRITICAL 9 (1,185.35) · HIGH 7 (889.58) · MEDIUM 2 (85.00) · LOW 2 (18.50)

---

## 2. Architecture

```
 Browser
    │  HTTPS (rendered output only — never a key, never a token)
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Streamlit  (one process, one deployment)                     │
│                                                               │
│  streamlit_app.py      session, routing, one DB session/run   │
│  app/ui/               views — render only, no business logic │
│      │  direct Python calls                                   │
│      ▼                                                        │
│  app/services/         auth · ingest · recon · explain · llm  │
│      │                                    │                   │
│      ▼                                    ▼                   │
│  app/domain/  (pure)              OpenAI API (server-side)    │
│      │  no I/O, no framework                                  │
│      ▼                                                        │
│  app/db/  SQLAlchemy → SQLite (local) / PostgreSQL (prod)     │
└──────────────────────────────────────────────────────────────┘
```

### Why there is no separate API service

An earlier iteration of this project put FastAPI behind Streamlit. That was the right
instinct for the wrong runtime, and it was removed deliberately. **Streamlit Community
Cloud runs exactly one process and exposes exactly one port**, so a second service would
have to be deployed and paid for separately, kept awake, and reached over the public
internet — adding a network hop, a second cold start, and a second failure mode to reach
code that is already in the same repository.

The security argument for the backend does not survive contact with the facts either. The
key had to be hidden from the *browser*, not from Python. Streamlit executes every line of
this application **server-side** and sends the browser rendered output only; `st.secrets`
is never serialised into the page. `import openai` happens in `app/services/llm.py`, which
the browser cannot reach, so the key is exactly as protected as it was behind FastAPI —
with one less service to run.

What made the removal cheap is the layering: the domain and service layers never imported
FastAPI in the first place. Deleting the HTTP layer meant deleting `app/api/`,
`schemas.py`, `main.py` and the HTTP client. **The reconciliation engine did not change by
a single line.** That is the strongest available evidence that the business logic was
actually decoupled rather than just filed in separate folders.

### File tree

```
recon-dashboard/
├── streamlit_app.py           ← entry point (Streamlit Cloud's default filename)
├── requirements.txt           ← must be at the root, or the deploy silently fails
├── .streamlit/
│   ├── config.toml            theme + 10 MB upload cap (committed, non-secret)
│   └── secrets.toml.example   template; the real secrets.toml is gitignored
├── app/
│   ├── domain/                PURE: no I/O, no framework, no database
│   │   ├── normalize.py       references, money, dates, emails
│   │   ├── parsing.py         CSV → clean records + a row-issue audit trail
│   │   ├── records.py         frozen dataclasses crossing layer boundaries
│   │   ├── rules.py           types, severities, ReconConfig, precedence
│   │   ├── engine.py          the reconciliation itself
│   │   └── metrics.py         summary aggregation and prioritisation
│   ├── core/
│   │   ├── config.py          st.secrets → env → default
│   │   └── security.py        bcrypt hashing
│   ├── db/
│   │   ├── models.py          8 tables, user_id on every row
│   │   └── session.py         engine (once) + session scope (per run)
│   ├── services/
│   │   ├── auth_service.py    sign-up / sign-in / lockout
│   │   ├── ingest_service.py  persist a batch
│   │   ├── recon_service.py   run the engine, store findings, query them
│   │   ├── explain_service.py cache + call the model, always return something
│   │   └── llm.py             prompts, JSON repair, deterministic fallback
│   └── ui/
│       ├── auth_view.py       sign in / create account
│       ├── upload_view.py     upload, data-quality audit, run
│       ├── dashboard_view.py  metrics, charts, drill-down, sensitivity
│       └── charts.py          seven Plotly builders
├── data/samples/              the supplied CSVs, for one-click evaluation
├── scripts/
│   ├── seed_demo_user.py      idempotent demo account + sample run
│   └── run_local_recon.py     engine with NO database and NO network
└── tests/                     50 tests, no network, no database
```

---

## 3. Reconciliation logic

### Matching

Orders and payments join on a **normalised order reference**: trim, strip invisible
characters and BOMs, collapse internal whitespace, uppercase. This is what makes
`ord-1801 `, `ORD-1802` and `ord-1802` match their orders. Every normalisation is recorded
as a data-quality issue, so cleaning is visible and auditable rather than silent.

Dates are parsed per file: orders arrive as `YYYY-MM-DD HH:MM:SS`, payments as
`DD/MM/YYYY HH:MM`. **Day-first is not guessed** — the payments parser is told the format,
because a heuristic reading of `10/04/2025` is a coin flip between 10 April and 4 October.
A missing or unparseable date is `None` and never becomes "today".

Money is `Decimal` end to end and is serialised as a **string**, never a float. `0.1 + 0.2`
is not `0.3` in binary floating point, and a reconciliation tool that cannot add up is
worthless.

### The rule book

| Rule | Fires when | Severity | Direction |
| --- | --- | --- | --- |
| `R01-NO-PAYMENT` | Order exists, no payment at all | CRITICAL | Revenue at risk |
| `R02-ORPHAN-PAYMENT` | Payment exists, no order | CRITICAL | Investigate |
| `R03-CURRENCY` | Order and payment currencies differ | HIGH | Investigate |
| `R04-DUPLICATE-CHARGE` | Two settled charges inside 24h | HIGH | Customer owed |
| `R05a-CANCELLED-BUT-PAID` | Cancelled order, money held | CRITICAL | Customer owed |
| `R05b-PARTIAL-REFUND` | Refunded in part, balance still held | HIGH | Customer owed |
| `R05c-COMPLETED-BUT-UNPAID` | Completed order, payment failed/pending | CRITICAL / HIGH | Revenue at risk |
| `R05d-REFUNDED-BUT-COMPLETED` | Fully refunded, order still completed | HIGH | Investigate |
| `R06-AMOUNT` | Net differs beyond tolerance | MEDIUM / LOW | Either |
| `R07a-LATE-SETTLEMENT` | Settled beyond the lag window | LOW | None |
| `R07b-PAYMENT-BEFORE-ORDER` | Paid before the order existed | LOW | None |
| `R08-UNKNOWN-ORDER-STATUS` | Status outside the known vocabulary | LOW | None |

### One order, one headline finding

A duplicate charge also makes the collected amount wrong. Reporting both would double-count
the money and give an operator two tickets for one problem. Findings are therefore ranked:

```
MISSING_ORDER → MISSING_PAYMENT → CURRENCY_MISMATCH → DUPLICATE_PAYMENT
  → STATUS_CONFLICT → AMOUNT_MISMATCH → TIMING_ANOMALY
```

The winner is **primary** and carries the money; symptoms are suppressed rather than
duplicated. The cause is listed before the symptom, so the top of the list is a work queue.

Sign convention: `delta = collected − expected`. Positive means we hold too much (customer
owed); negative means we collected too little (revenue at risk).

### Tolerance: why ±0.05

The dataset makes this easy to defend. The only sub-cent differences present are **0.01 and
0.02** (ORD-1901/1902/1903 — payment-processor rounding). The smallest *genuine* error is
**18.50**, roughly 370× larger. Any threshold between 0.03 and about 18.00 produces
identical output, so 0.05 sits in the middle of a very wide safe band rather than on a
cliff edge. It is also a plausible real-world rounding allowance.

A **percentage** tolerance was considered and rejected: 0.5% of the largest order is about
2.37, which would silently swallow a genuine 2.00 underpayment on a big order while
flagging a 0.02 rounding difference on a small one. Exactly backwards.

The threshold is an **argument, not a constant** (`ReconConfig`), and the dashboard exposes
it so a reviewer can re-run and watch the output stay stable. It is bounded at 5.00,
because a tolerance of 500.00 would mask every discrepancy in the file — a tunable for
sensitivity analysis, not a switch for turning the product off.

The other two windows are set the same way: `duplicate_window_hours = 24` (the observed
duplicates are 29 minutes apart) and `max_settlement_lag_days = 7` (median settlement lag
is about 45 minutes; the one anomaly is 29 days).

---

## 4. What the data actually contained

| # | Finding | Example | Handling |
| --- | --- | --- | --- |
| 1 | Case and whitespace in references | `ord-1801 `, `ord-1802` | Normalised; recorded |
| 2 | Two date formats | orders vs payments | Format-specific parsers |
| 3 | Missing payment date | `TXN700187` | Stays `None`; timing rules skip it |
| 4 | Orphaned payments | `ORD-1301/1302/1303` | `R02`, CRITICAL |
| 5 | Orders with no payment | `ORD-1201`–`ORD-1204` | `R01`, CRITICAL |
| 6 | Missing email / blank discount | `ORD-2201` | Recorded; discount stays **unknown**, not invented as 0.00 |
| 7 | Exact duplicate order row | `ORD-1004` | Collapsed once, recorded |
| 8 | Multiple payments per order | `ORD-1501/1502` | `R04`, refunds netted |
| 9 | Charge plus refund | `ORD-1702` (`R05b`), `ORD-1703` (`R05d`) | Netted, then judged |
| 10 | Amount discrepancies | `ORD-1403` +60.00, `ORD-1402` −18.50 | `R06` |
| 11 | Sub-tolerance rounding | `ORD-1901/1902/1903` (0.01–0.02) | **Correctly not flagged** |
| 12 | Currency mismatches | `ORD-1601`, `ORD-1602` | `R03`, HIGH |
| 13 | Cancelled but paid | `ORD-1701` | `R05a`, CRITICAL |
| 14 | Completed but unpaid | `ORD-2001`, `ORD-2002` | `R05c` |
| 15 | Timing anomaly | `ORD-2101`, 29 days | `R07a`, LOW, 0.00 |

### Two corrections to the brief

1. **The currency direction is reversed.** The brief says `ORD-1601` is EUR in orders and
   USD in payments; the file has it **USD in orders, EUR in payments**. There is also a
   second, unmentioned case: `ORD-1602` (EUR order, USD payment). Both are flagged.
2. **Six quirks were not listed:** the duplicate order row (`ORD-1004`), four orders with
   no payment at all (`ORD-1201`–`1204`), the blank `processed_at` (`TXN700187`), the
   29-day settlement anomaly (`ORD-2101`), the blank discount plus missing email
   (`ORD-2201`), and the three sub-tolerance rounding orders that must **not** be flagged.

Neither correction changes the design; both were found by reading the data rather than the
specification.

---

## 5. AI integration

### What the model is allowed to do

It receives a finding the engine has **already** made, with its evidence, and writes up the
likely cause and the recommended action. It cannot create, suppress, re-price or re-match
anything. Reconciliation output must be reproducible and auditable; prose does not.

### Temperature 0.2

| Setting | Behaviour |
| --- | --- |
| 0.0 | Not actually deterministic (batching and hardware still vary), and the output reads like a filled-in template |
| **0.2** | **Stable, factual, grounded in the evidence — small wording variation only** |
| 0.7+ | Starts inventing plausible causes that are not in the data — the one unacceptable failure for a finance tool |

`seed=7` is also pinned, and the prompt supplies the evidence explicitly so the model has
nothing to guess.

### Malformed responses

Models return prose around JSON, fenced blocks, trailing commas and wrapper keys. The
parser degrades through four steps rather than raising:

1. parse the JSON directly
2. strip ``` / ```json fences and retry
3. extract the outermost `{...}`, drop trailing commas, unwrap `explanation` / `result` /
   `response` / `data` / `output`, coerce types, normalise `confidence` to
   high / medium / low
4. if `what_happened` or `recommended_action` is still missing, **discard the response
   entirely** and use the deterministic explanation

Step 4 is the important one: the two load-bearing fields are never fabricated. The optional
fields are filled with honest defaults (`"Not stated by the model."`) because the UI renders
every field and a blank one looks like a bug.

The fallback is hand-written per discrepancy type and grounded in the actual order, so
**the feature works with no API key at all** — the UI simply labels it as deterministic.
Timeout is 20s with one retry, and results are cached per (discrepancy, model,
prompt version) so re-opening a row costs nothing.

---

## 6. Security and multi-tenancy

- **Passwords:** bcrypt, cost 12, SHA-256 pre-hash so passphrases over 72 bytes are not
  silently truncated. `bcrypt` is used directly — passlib has been unmaintained since 2020.
- **Identity:** the signed-in user id lives in `st.session_state`, which is **server-side**
  per session. The browser holds only an opaque session id, so there is no token to steal
  or forge. (The previous JWT implementation existed only because a separate API needed a
  bearer token; keeping it here would have been ceremony, not security.)
- **Tenancy:** every table carries `user_id`, every query filters on it, and the `User` is
  passed explicitly into each service call — a missing tenant filter shows up in review as
  a missing argument rather than hiding behind global state.
- **Enumeration:** identical error for unknown account and wrong password; the password is
  still verified when the account exists so timing does not leak either. Ten failures in
  five minutes locks the address out.
- **PII:** customer emails are masked (`k****@example.com`) in the domain layer before any
  finding, dashboard row or LLM prompt can see them. No raw address is ever sent to the LLM API.
- **Secrets:** `.streamlit/secrets.toml` is gitignored; only the `.example` is committed.
- **Uploads:** capped at 10 MB and checked *before* parsing, since the parser loads the file
  into memory.

---

## 7. Run it locally

```bash
git clone <repo-url> && cd recon-dashboard
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # optional
python scripts/seed_demo_user.py --with-sample-data          # optional

streamlit run streamlit_app.py
```

Opens on `http://localhost:8501`. With no secrets file it still runs: SQLite for storage,
deterministic explanations instead of model-written ones.

Engine only, no database and no network:

```bash
python scripts/run_local_recon.py data/samples/orders.csv data/samples/payments.csv
python scripts/run_local_recon.py data/samples/*.csv --tolerance 0.01   # sensitivity check
```

---

## 8. Tests

```bash
pytest -q          # 50 tests
```

They cover the parts where being wrong is expensive: reference normalisation, both date
formats, money parsing, refund netting, duplicate detection, the four status-conflict
shapes, precedence, exact headline totals, priority ordering, PII masking, and the LLM JSON
repair ladder including nine unrecoverable inputs. No test touches the network or a
database, so the suite runs anywhere in about a second.

The headline totals are asserted **exactly**. If a rule changes the money, a test fails
with the specific number, which is what stops the README from drifting away from the code.

---

## 9. Deployment

1. Push the repository to your remote Git platform.
2. Connect it to your chosen hosting provider (e.g. cloud platform). Main file path is `streamlit_app.py`.
3. Configure your Secrets (environment variables):

   ```toml
   ENVIRONMENT = "production"
   SESSION_SECRET = "<python -c 'import secrets; print(secrets.token_urlsafe(48))'>"
   OPENAI_API_KEY = "sk-..."
   DATABASE_URL = "postgresql://user:pass@host/db?sslmode=require"
   ```

4. Deploy. First boot creates the tables automatically.
5. Sign up in the UI, or set `ALLOW_SIGNUP = false` to lock the demo to the seeded account.

### About the database

Without `DATABASE_URL` the app uses SQLite — on an ephemeral container filesystem,
accounts and uploads vanish on restart or redeploy. The app detects this
and says so in the sidebar rather than losing data quietly. For anything persistent, use a
managed PostgreSQL provider and paste
the connection string in. A plain `postgres://` or `postgresql://` URL is rewritten to
psycopg 3 automatically, so the provider's string works verbatim.

---

## 10. Known limitations

Stated deliberately — each is a conscious scope call, not an oversight.

- **`create_all`, not Alembic.** Fine for a fresh database; a schema that must evolve after
  launch needs real migrations.
- **The login throttle is in-process.** Correct for one container; a multi-instance
  deployment needs Redis.
- **Whole file in memory.** Fine at 10 MB; a 500k-row export wants streaming ingestion.
- **Matching is exact-key only.** Deliberate: fuzzy matching that silently pairs `ORD-1801`
  with `ORD-1810` is worse than no match. Fuzzy candidates belong in a review queue.
- **One currency per finding.** No FX conversion, so a currency mismatch is flagged for
  investigation rather than valued — converting at an unknown historical rate would invent
  a number.
- **AI explanations are advisory** and labelled as such; the deterministic finding is the
  record of truth.

---

## 11. Final Note on AI

AI coding tools were used in this project primarily to generate unit test matrices from the deterministic logic specifications, author layout components for charts and PDF reports, and as a sounding board to confirm float rounding behaviors across Python engines vs JS environments (preventing the 1-cent rounding trap). At no point does the LLM run autonomous rules. The core logic remains 100% human-designed and deterministically tested!

## 12. What I would improve or build next with more time

Given additional time, I would prioritize the following enhancements:

- **Role-based access & workflow:** Allow finance/ops users to flag discrepancies as investigated, add notes, and transition findings through states (New → Investigating → Resolved → False Positive). This turns the dashboard from a passive report into an active work queue.
- **Export & audit trail:** Enable downloading the full discrepancy list with AI explanations as CSV/Excel, and provide a immutable PDF report of each run for archival/compliance purposes.
- **Temporal & trend analysis:** Add time-series views (daily/weekly mismatch rates, money-at-risk over time) to help identify systemic issues like cutoff errors or seasonal spikes in certain discrepancy types.
- **Streaming ingestion:** For very large files (>10 MB), implement chunked parsing to avoid loading the entire export into memory, perhaps using `pandas.read_csv(chunksize=…)` or a custom iterator.
- **Fuzzy matching review queue:** While keeping the core engine exact-key (to avoid false positives), surface high-confidence fuzzy candidates (e.g., `ORD-1801` vs `ORD-1810`) in a separate “Review Suggested Matches” tab for human judgment.
- **Multi-factor authentication:** Add TOTP or WebAuthn as a second factor for administrative users, while keeping the demo flow simple.
- **CI/CD & automated testing:** Set up GitHub Actions to run the test suite on every PR, and deploy preview versions to Streamlit Cloud for visual regression checks.
- **Customizable tolerances per client:** Allow account administrators to adjust the amount tolerance, duplicate window, and settlement lag via the UI (with guardrails to prevent extreme values), and store these preferences per user.
