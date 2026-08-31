"""
run_local_recon.py -- Run the reconciliation engine over two CSV files with NO database,
NO API and NO network. Pure domain layer.

Why this script exists (and why it is worth mentioning in the interview):
  * It proves the engine is a pure function -- if it needs a web server to run, it is not.
  * It is how the findings quoted in the README were produced, so the README cannot drift
    from the code.
  * It gives a reviewer a 2-second way to sanity check the app without signing up.

Usage:
    python scripts/run_local_recon.py data/samples/orders.csv data/samples/payments.csv
    python scripts/run_local_recon.py orders.csv payments.csv --json > findings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.domain.engine import reconcile  # noqa: E402
from app.domain.metrics import build_summary, summarise_row_issues, top_priorities  # noqa: E402
from app.domain.parsing import parse_orders_csv, parse_payments_csv  # noqa: E402
from app.domain.rules import ReconConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile two CSV exports locally.")
    parser.add_argument("orders_csv", type=Path)
    parser.add_argument("payments_csv", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a report.")
    parser.add_argument("--tolerance", default=None, help="Override the amount tolerance, e.g. 0.01")
    args = parser.parse_args()

    config = ReconConfig()
    if args.tolerance is not None:
        from decimal import Decimal
        config = ReconConfig(amount_tolerance=Decimal(args.tolerance))

    orders_parse = parse_orders_csv(args.orders_csv.read_bytes())
    payments_parse = parse_payments_csv(args.payments_csv.read_bytes())
    issues = orders_parse.issues + payments_parse.issues

    output = reconcile(orders_parse.orders, payments_parse.payments, config, issues)
    summary = build_summary(output)

    if args.json:
        print(json.dumps({
            "summary": summary,
            "findings": [
                {
                    "order_key": f.order_key,
                    "type": f.discrepancy_type.value,
                    "severity": f.severity.value,
                    "rule_id": f.rule_id,
                    "is_primary": f.is_primary,
                    "amount_at_risk": str(f.amount_at_risk),
                    "summary": f.summary,
                }
                for f in output.findings
            ],
        }, indent=2))
        return 0

    w = sys.stdout.write
    w("\n" + "=" * 78 + "\n  RECONCILIATION REPORT\n" + "=" * 78 + "\n")
    w(f"  orders file    : {args.orders_csv}  ({orders_parse.rows_read} rows read, "
      f"{orders_parse.rows_loaded} loaded, {orders_parse.duplicate_rows_collapsed} duplicate rows collapsed)\n")
    w(f"  payments file  : {args.payments_csv}  ({payments_parse.rows_read} rows read, "
      f"{payments_parse.rows_loaded} loaded)\n\n")

    w("  HEADLINES\n")
    w(f"    orders reconciled      : {summary['total_reconciled_keys']} / "
      f"{summary['total_reconciled_keys'] + summary['total_flagged_keys']}  "
      f"({summary['match_rate_pct']}% clean)\n")
    w(f"    total order value      : {summary['total_order_value']}\n")
    w(f"    total settled payments : {summary['total_payments_settled']}\n")
    w(f"    reconciled value       : {summary['reconciled_value']}\n")
    w(f"    disputed value         : {summary['disputed_value']}\n")
    w(f"    MONEY AT RISK          : {summary['money_at_risk']}\n")
    w(f"      - revenue at risk    : {summary['revenue_at_risk']}\n")
    w(f"      - owed to customers  : {summary['customer_owed']}\n")
    w(f"      - needs investigation: {summary['needs_investigation']}\n\n")

    w("  BY TYPE\n")
    for row in summary["by_type"]:
        flag = "" if row["is_financial"] else "   (review only)"
        w(f"    {row['label']:<20} count={row['count']:<4} at_risk={row['amount_at_risk']:>10}{flag}\n")

    w("\n  BY SEVERITY\n")
    for row in summary["by_severity"]:
        w(f"    {row['severity']:<10} count={row['count']:<4} at_risk={row['amount_at_risk']:>10}\n")

    w("\n  WORK THIS FIRST\n")
    for r in top_priorities(output, 12):
        w(f"    [{(r.primary_severity.value if r.primary_severity else '-'):<8}] "
          f"{r.order_key:<10} {r.amount_at_risk:>9}  "
          f"{(r.primary_type.value if r.primary_type else '-'):<18} "
          f"{(r.findings[0].summary if r.findings else '')}\n")

    w("\n  EVERY FLAGGED ORDER (deterministic order)\n")
    for r in sorted((x for x in output.results if not x.is_reconciled), key=lambda x: x.order_key):
        types = ",".join(sorted({f.discrepancy_type.value for f in r.findings}))
        w(f"    {r.order_key:<10} expected={r.expected_amount:>9} collected={r.collected_amount:>9} "
          f"delta={r.delta:>9} risk={r.amount_at_risk:>9}  {types}\n")

    w("\n  DATA QUALITY (row-level, kept out of the money figures)\n")
    for entry in summarise_row_issues(issues):
        w(f"    {entry['code']:<32} count={entry['count']:<4} dropped={entry['dropped']}\n")
    w("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
