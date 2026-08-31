"""
seed_demo_user.py -- Create the demo account, optionally with the sample data loaded.

WHY THIS EXISTS
  A reviewer should be able to open the deployed URL and see a populated dashboard without
  creating an account or finding a CSV. This script makes that state reproducible instead
  of hand-made.

USAGE
    python scripts/seed_demo_user.py
    python scripts/seed_demo_user.py --with-sample-data
    python scripts/seed_demo_user.py --email me@example.com --password 'something long'

It is idempotent: running it twice does not create a second account and does not duplicate
the sample batch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repository root importable when this file is run directly as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.security import AuthError, hash_password, normalise_email  # noqa: E402
from app.db.models import User  # noqa: E402
from app.db.session import init_db, session_scope  # noqa: E402
from app.services import ingest_service, recon_service  # noqa: E402

from sqlalchemy import func, select  # noqa: E402

DEFAULT_EMAIL = "demo@example.com"
DEFAULT_PASSWORD = "ReconDemo2025!"
SAMPLES = ROOT / "data" / "samples"
SAMPLE_LABEL = "Sample dataset"


def seed(email: str, password: str, with_sample_data: bool) -> int:
    init_db()
    address = normalise_email(email)

    with session_scope() as db:
        user = db.scalars(select(User).where(func.lower(User.email) == address)).first()
        if user is None:
            user = User(email=address, password_hash=hash_password(password), is_active=True)
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created user {address}")
        else:
            print(f"User {address} already exists")

        if not with_sample_data:
            return 0

        existing = [b for b in ingest_service.list_batches(db, user) if b.label == SAMPLE_LABEL]
        if existing:
            print(f"Sample batch already present (#{existing[0].id})")
            return 0

        orders_csv = SAMPLES / "orders.csv"
        payments_csv = SAMPLES / "payments.csv"
        if not orders_csv.exists() or not payments_csv.exists():
            print(f"Sample CSVs not found in {SAMPLES}", file=sys.stderr)
            return 1

        batch, orders_result, payments_result = ingest_service.ingest_files(
            db, user,
            orders_csv.read_bytes(), payments_csv.read_bytes(),
            "orders.csv", "payments.csv", SAMPLE_LABEL,
        )
        print(
            f"Ingested batch #{batch.id}: "
            f"{orders_result.rows_loaded} orders, {payments_result.rows_loaded} payments"
        )

        run = recon_service.run_reconciliation(db, user, batch)
        summary = run.summary_json
        print(
            f"Run #{run.id}: {summary['total_flagged_keys']} flagged of "
            f"{summary['total_order_keys']} references "
            f"({summary['match_rate_pct']}% matched), "
            f"{summary['money_at_risk']} at risk"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the demo account.")
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument(
        "--with-sample-data", action="store_true",
        help="Also ingest data/samples and run the engine once.",
    )
    args = parser.parse_args()

    try:
        return seed(args.email, args.password, args.with_sample_data)
    except AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
