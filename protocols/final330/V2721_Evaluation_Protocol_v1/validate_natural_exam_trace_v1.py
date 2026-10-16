#!/usr/bin/env python3
"""
Validate isolation and database correspondence of the natural exam trace.

Modes:
- preflight: trace must not exist or must be zero bytes
- postrun: trace must contain only the requested users/date interval and must
  match the corresponding azmoon rows in the database.

Exit codes: 0 PASS, 2 FAIL.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("preflight", "postrun"))
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--db", type=Path)
    p.add_argument("--start-date")
    p.add_argument("--days", type=int)
    p.add_argument("--from-user", type=int)
    p.add_argument("--to-user", type=int)
    return p.parse_args()


def norm_int(value):
    return int(round(float(value)))


def main():
    args = parse_args()
    trace = args.trace.resolve()

    if args.mode == "preflight":
        if trace.exists() and trace.stat().st_size > 0:
            raise RuntimeError(f"Trace already exists and is nonempty: {trace}")
        print("NATURAL EXAM TRACE PREFLIGHT: PASS")
        print(f"Trace absent/empty: {trace}")
        return 0

    required = (
        args.db,
        args.start_date,
        args.days,
        args.from_user,
        args.to_user,
    )
    if any(value is None for value in required):
        raise RuntimeError("postrun requires db/start-date/days/from-user/to-user")
    if not trace.is_file():
        raise RuntimeError(f"Trace missing: {trace}")

    start = date.fromisoformat(args.start_date)
    end = start + timedelta(days=args.days - 1)
    allowed_users = {
        f"USim{i}" for i in range(args.from_user, args.to_user + 1)
    }

    with trace.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        # Zero natural exams is allowed only if DB also has zero.
        rows = []

    out_of_scope = []
    trace_counter = Counter()
    for row in rows:
        username = str(row.get("username", "")).strip()
        d = date.fromisoformat(str(row.get("date", "")).strip()[:10])
        if username not in allowed_users or not (start <= d <= end):
            out_of_scope.append((username, d.isoformat()))
        key = (
            username,
            d.isoformat(),
            str(row.get("book", "")).strip(),
            norm_int(row.get("start_page", 0)),
            norm_int(row.get("end_page", 0)),
            norm_int(row.get("score", 0)),
        )
        trace_counter[key] += 1

    if out_of_scope:
        raise RuntimeError(f"Out-of-scope trace rows: {out_of_scope[:10]}")

    db = args.db.resolve()
    con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    integrity = str(con.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity.lower() != "ok":
        raise RuntimeError(f"DB integrity failure: {integrity}")

    marks = ",".join("?" for _ in allowed_users)
    user_rows = con.execute(
        f"SELECT id, username FROM auth_user WHERE username IN ({marks})",
        tuple(sorted(allowed_users)),
    ).fetchall()
    user_map = {int(r["id"]): str(r["username"]) for r in user_rows}
    if len(user_map) != len(allowed_users):
        raise RuntimeError("One or more requested users are missing")

    id_marks = ",".join("?" for _ in user_map)
    exam_rows = con.execute(
        f"""
        SELECT a.user_id, DATE(a.tarikh_en_user) AS exam_date,
               b.name AS book, a.az_safhe, a.ta_safhe, a.javab_darsad
        FROM study_azmoon_model AS a
        JOIN study_books_model AS b ON b.id = a.book_id
        WHERE a.user_id IN ({id_marks})
          AND DATE(a.tarikh_en_user) BETWEEN DATE(?) AND DATE(?)
        """,
        (*user_map.keys(), start.isoformat(), end.isoformat()),
    ).fetchall()
    con.close()

    db_counter = Counter(
        (
            user_map[int(r["user_id"])],
            str(r["exam_date"]),
            str(r["book"]).strip(),
            int(r["az_safhe"]),
            int(r["ta_safhe"]),
            int(r["javab_darsad"]),
        )
        for r in exam_rows
    )

    if trace_counter != db_counter:
        missing = list((db_counter - trace_counter).elements())[:10]
        extra = list((trace_counter - db_counter).elements())[:10]
        raise RuntimeError(
            f"Trace/DB mismatch | trace={sum(trace_counter.values())} "
            f"db={sum(db_counter.values())} | missing={missing} | extra={extra}"
        )

    exact_duplicates = [key for key, count in trace_counter.items() if count > 1]
    if exact_duplicates:
        raise RuntimeError(f"Exact duplicate trace keys: {exact_duplicates[:10]}")

    print("NATURAL EXAM TRACE POSTRUN VALIDATION: PASS")
    print(f"Rows: {len(rows)}")
    print(f"Users allowed: USim{args.from_user}..USim{args.to_user}")
    print(f"Interval: {start} to {end}")
    print("Trace and database exam rows match exactly.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"NATURAL EXAM TRACE VALIDATION FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
