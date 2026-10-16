#!/usr/bin/env python3
"""
V27 perfect-adherence audit (read-only).

Checks, for each selected USim user and each pilot day:
1) a schedule exists;
2) planned vs realized study-event counts match;
3) planned vs realized exam-event counts match;
4) planned vs realized non-study/non-exam activity counts match;
5) planned vs realized study book/page-range multisets match;
6) planned vs realized exam book/page-range multisets match;
7) planned vs realized activity-type multisets match.

This is a supplemental audit for the five perfect-adherence users. It does not
modify the database.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import date, timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only exact planned-versus-realized audit for v27 perfect-adherence users."
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--start-date", default="2025-07-26")
    parser.add_argument("--days", required=True, type=int)
    parser.add_argument("--from-user", default=96, type=int)
    parser.add_argument("--to-user", default=100, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def counter_text(counter: Counter) -> str:
    return json.dumps(
        sorted(
            [
                {"key": list(key) if isinstance(key, tuple) else key, "count": count}
                for key, count in counter.items()
            ],
            key=lambda item: json.dumps(item["key"], ensure_ascii=False),
        ),
        ensure_ascii=False,
    )


def main() -> None:
    args = parse_args()
    db = args.db.resolve()
    output_dir = args.output_dir.resolve()

    if not db.is_file():
        raise SystemExit(f"Database not found: {db}")
    if args.days < 1:
        raise SystemExit("--days must be at least 1")
    if args.from_user > args.to_user:
        raise SystemExit("--from-user cannot exceed --to-user")

    output_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start_date)
    end = start + timedelta(days=args.days - 1)
    usernames = [f"USim{i}" for i in range(args.from_user, args.to_user + 1)]

    connection = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if str(integrity).lower() != "ok":
        raise SystemExit(f"SQLite integrity check failed: {integrity}")

    placeholders = ",".join("?" for _ in usernames)
    user_rows = connection.execute(
        f"SELECT id, username FROM auth_user WHERE username IN ({placeholders})",
        usernames,
    ).fetchall()
    user_map = {str(row["username"]): int(row["id"]) for row in user_rows}
    missing = sorted(set(usernames) - set(user_map))
    if missing:
        raise SystemExit(f"Missing users: {missing}")

    daily_rows: list[dict] = []
    violations: list[dict] = []

    for username in usernames:
        user_id = user_map[username]
        current = start

        while current <= end:
            day_text = current.isoformat()

            schedule_rows = connection.execute(
                """
                SELECT actiontype, bookname, startpage, endpage
                FROM study_dailyschedule
                WHERE user_id=? AND date=?
                ORDER BY id
                """,
                (user_id, day_text),
            ).fetchall()

            planned_study = [
                row for row in schedule_rows
                if str(row["actiontype"]).strip() == "مطالعه"
            ]
            planned_exam = [
                row for row in schedule_rows
                if str(row["actiontype"]).strip() == "آزمون"
            ]
            planned_activity = [
                row for row in schedule_rows
                if str(row["actiontype"]).strip() not in {"مطالعه", "آزمون"}
            ]

            actual_study = connection.execute(
                """
                SELECT b.name AS bookname, s.az_safhe AS startpage, s.ta_safhe AS endpage
                FROM study_study_model s
                JOIN study_books_model b ON b.id=s.book_id
                WHERE s.user_id=? AND s.tarikh_en_user=?
                ORDER BY s.id
                """,
                (user_id, day_text),
            ).fetchall()

            actual_exam = connection.execute(
                """
                SELECT b.name AS bookname, e.az_safhe AS startpage, e.ta_safhe AS endpage
                FROM study_azmoon_model e
                JOIN study_books_model b ON b.id=e.book_id
                WHERE e.user_id=? AND e.tarikh_en_user=?
                ORDER BY e.id
                """,
                (user_id, day_text),
            ).fetchall()

            actual_activity = connection.execute(
                """
                SELECT text_type_faliat AS actiontype
                FROM study_faliat_model
                WHERE user_id=? AND tarikh_en_user=?
                ORDER BY id
                """,
                (user_id, day_text),
            ).fetchall()

            planned_study_ranges = Counter(
                (
                    str(row["bookname"]),
                    int(row["startpage"] or 0),
                    int(row["endpage"] or 0),
                )
                for row in planned_study
            )
            actual_study_ranges = Counter(
                (
                    str(row["bookname"]),
                    int(row["startpage"] or 0),
                    int(row["endpage"] or 0),
                )
                for row in actual_study
            )

            planned_exam_ranges = Counter(
                (
                    str(row["bookname"]),
                    int(row["startpage"] or 0),
                    int(row["endpage"] or 0),
                )
                for row in planned_exam
            )
            actual_exam_ranges = Counter(
                (
                    str(row["bookname"]),
                    int(row["startpage"] or 0),
                    int(row["endpage"] or 0),
                )
                for row in actual_exam
            )

            planned_activity_types = Counter(
                str(row["actiontype"]).strip() for row in planned_activity
            )
            actual_activity_types = Counter(
                str(row["actiontype"]).strip() for row in actual_activity
            )

            schedule_exists = len(schedule_rows) > 0
            study_count_match = len(planned_study) == len(actual_study)
            exam_count_match = len(planned_exam) == len(actual_exam)
            activity_count_match = len(planned_activity) == len(actual_activity)
            study_ranges_match = planned_study_ranges == actual_study_ranges
            exam_ranges_match = planned_exam_ranges == actual_exam_ranges
            activity_types_match = planned_activity_types == actual_activity_types

            checks = {
                "missing_schedule": schedule_exists,
                "study_count_mismatch": study_count_match,
                "exam_count_mismatch": exam_count_match,
                "activity_count_mismatch": activity_count_match,
                "study_range_multiset_mismatch": study_ranges_match,
                "exam_range_multiset_mismatch": exam_ranges_match,
                "activity_type_multiset_mismatch": activity_types_match,
            }

            for violation_name, passed in checks.items():
                if not passed:
                    violations.append(
                        {
                            "username": username,
                            "date": day_text,
                            "violation": violation_name,
                            "planned_study": len(planned_study),
                            "actual_study": len(actual_study),
                            "planned_exam": len(planned_exam),
                            "actual_exam": len(actual_exam),
                            "planned_activity": len(planned_activity),
                            "actual_activity": len(actual_activity),
                        }
                    )

            daily_rows.append(
                {
                    "username": username,
                    "date": day_text,
                    "schedule_total": len(schedule_rows),
                    "planned_study": len(planned_study),
                    "actual_study": len(actual_study),
                    "planned_exam": len(planned_exam),
                    "actual_exam": len(actual_exam),
                    "planned_activity": len(planned_activity),
                    "actual_activity": len(actual_activity),
                    "study_count_match": study_count_match,
                    "exam_count_match": exam_count_match,
                    "activity_count_match": activity_count_match,
                    "study_ranges_match": study_ranges_match,
                    "exam_ranges_match": exam_ranges_match,
                    "activity_types_match": activity_types_match,
                    "planned_study_ranges": counter_text(planned_study_ranges),
                    "actual_study_ranges": counter_text(actual_study_ranges),
                    "planned_exam_ranges": counter_text(planned_exam_ranges),
                    "actual_exam_ranges": counter_text(actual_exam_ranges),
                    "planned_activity_types": counter_text(planned_activity_types),
                    "actual_activity_types": counter_text(actual_activity_types),
                }
            )

            current += timedelta(days=1)

    connection.close()

    daily_path = output_dir / "v27_perfect_adherence_daily_audit.csv"
    violation_path = output_dir / "v27_perfect_adherence_violations.csv"
    report_path = output_dir / "V27_PERFECT_ADHERENCE_REPORT.txt"
    json_path = output_dir / "V27_PERFECT_ADHERENCE_REPORT.json"

    with daily_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(daily_rows[0].keys()))
        writer.writeheader()
        writer.writerows(daily_rows)

    violation_fields = [
        "username",
        "date",
        "violation",
        "planned_study",
        "actual_study",
        "planned_exam",
        "actual_exam",
        "planned_activity",
        "actual_activity",
    ]
    with violation_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=violation_fields)
        writer.writeheader()
        writer.writerows(violations)

    status = "PASS" if not violations else "FAIL"
    report_data = {
        "status": status,
        "database": str(db),
        "database_sha256": sha256_file(db),
        "database_integrity": integrity,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "users": usernames,
        "user_days_expected": len(usernames) * args.days,
        "user_days_audited": len(daily_rows),
        "violations": len(violations),
    }
    json_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "V27 PERFECT-ADHERENCE EXACT AUDIT",
        "=" * 78,
        f"Status: {status}",
        f"Database: {db}",
        f"Database SHA256: {report_data['database_sha256']}",
        f"Database integrity: {integrity}",
        f"Interval: {start} to {end}",
        f"Users: {usernames[0]} to {usernames[-1]}",
        f"Expected user-days: {report_data['user_days_expected']}",
        f"Audited user-days: {report_data['user_days_audited']}",
        f"Violations: {len(violations)}",
        "",
    ]
    report_lines.extend(
        f"{row['username']} {row['date']}: {row['violation']}"
        for row in violations
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"V27 perfect-adherence audit status: {status}")
    print(f"Output directory: {output_dir}")
    print(f"Violations: {len(violations)}")

    if violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
