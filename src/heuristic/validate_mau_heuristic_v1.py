#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Validate MAU-Heuristic v1 smoke or full run.")
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--start-date", default="2025-07-26")
    p.add_argument("--days", required=True, type=int)
    p.add_argument("--from-user", required=True, type=int)
    p.add_argument("--to-user", required=True, type=int)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.project_root.resolve()
    db = root / "db.sqlite3"
    audit = root / "paper_exports_final" / "mau_heuristic_v1_audit.csv"
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    errors = []
    warnings = []
    if not db.exists():
        raise SystemExit(f"Missing database: {db}")
    if not audit.exists():
        errors.append(f"missing_audit_log={audit}")

    con = sqlite3.connect(db)
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    if str(integrity).lower() != "ok":
        errors.append(f"sqlite_integrity={integrity}")

    start = date.fromisoformat(args.start_date)
    end = start + timedelta(days=args.days - 1)
    usernames = [f"USim{i}" for i in range(args.from_user, args.to_user + 1)]
    marks = ",".join("?" for _ in usernames)
    rows = con.execute(
        f"SELECT id, username FROM auth_user WHERE username IN ({marks})", usernames
    ).fetchall()
    user_map = {name: int(uid) for uid, name in rows}
    missing = sorted(set(usernames) - set(user_map))
    if missing:
        errors.append(f"missing_users={missing}")

    summaries = []
    schedule_export = []
    for username in usernames:
        uid = user_map.get(username)
        if uid is None:
            continue
        schedule_rows = con.execute(
            """
            SELECT date, actiontype, duringtime, bookname, startpage, endpage
            FROM study_dailyschedule
            WHERE user_id=? AND date BETWEEN ? AND ?
            ORDER BY date, id
            """,
            (uid, start.isoformat(), end.isoformat()),
        ).fetchall()
        by_day = defaultdict(list)
        for row in schedule_rows:
            by_day[str(row[0])].append(row)
            schedule_export.append({
                "username": username,
                "date": row[0],
                "actiontype": row[1],
                "duringtime": row[2],
                "bookname": row[3],
                "startpage": row[4],
                "endpage": row[5],
            })

        missing_days = []
        no_study_days = []
        invalid_study_rows = 0
        for offset in range(args.days):
            day = (start + timedelta(days=offset)).isoformat()
            day_rows = by_day.get(day, [])
            if not day_rows:
                missing_days.append(day)
                continue
            studies = [r for r in day_rows if str(r[1] or "").strip() == "مطالعه"]
            if not studies:
                no_study_days.append(day)
            for r in studies:
                if not r[3] or r[4] is None or r[5] is None or int(r[4]) < 1 or int(r[5]) < int(r[4]):
                    invalid_study_rows += 1

        if missing_days:
            errors.append(f"{username}:missing_schedule_days={len(missing_days)}")
        if no_study_days:
            errors.append(f"{username}:no_study_days={len(no_study_days)}")
        if invalid_study_rows:
            errors.append(f"{username}:invalid_study_rows={invalid_study_rows}")

        summaries.append({
            "username": username,
            "schedule_rows": len(schedule_rows),
            "covered_days": len(by_day),
            "missing_days": len(missing_days),
            "no_study_days": len(no_study_days),
            "invalid_study_rows": invalid_study_rows,
        })

    audit_rows = []
    if audit.exists():
        with audit.open("r", encoding="utf-8-sig", newline="") as fh:
            audit_rows = list(csv.DictReader(fh))
        expected = len(usernames) * args.days
        if len(audit_rows) != expected:
            errors.append(f"audit_rows={len(audit_rows)} expected={expected}")
        for row in audit_rows:
            if row.get("protocol_version") != "MAU_HEURISTIC_V1_LOCKED":
                errors.append("audit_protocol_version_mismatch")
                break
            if row.get("actor_used") != "0" or row.get("critic_used") != "0" or row.get("checkpoint_used") != "0":
                errors.append("rl_component_flag_not_zero")
                break

    forbidden = []
    for pattern in [
        "dma_dsac_checkpoints/**/*",
        "**/actor.weights.h5",
        "**/critic.weights.h5",
        "**/target.weights.h5",
        "paper_exports_final/dma_dsac_training_log*.csv",
    ]:
        forbidden.extend(root.glob(pattern))
    if forbidden:
        errors.append("forbidden_rl_artifacts=" + ",".join(str(x) for x in forbidden[:10]))

    with (out / "mau_heuristic_user_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()) if summaries else ["username"])
        writer.writeheader()
        writer.writerows(summaries)

    with (out / "mau_heuristic_schedule_export.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        fields = ["username", "date", "actiontype", "duringtime", "bookname", "startpage", "endpage"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(schedule_export)

    report = {
        "technical_status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "database": str(db),
        "audit_log": str(audit),
        "start_date": start.isoformat(),
        "days": args.days,
        "users": usernames,
        "audit_rows": len(audit_rows),
        "errors": errors,
        "warnings": warnings,
    }
    (out / "mau_heuristic_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("MAU-HEURISTIC V1 VALIDATION")
    print("=" * 72)
    print("Technical status:", report["technical_status"])
    print("Audit rows:", len(audit_rows))
    if errors:
        for err in errors:
            print("ERROR:", err)
        raise SystemExit(2)
    print("No Actor/Critic/checkpoint artifacts detected: YES")
    print("All requested user-days have schedules and study: YES")
    print("Output directory:", out)


if __name__ == "__main__":
    main()
