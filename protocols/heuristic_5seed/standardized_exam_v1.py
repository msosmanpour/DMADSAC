#!/usr/bin/env python3
"""
V27.2.1 Evaluation Protocol v1 — standardized paired exam engine.

This script is read-only with respect to both SQLite databases.

Primary design:
- same learners, checkpoints, books, pages, and question numbers in Full/Static
- 9 books, 10 deterministic questions per book and scope
- equal weight for every book
- no satisfaction query or use
- no existing exam-row query or use
- no nonzero fallback for unstudied content
- no random noise in the primary expected-recall outcome
- common-random-number realized MC score only as a secondary sensitivity outcome

Exit codes:
0 technical PASS
2 technical/validation FAIL
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

VERSION = "V2721-EVALUATION-PROTOCOL-V1"
BOOKS_REQUIRED = 9
QUESTIONS_PER_BOOK = 10
BASE_HALF_LIFE_DAYS = 2.0
HALF_LIFE_MIN = 0.75
HALF_LIFE_MAX = 14.0
NEUTRAL_EXAM_STRESS = 0.50
NEUTRAL_EXAM_FATIGUE = 0.50
GUESSING_BASELINE = 0.25


@dataclass(frozen=True)
class Exposure:
    study_date: date
    quality: float
    is_review: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=VERSION)
    p.add_argument("--full-db", required=True, type=Path)
    p.add_argument("--static-db", required=True, type=Path)
    p.add_argument("--start-date", required=True)
    p.add_argument("--days", required=True, type=int, choices=(30, 60, 330))
    p.add_argument(
        "--protocol",
        required=True,
        choices=("gate30", "gate60", "final330"),
    )
    p.add_argument("--from-user", type=int, default=1)
    p.add_argument("--to-user", type=int, default=100)
    p.add_argument("--master-seed", type=int, default=14040504)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def open_ro(path: Path) -> sqlite3.Connection:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    integrity = str(con.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity.lower() != "ok":
        con.close()
        raise RuntimeError(f"SQLite integrity failure: {path}: {integrity}")
    return con


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def parse_time(value: Any) -> time | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for parser in (
        lambda t: time.fromisoformat(t),
        lambda t: datetime.strptime(t, "%H:%M:%S").time(),
        lambda t: datetime.strptime(t, "%H:%M").time(),
    ):
        try:
            return parser(text)
        except ValueError:
            pass
    return None


def duration_hours(start_value: Any, end_value: Any) -> float:
    start = parse_time(start_value)
    end = parse_time(end_value)
    if start is None or end is None:
        return 0.05
    begin = datetime.combine(date(2000, 1, 1), start)
    finish = datetime.combine(date(2000, 1, 1), end)
    if finish < begin:
        finish += timedelta(days=1)
    return max((finish - begin).total_seconds() / 3600.0, 0.05)


def norm_percent(value: Any, default: float = 60.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return clamp(number / 100.0, 0.0, 1.0)


def study_quality(row: sqlite3.Row) -> float:
    """
    Satisfaction-free source-quality model.

    The 0.60 intercept keeps the neutral calibration of the prior model while
    satisfaction is neither queried nor used.
    """
    stress = norm_percent(row["mizan_esters"], 60.0)
    hours = duration_hours(row["az_saat"], row["ta_saat"])
    pages = max(1, int(row["ta_safhe"]) - int(row["az_safhe"]) + 1)
    pace = pages / max(hours, 0.05)
    pace_quality = clamp(1.0 - abs(pace - 7.0) / 14.0, 0.30, 1.0)
    review_bonus = 0.10 if "مرور" in str(row["study_type_name"] or "") else 0.0
    return clamp(0.60 - 0.20 * stress + 0.20 * pace_quality + review_bonus, 0.0, 1.0)


def usernames(first: int, last: int) -> list[str]:
    if first > last:
        raise ValueError("--from-user must be <= --to-user")
    return [f"USim{i}" for i in range(first, last + 1)]


def checkpoint_plan(start: date, days: int, protocol: str) -> list[dict[str, Any]]:
    expected = {"gate30": 30, "gate60": 60, "final330": 330}[protocol]
    if days != expected:
        raise ValueError(f"{protocol} requires --days {expected}")

    def cp(name: str, day_number: int, cumulative_fraction: float) -> dict[str, Any]:
        cutoff = start + timedelta(days=day_number - 1)
        return {
            "checkpoint": name,
            "day_number": day_number,
            "cutoff_date": cutoff,
            "assessment_date": cutoff + timedelta(days=1),
            "cumulative_fraction": cumulative_fraction,
        }

    pre = {
        "checkpoint": "PRE",
        "day_number": 0,
        "cutoff_date": start - timedelta(days=1),
        "assessment_date": start,
        "cumulative_fraction": 1.0 / 3.0,
    }
    if protocol == "gate30":
        return [pre, cp("D30", 30, 1.0 / 3.0)]
    if protocol == "gate60":
        return [
            pre,
            cp("D30", 30, 1.0 / 3.0),
            cp("D60", 60, 1.0 / 3.0),
        ]
    return [
        pre,
        cp("T1_D110", 110, 1.0 / 3.0),
        cp("T2_D220", 220, 2.0 / 3.0),
        cp("FINAL_D330", 330, 1.0),
    ]


def get_users(con: sqlite3.Connection, names: Sequence[str]) -> dict[str, int]:
    marks = ",".join("?" for _ in names)
    rows = con.execute(
        f"SELECT id, username FROM auth_user WHERE username IN ({marks})",
        tuple(names),
    ).fetchall()
    found = {str(r["username"]): int(r["id"]) for r in rows}
    missing = sorted(set(names) - set(found))
    if missing:
        raise RuntimeError(f"Missing users: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return found


def get_books(con: sqlite3.Connection, user_id: int) -> list[sqlite3.Row]:
    rows = con.execute(
        """
        SELECT id, name, tedad_safahat
        FROM study_books_model
        WHERE user_id=?
        ORDER BY name, id
        """,
        (user_id,),
    ).fetchall()
    if len(rows) != BOOKS_REQUIRED:
        raise RuntimeError(f"user_id={user_id}: expected 9 books, found {len(rows)}")
    for r in rows:
        if int(r["tedad_safahat"]) < QUESTIONS_PER_BOOK:
            raise RuntimeError(
                f"user_id={user_id}, book={r['name']}: fewer than 10 pages"
            )
    return rows


def get_studies(
    con: sqlite3.Connection,
    user_id: int,
    cutoff: date,
) -> list[sqlite3.Row]:
    # Satisfaction and exam tables are intentionally absent.
    return con.execute(
        """
        SELECT s.id, s.book_id, s.az_safhe, s.ta_safhe,
               s.az_saat, s.ta_saat, s.tarikh_en_user,
               s.mizan_esters, COALESCE(n.name, '') AS study_type_name
        FROM study_study_model AS s
        LEFT JOIN study_noe_motaleh AS n
          ON n.id = s.noeMotale_id
        WHERE s.user_id=?
          AND s.tarikh_en_user IS NOT NULL
          AND DATE(s.tarikh_en_user) <= DATE(?)
        ORDER BY s.tarikh_en_user, s.az_saat, s.id
        """,
        (user_id, cutoff.isoformat()),
    ).fetchall()


def keyed_seed(*parts: Any) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def deterministic_pages(
    master_seed: int,
    username: str,
    book_name: str,
    page_count: int,
    scope: str,
    fraction: float,
) -> list[int]:
    maximum = max(1, min(page_count, math.ceil(page_count * fraction)))
    population = list(range(1, maximum + 1))
    if len(population) < QUESTIONS_PER_BOOK:
        raise RuntimeError(f"{book_name}: assessed range has fewer than 10 pages")
    # Anchor and global are fixed across checkpoints; cumulative depends on fraction.
    key_fraction = (
        f"{fraction:.8f}" if scope == "cumulative" else "FIXED"
    )
    rng = random.Random(
        keyed_seed(
            VERSION,
            master_seed,
            username,
            book_name,
            page_count,
            scope,
            key_fraction,
        )
    )
    return sorted(rng.sample(population, QUESTIONS_PER_BOOK))


def common_uniform(
    master_seed: int,
    username: str,
    checkpoint: str,
    book_name: str,
    scope: str,
    question_number: int,
    page: int,
) -> float:
    # Method is deliberately excluded: Full and Static receive the same draw.
    rng = random.Random(
        keyed_seed(
            "COMMON-RANDOM-NUMBER",
            master_seed,
            username,
            checkpoint,
            book_name,
            scope,
            question_number,
            page,
        )
    )
    return rng.random()


def build_exposure_map(
    studies: Sequence[sqlite3.Row],
    books: Sequence[sqlite3.Row],
    cutoff: date,
) -> dict[int, dict[int, list[Exposure]]]:
    page_counts = {int(b["id"]): int(b["tedad_safahat"]) for b in books}
    output: dict[int, dict[int, list[Exposure]]] = {
        book_id: defaultdict(list) for book_id in page_counts
    }
    invalid = 0
    for row in studies:
        d = parse_date(row["tarikh_en_user"])
        try:
            book_id = int(row["book_id"])
            first = int(row["az_safhe"])
            last = int(row["ta_safhe"])
        except (TypeError, ValueError):
            invalid += 1
            continue
        if (
            d is None
            or d > cutoff
            or book_id not in page_counts
            or first < 1
            or last < first
            or last > page_counts[book_id]
        ):
            invalid += 1
            continue
        exp = Exposure(
            study_date=d,
            quality=study_quality(row),
            is_review=("مرور" in str(row["study_type_name"] or "")),
        )
        for page in range(first, last + 1):
            output[book_id][page].append(exp)
    if invalid:
        raise RuntimeError(f"{invalid} invalid study rows found")
    return output


def recall_features(
    exposures: Sequence[Exposure],
    assessment: date,
) -> dict[str, float | int]:
    if not exposures:
        return {
            "exposure_count": 0,
            "distinct_study_dates": 0,
            "review_exposure_count": 0,
            "days_since_last": -1,
            "source_quality": 0.0,
            "half_life_days": 0.0,
            "recall_probability": 0.0,
        }

    dates = sorted({e.study_date for e in exposures})
    last_date = max(dates)
    days_since_last = (assessment - last_date).days
    if days_since_last < 0:
        raise RuntimeError("Future exposure entered an assessment")

    gaps = [(assessment - e.study_date).days for e in exposures]
    minimum_gap = min(gaps)
    weights = [math.exp(-(gap - minimum_gap) / 5.0) for gap in gaps]
    source_quality = sum(
        e.quality * w for e, w in zip(exposures, weights)
    ) / sum(weights)

    repetition_count = max(0, len(exposures) - 1)
    distinct_revisits = max(0, len(dates) - 1)
    review_exposure_count = sum(1 for e in exposures if e.is_review)

    repetition_signal = clamp(
        math.log1p(repetition_count) / math.log(6.0),
        0.0,
        1.0,
    )
    review_bonus = clamp(
        max(distinct_revisits, review_exposure_count) / 3.0,
        0.0,
        1.0,
    )
    encoding_strength = clamp(
        0.50 * source_quality
        + 0.20
        + 0.20 * repetition_signal
        + 0.10 * review_bonus,
        0.0,
        1.0,
    )

    quality_factor = 0.70 + 0.60 * source_quality
    repetition_factor = 1.00 + 0.30 * math.log1p(repetition_count)
    review_factor = 1.00 + 0.25 * max(distinct_revisits, review_exposure_count)
    stress_factor = 1.00 - 0.25 * NEUTRAL_EXAM_STRESS
    fatigue_factor = 1.00 - 0.15 * NEUTRAL_EXAM_FATIGUE

    half_life = clamp(
        BASE_HALF_LIFE_DAYS
        * quality_factor
        * repetition_factor
        * review_factor
        * stress_factor
        * fatigue_factor,
        HALF_LIFE_MIN,
        HALF_LIFE_MAX,
    )
    retention = 2.0 ** (-days_since_last / half_life)
    recall = clamp(encoding_strength * retention, 0.0, 1.0)

    return {
        "exposure_count": len(exposures),
        "distinct_study_dates": len(dates),
        "review_exposure_count": review_exposure_count,
        "days_since_last": days_since_last,
        "source_quality": source_quality,
        "half_life_days": half_life,
        "recall_probability": recall,
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_method(
    method: str,
    db_path: Path,
    target_users: Sequence[str],
    checkpoints: Sequence[dict[str, Any]],
    master_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    con = open_ro(db_path)
    user_ids = get_users(con, target_users)
    question_rows: list[dict[str, Any]] = []
    manifest: dict[str, dict[str, int]] = {}

    for username in target_users:
        books = get_books(con, user_ids[username])
        manifest[username] = {
            str(b["name"]): int(b["tedad_safahat"]) for b in books
        }

        for checkpoint in checkpoints:
            studies = get_studies(con, user_ids[username], checkpoint["cutoff_date"])
            exposures = build_exposure_map(studies, books, checkpoint["cutoff_date"])

            for book in books:
                book_id = int(book["id"])
                book_name = str(book["name"])
                page_count = int(book["tedad_safahat"])
                scope_fractions = {
                    "anchor": 1.0 / 3.0,
                    "global": 1.0,
                    "cumulative": float(checkpoint["cumulative_fraction"]),
                }
                for scope, fraction in scope_fractions.items():
                    pages = deterministic_pages(
                        master_seed,
                        username,
                        book_name,
                        page_count,
                        scope,
                        fraction,
                    )
                    for q_number, page in enumerate(pages, start=1):
                        features = recall_features(
                            exposures[book_id].get(page, []),
                            checkpoint["assessment_date"],
                        )
                        p_recall = float(features["recall_probability"])
                        p_mc = GUESSING_BASELINE + (1.0 - GUESSING_BASELINE) * p_recall
                        u = common_uniform(
                            master_seed,
                            username,
                            checkpoint["checkpoint"],
                            book_name,
                            scope,
                            q_number,
                            page,
                        )
                        row = {
                            "method": method,
                            "username": username,
                            "checkpoint": checkpoint["checkpoint"],
                            "day_number": checkpoint["day_number"],
                            "cutoff_date": checkpoint["cutoff_date"].isoformat(),
                            "assessment_date": checkpoint["assessment_date"].isoformat(),
                            "scope": scope,
                            "book": book_name,
                            "book_pages": page_count,
                            "question_number": q_number,
                            "page": page,
                            "expected_recall_score": 100.0 * p_recall,
                            "expected_mc_score": 100.0 * p_mc,
                            "common_uniform": u,
                            "realized_mc_correct": int(u < p_mc),
                            **features,
                        }
                        question_rows.append(row)
    con.close()
    return question_rows, manifest


def aggregate(question_rows: Sequence[dict[str, Any]]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    by_book: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in question_rows:
        key = (
            str(row["method"]),
            str(row["username"]),
            str(row["checkpoint"]),
            str(row["scope"]),
            str(row["book"]),
        )
        by_book[key].append(row)

    book_rows: list[dict[str, Any]] = []
    for key, rows in sorted(by_book.items()):
        method, username, checkpoint, scope, book = key
        if len(rows) != QUESTIONS_PER_BOOK:
            raise RuntimeError(f"{key}: expected 10 questions, found {len(rows)}")
        book_rows.append(
            {
                "method": method,
                "username": username,
                "checkpoint": checkpoint,
                "day_number": rows[0]["day_number"],
                "assessment_date": rows[0]["assessment_date"],
                "scope": scope,
                "book": book,
                "questions": len(rows),
                "book_expected_recall_score": mean(
                    [float(r["expected_recall_score"]) for r in rows]
                ),
                "book_expected_mc_score": mean(
                    [float(r["expected_mc_score"]) for r in rows]
                ),
                "book_realized_mc_score": 100.0 * mean(
                    [float(r["realized_mc_correct"]) for r in rows]
                ),
                "book_studied_question_fraction": mean(
                    [float(int(r["exposure_count"]) > 0) for r in rows]
                ),
            }
        )

    by_user: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in book_rows:
        key = (
            str(row["method"]),
            str(row["username"]),
            str(row["checkpoint"]),
            str(row["scope"]),
        )
        by_user[key].append(row)

    user_rows: list[dict[str, Any]] = []
    for key, rows in sorted(by_user.items()):
        method, username, checkpoint, scope = key
        if len(rows) != BOOKS_REQUIRED:
            raise RuntimeError(f"{key}: expected 9 books, found {len(rows)}")
        # Equal book weighting: mean of nine book scores.
        user_rows.append(
            {
                "method": method,
                "username": username,
                "checkpoint": checkpoint,
                "day_number": rows[0]["day_number"],
                "assessment_date": rows[0]["assessment_date"],
                "scope": scope,
                "books": len(rows),
                "questions": sum(int(r["questions"]) for r in rows),
                "equal_book_expected_recall_score": mean(
                    [float(r["book_expected_recall_score"]) for r in rows]
                ),
                "equal_book_expected_mc_score": mean(
                    [float(r["book_expected_mc_score"]) for r in rows]
                ),
                "equal_book_realized_mc_score": mean(
                    [float(r["book_realized_mc_score"]) for r in rows]
                ),
                "equal_book_studied_question_fraction": mean(
                    [float(r["book_studied_question_fraction"]) for r in rows]
                ),
            }
        )
    return book_rows, user_rows


def paired_rows(user_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            str(r["method"]),
            str(r["username"]),
            str(r["checkpoint"]),
            str(r["scope"]),
        ): r
        for r in user_rows
    }
    output: list[dict[str, Any]] = []
    identities = sorted(
        {
            (str(r["username"]), str(r["checkpoint"]), str(r["scope"]))
            for r in user_rows
        }
    )
    for username, checkpoint, scope in identities:
        full = index[("Full", username, checkpoint, scope)]
        static = index[("Static", username, checkpoint, scope)]
        row = {
            "username": username,
            "checkpoint": checkpoint,
            "day_number": full["day_number"],
            "assessment_date": full["assessment_date"],
            "scope": scope,
        }
        for metric in (
            "equal_book_expected_recall_score",
            "equal_book_expected_mc_score",
            "equal_book_realized_mc_score",
            "equal_book_studied_question_fraction",
        ):
            fv = float(full[metric])
            sv = float(static[metric])
            row[f"full_{metric}"] = fv
            row[f"static_{metric}"] = sv
            row[f"difference_{metric}"] = fv - sv
        output.append(row)
    return output


def method_summary(user_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in user_rows:
        groups[
            (str(row["method"]), str(row["checkpoint"]), str(row["scope"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for (method, checkpoint, scope), rows in sorted(groups.items()):
        for metric in (
            "equal_book_expected_recall_score",
            "equal_book_expected_mc_score",
            "equal_book_realized_mc_score",
            "equal_book_studied_question_fraction",
        ):
            values = [float(r[metric]) for r in rows]
            output.append(
                {
                    "method": method,
                    "checkpoint": checkpoint,
                    "scope": scope,
                    "metric": metric,
                    "n_users": len(values),
                    "mean": mean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_baseline_and_question_pairing(
    question_rows: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
) -> None:
    # Same question bank for Full and Static.
    bank: dict[tuple[str, str, str, str, str, int], dict[str, int]] = defaultdict(dict)
    for r in question_rows:
        key = (
            str(r["username"]),
            str(r["checkpoint"]),
            str(r["scope"]),
            str(r["book"]),
            str(r["assessment_date"]),
            int(r["question_number"]),
        )
        bank[key][str(r["method"])] = int(r["page"])
    bad_bank = [
        key for key, methods in bank.items()
        if set(methods) != {"Full", "Static"} or methods["Full"] != methods["Static"]
    ]
    if bad_bank:
        raise RuntimeError(f"Question pairing failure: {bad_bank[:3]}")

    # PRE must be identical for paired databases.
    bad_pre = []
    for r in paired:
        if str(r["checkpoint"]) != "PRE":
            continue
        for metric in (
            "equal_book_expected_recall_score",
            "equal_book_expected_mc_score",
            "equal_book_realized_mc_score",
            "equal_book_studied_question_fraction",
        ):
            if abs(float(r[f"difference_{metric}"])) > 1e-9:
                bad_pre.append((r["username"], r["scope"], metric))
    if bad_pre:
        raise RuntimeError(f"Full/Static PRE mismatch: {bad_pre[:5]}")


def main() -> int:
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    cps = checkpoint_plan(start, args.days, args.protocol)
    target = usernames(args.from_user, args.to_user)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    dbs = {
        "Full": args.full_db.resolve(),
        "Static": args.static_db.resolve(),
    }
    hash_before = {k: sha256_file(v) for k, v in dbs.items()}

    full_q, full_manifest = evaluate_method(
        "Full", dbs["Full"], target, cps, args.master_seed
    )
    static_q, static_manifest = evaluate_method(
        "Static", dbs["Static"], target, cps, args.master_seed
    )
    if full_manifest != static_manifest:
        raise RuntimeError("Full/Static user-book manifests differ")

    question_rows = full_q + static_q
    book_rows, user_rows = aggregate(question_rows)
    paired = paired_rows(user_rows)
    summary = method_summary(user_rows)
    validate_baseline_and_question_pairing(question_rows, paired)

    write_csv(output / "standardized_exam_v1_question_level.csv", question_rows)
    write_csv(output / "standardized_exam_v1_book_level.csv", book_rows)
    write_csv(output / "standardized_exam_v1_user_level.csv", user_rows)
    write_csv(output / "standardized_exam_v1_paired_full_static.csv", paired)
    write_csv(output / "standardized_exam_v1_method_summary.csv", summary)

    hash_after = {k: sha256_file(v) for k, v in dbs.items()}
    if hash_before != hash_after:
        raise RuntimeError("A database hash changed during read-only evaluation")

    protocol = {
        "version": VERSION,
        "status": "FROZEN-CANDIDATE; FREEZE AFTER 5-USER TECHNICAL PASS",
        "protocol": args.protocol,
        "days": args.days,
        "start_date": args.start_date,
        "users": [target[0], target[-1]],
        "master_seed": args.master_seed,
        "books_per_user": BOOKS_REQUIRED,
        "questions_per_book_per_scope": QUESTIONS_PER_BOOK,
        "primary_outcome": "equal_book_expected_recall_score",
        "primary_longitudinal_scope": "anchor",
        "secondary_outcomes": [
            "global equal-book expected recall",
            "cumulative equal-book expected recall",
            "expected multiple-choice score with 25% guessing",
            "realized multiple-choice score with paired keyed random numbers",
        ],
        "satisfaction_queried_or_used": False,
        "existing_exam_rows_queried_or_used": False,
        "unstudied_question_recall": 0.0,
        "primary_random_noise": False,
        "realized_mc_common_random_numbers": True,
        "equal_book_weighting": True,
        "input_database_sha256": hash_before,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoints": [
            {
                **cp,
                "cutoff_date": cp["cutoff_date"].isoformat(),
                "assessment_date": cp["assessment_date"].isoformat(),
            }
            for cp in cps
        ],
    }
    (output / "EVALUATION_PROTOCOL_V1.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "V27.2.1 STANDARDIZED PAIRED EXAM — EVALUATION PROTOCOL V1",
        "=" * 88,
        "Technical status: PASS",
        f"Protocol: {args.protocol}",
        f"Users: {target[0]} to {target[-1]}",
        f"Books per user: {BOOKS_REQUIRED}",
        f"Questions per book per scope: {QUESTIONS_PER_BOOK}",
        "Equal book weighting: YES",
        "Satisfaction queried/used: NO",
        "Existing exam rows queried/used: NO",
        "Unstudied question recall: 0",
        "Primary random noise: NO",
        "Secondary realized MC uses paired keyed random numbers: YES",
        "Full/Static PRE identical: YES",
        "Full/Static question pages identical: YES",
        "Databases unchanged: YES",
        "",
        "CHECKPOINTS",
        "-" * 88,
    ]
    for cp in cps:
        report.append(
            f"{cp['checkpoint']} | day={cp['day_number']} | "
            f"cutoff={cp['cutoff_date']} | assessment={cp['assessment_date']} | "
            f"cumulative_fraction={cp['cumulative_fraction']:.6f}"
        )
    report.extend(
        [
            "",
            "PRIMARY OUTCOME",
            "-" * 88,
            "Equal-book expected-recall score in the fixed first-third anchor scope.",
            "For the final 330-day study, compare PRE-to-FINAL change in Full vs Static.",
        ]
    )
    (output / "EVALUATION_PROTOCOL_V1_REPORT.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    sums = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS.txt").write_text(
        "\n".join(sums) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report))
    print(f"\nQuestion rows: {len(question_rows)}")
    print(f"Book rows: {len(book_rows)}")
    print(f"User rows: {len(user_rows)}")
    print(f"Paired rows: {len(paired)}")
    print(f"Output directory: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EVALUATION PROTOCOL V1 FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
