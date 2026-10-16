#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable


def parse_args():
    p = argparse.ArgumentParser(description='Validate v27.2 scheduler guardrails on a SQLite database.')
    p.add_argument('--db', required=True, type=Path)
    p.add_argument('--start-date', default='2025-07-26')
    p.add_argument('--days', type=int, required=True)
    p.add_argument('--from-user', type=int, default=96)
    p.add_argument('--to-user', type=int, default=100)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--expect-perfect', action='store_true')
    p.add_argument('--require-two-passes', action='store_true')
    return p.parse_args()


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_rows(con, sql: str, params=()):
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def hours_between(start_text, end_text):
    try:
        s = datetime.strptime(str(start_text), '%H:%M:%S')
        e = datetime.strptime(str(end_text), '%H:%M:%S')
        if e < s:
            e += timedelta(days=1)
        return max(0.0, (e - s).total_seconds() / 3600.0)
    except Exception:
        return 0.0


def next_exam_date(day: date) -> date:
    exam = date(day.year, 7, 1)
    if day > exam:
        exam = date(day.year + 1, 7, 1)
    return exam


def phase_limits(day: date):
    remaining = max(0, (next_exam_date(day) - day).days)
    if remaining <= 30:
        return {'block_cap': 3.0, 'hard_daily_cap': 15.0, 'max_zero_gap': 0}
    if remaining <= 90:
        return {'block_cap': 2.5, 'hard_daily_cap': 13.0, 'max_zero_gap': 0}
    if remaining <= 180:
        return {'block_cap': 2.0, 'hard_daily_cap': 11.5, 'max_zero_gap': 0}
    return {'block_cap': 2.0, 'hard_daily_cap': 10.0, 'max_zero_gap': 0}


def consecutive_zero_days(values: list[bool]):
    best = current = 0
    for is_zero in values:
        current = current + 1 if is_zero else 0
        best = max(best, current)
    return best


def main():
    args = parse_args()
    db = args.db.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(args.start_date)
    end = start + timedelta(days=args.days - 1)
    usernames = [f'USim{i}' for i in range(args.from_user, args.to_user + 1)]

    uri = db.as_uri() + '?mode=ro'
    con = sqlite3.connect(uri, uri=True)
    integrity = con.execute('PRAGMA integrity_check').fetchone()[0]
    if str(integrity).lower() != 'ok':
        raise SystemExit(f'SQLite integrity failed: {integrity}')

    marks = ','.join('?' for _ in usernames)
    user_rows = read_rows(con, f'SELECT id, username FROM auth_user WHERE username IN ({marks})', usernames)
    user_map = {r['username']: int(r['id']) for r in user_rows}
    missing = sorted(set(usernames) - set(user_map))
    if missing:
        raise SystemExit(f'Missing users: {missing}')

    summaries = []
    violations = []
    book_rows_out = []

    for username in usernames:
        uid = user_map[username]
        books = read_rows(
            con,
            'SELECT id, name, tedad_safahat FROM study_books_model WHERE user_id=? ORDER BY id',
            (uid,),
        )
        book_by_name = {str(r['name']): r for r in books}

        schedules = read_rows(
            con,
            '''
            SELECT date, actiontype, starttime, duringtime, bookname, startpage, endpage
            FROM study_dailyschedule
            WHERE user_id=? AND date BETWEEN ? AND ?
            ORDER BY date, starttime, id
            ''',
            (uid, start.isoformat(), end.isoformat()),
        )
        planned = [r for r in schedules if str(r['actiontype']).strip() == 'مطالعه']

        actual = read_rows(
            con,
            '''
            SELECT s.tarikh_en_user AS date, s.az_saat, s.ta_saat,
                   s.az_safhe, s.ta_safhe, b.name AS bookname, b.tedad_safahat
            FROM study_study_model s
            JOIN study_books_model b ON b.id=s.book_id
            WHERE s.user_id=? AND s.tarikh_en_user BETWEEN ? AND ?
            ORDER BY s.tarikh_en_user, s.az_saat, s.id
            ''',
            (uid, start.isoformat(), end.isoformat()),
        )

        planned_by_day = defaultdict(list)
        for row in planned:
            planned_by_day[str(row['date'])].append(row)

        actual_by_day = defaultdict(list)
        for row in actual:
            actual_by_day[str(row['date'])].append(row)

        duplicate_count = 0
        invalid_book_count = 0
        over_block_count = 0
        over_daily_count = 0
        high_load_streak_count = 0
        planned_daily_hours = {}
        actual_daily_hours = {}

        previous_two_actual = []
        day_flags = []
        current_zero_gap = 0
        maximum_phase_gap_excess = 0
        day = start
        while day <= end:
            d = day.isoformat()
            limits = phase_limits(day)
            rows = planned_by_day.get(d, [])
            hours = sum(float(r['duringtime'] or 0.0) for r in rows)
            planned_daily_hours[d] = hours
            actual_hours = sum(hours_between(r['az_saat'], r['ta_saat']) for r in actual_by_day.get(d, []))
            actual_daily_hours[d] = actual_hours
            is_zero_study_day = len(rows) == 0
            day_flags.append(is_zero_study_day)
            current_zero_gap = current_zero_gap + 1 if is_zero_study_day else 0
            maximum_phase_gap_excess = max(
                maximum_phase_gap_excess,
                current_zero_gap - limits['max_zero_gap'],
            )

            keys = set()
            for row in rows:
                key = (str(row['bookname']), int(row['startpage'] or 0), int(row['endpage'] or 0))
                if key in keys:
                    duplicate_count += 1
                keys.add(key)
                if str(row['bookname']) not in book_by_name:
                    invalid_book_count += 1
                if float(row['duringtime'] or 0.0) > limits['block_cap'] + 1e-6:
                    over_block_count += 1

            if hours > limits['hard_daily_cap'] + 1e-6:
                over_daily_count += 1

            previous_two_actual.append(actual_hours)
            if len(previous_two_actual) > 3:
                previous_two_actual.pop(0)
            if len(previous_two_actual) == 3 and all(x > 12.0 for x in previous_two_actual):
                high_load_streak_count += 1

            day += timedelta(days=1)

        # Coverage and lower-bound complete passes from actual study records.
        page_dates = {}
        for book in books:
            total = int(book['tedad_safahat'])
            page_dates[str(book['name'])] = [set() for _ in range(total)]

        for row in actual:
            name = str(row['bookname'])
            if name not in page_dates:
                continue
            total = len(page_dates[name])
            left = max(1, int(row['az_safhe'] or 1))
            right = min(total, max(left, int(row['ta_safhe'] or left)))
            for p in range(left - 1, right):
                page_dates[name][p].add(str(row['date']))

        total_pages = 0
        studied_pages = 0
        overall_min_pass = math.inf
        books_started = 0
        books_full = 0

        for book in books:
            name = str(book['name'])
            counts = [len(x) for x in page_dates[name]]
            total = len(counts)
            covered = sum(c > 0 for c in counts)
            min_pass = min(counts) if counts else 0
            total_pages += total
            studied_pages += covered
            overall_min_pass = min(overall_min_pass, min_pass)
            books_started += int(covered > 0)
            books_full += int(covered == total)
            book_rows_out.append({
                'username': username,
                'book': name,
                'pages': total,
                'covered_pages': covered,
                'coverage_percentage': 100.0 * covered / max(1, total),
                'minimum_complete_passes': min_pass,
            })

        if overall_min_pass is math.inf:
            overall_min_pass = 0

        first14_end = min(end, start + timedelta(days=13))
        first14_books = {
            str(r['bookname'])
            for r in planned
            if start.isoformat() <= str(r['date']) <= first14_end.isoformat()
        }
        valid_first14_books = first14_books & set(book_by_name)

        max_zero_gap = consecutive_zero_days(day_flags)

        if duplicate_count:
            violations.append((username, 'same_day_duplicate_range', duplicate_count))
        if invalid_book_count:
            violations.append((username, 'invalid_book_reference', invalid_book_count))
        if over_block_count:
            violations.append((username, 'study_block_above_phase_cap', over_block_count))
        if over_daily_count:
            violations.append((username, 'daily_study_above_phase_hard_cap', over_daily_count))
        if high_load_streak_count:
            violations.append((username, 'three_consecutive_actual_days_above_12h', high_load_streak_count))
        if args.days >= 14 and len(valid_first14_books) < len(books):
            violations.append((username, 'not_all_books_selected_in_first_14_days', len(books) - len(valid_first14_books)))
        if args.expect_perfect and maximum_phase_gap_excess > 0:
            violations.append((
                username,
                'excessive_planned_zero_study_gap',
                maximum_phase_gap_excess,
            ))
        if args.days >= 300 and books_full < len(books):
            violations.append((username, 'incomplete_final_book_coverage', len(books) - books_full))
        if args.days >= 300 and args.require_two_passes and overall_min_pass < 2:
            violations.append((username, 'fewer_than_two_complete_curriculum_passes', overall_min_pass))

        summaries.append({
            'username': username,
            'book_count': len(books),
            'planned_study_events': len(planned),
            'actual_study_events': len(actual),
            'planned_study_hours': sum(planned_daily_hours.values()),
            'actual_study_hours': sum(actual_daily_hours.values()),
            'maximum_planned_daily_hours': max(planned_daily_hours.values(), default=0.0),
            'maximum_actual_daily_hours': max(actual_daily_hours.values(), default=0.0),
            'maximum_zero_study_gap_days': max_zero_gap,
            'same_day_duplicate_ranges': duplicate_count,
            'invalid_book_references': invalid_book_count,
            'blocks_above_phase_cap': over_block_count,
            'days_above_phase_hard_cap': over_daily_count,
            'three_day_high_load_streaks': high_load_streak_count,
            'books_selected_in_first_14_days': len(valid_first14_books),
            'books_fully_covered': books_full,
            'overall_page_coverage_percentage': 100.0 * studied_pages / max(1, total_pages),
            'minimum_complete_curriculum_passes': int(overall_min_pass),
        })

    con.close()

    with (out / 'v27_scheduler_user_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)

    with (out / 'v27_scheduler_book_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(book_rows_out[0]))
        writer.writeheader(); writer.writerows(book_rows_out)

    with (out / 'v27_scheduler_violations.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['username', 'violation', 'value'])
        writer.writerows(violations)

    status = 'PASS' if not violations else 'FAIL'
    report = {
        'status': status,
        'database': str(db),
        'integrity': integrity,
        'start_date': start.isoformat(),
        'end_date': end.isoformat(),
        'users': usernames,
        'violations': len(violations),
    }
    (out / 'V27_VALIDATION_REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'V27_VALIDATION_REPORT.txt').write_text(
        '\n'.join([
            'V27.2 SCHEDULER VALIDATION',
            '=' * 72,
            f"Status: {status}",
            f"Database integrity: {integrity}",
            f"Interval: {start} to {end}",
            f"Users: {usernames[0]} to {usernames[-1]}",
            f"Violations: {len(violations)}",
            '',
            *[f'{u}: {kind} = {value}' for u, kind, value in violations],
        ]),
        encoding='utf-8',
    )

    print(f'V27.2 validation status: {status}')
    print(f'Output directory: {out}')
    if violations:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
