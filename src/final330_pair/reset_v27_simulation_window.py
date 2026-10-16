#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Back up and remove prior simulation-window rows for selected USim users.')
    p.add_argument('--db', required=True, type=Path)
    p.add_argument('--start-date', default='2025-07-26')
    p.add_argument('--from-user', type=int, required=True)
    p.add_argument('--to-user', type=int, required=True)
    p.add_argument('--confirm-reset', required=True, choices=['V27_RESET'])
    p.add_argument('--delete-user-custom-activities', action='store_true')
    return p.parse_args()


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table_exists(con, name):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def main():
    args = parse_args()
    db = args.db.resolve()
    if not db.is_file():
        raise SystemExit(f'Database not found: {db}')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = db.with_name(f'{db.stem}_BEFORE_V27_RESET_{stamp}{db.suffix}')
    shutil.copy2(db, backup)

    con = sqlite3.connect(db)
    con.execute('PRAGMA foreign_keys=ON')
    integrity = con.execute('PRAGMA integrity_check').fetchone()[0]
    if str(integrity).lower() != 'ok':
        con.close()
        raise SystemExit(f'Integrity check failed before reset: {integrity}')

    usernames = [f'USim{i}' for i in range(args.from_user, args.to_user + 1)]
    marks = ','.join('?' for _ in usernames)
    ids = [r[0] for r in con.execute(f'SELECT id FROM auth_user WHERE username IN ({marks})', usernames).fetchall()]
    if len(ids) != len(usernames):
        con.close()
        raise SystemExit(f'Expected {len(usernames)} users, found {len(ids)}. Backup: {backup}')

    id_marks = ','.join('?' for _ in ids)
    deletion_specs = [
        ('study_dailyschedule', 'date'),
        ('study_study_model', 'tarikh_en_user'),
        ('study_azmoon_model', 'tarikh_en_user'),
        ('study_faliat_model', 'tarikh_en_user'),
    ]

    deleted = {}
    try:
        con.execute('BEGIN IMMEDIATE')
        for table, date_col in deletion_specs:
            if not table_exists(con, table):
                continue
            sql = f'DELETE FROM "{table}" WHERE user_id IN ({id_marks}) AND "{date_col}" >= ?'
            cur = con.execute(sql, [*ids, args.start_date])
            deleted[table] = cur.rowcount

        if args.delete_user_custom_activities and table_exists(con, 'study_type_faliat_model'):
            cur = con.execute(
                f'DELETE FROM study_type_faliat_model WHERE user_id IN ({id_marks})',
                ids,
            )
            deleted['study_type_faliat_model'] = cur.rowcount

        con.commit()
    except Exception:
        con.rollback()
        con.close()
        raise

    integrity_after = con.execute('PRAGMA integrity_check').fetchone()[0]
    remaining = {}
    for table, date_col in deletion_specs:
        if table_exists(con, table):
            remaining[table] = con.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE user_id IN ({id_marks}) AND "{date_col}" >= ?',
                [*ids, args.start_date],
            ).fetchone()[0]
    con.close()

    if str(integrity_after).lower() != 'ok' or any(remaining.values()):
        raise SystemExit(
            f'Reset validation failed. Restore backup: {backup}. '
            f'Integrity={integrity_after}, remaining={remaining}'
        )

    print('V27 reset completed successfully.')
    print(f'Database: {db}')
    print(f'Backup:   {backup}')
    print(f'Backup SHA256: {sha256(backup)}')
    print(f'Deleted rows: {deleted}')
    print(f'Remaining simulation-window rows: {remaining}')


if __name__ == '__main__':
    main()
