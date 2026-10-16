"""DMA-DSAC v27.2 runtime patch.

This module is intentionally imported at the *end* of study/shoan.py.  It
replaces the slow/untrained runtime path while preserving the existing schedule
materialisation helpers (book/page selection, meal selection, etc.).

Main changes
------------
* 120-day rolling educational-history window (configurable).
* 60-day activity recency mask (configurable); stale custom activities remain
  mapped to stable slots but are not selectable.
* Vectorised/linear-time reward preparation (removes the quadratic future scan).
* Real offline Discrete-SAC updates from historical transitions before planning.
* Deterministic paired seeds for Full and Static variants.
* Target critic build/copy ordering fixed.
* Dynamic action masking during both policy optimisation and inference.
* Memory features and 1/7/30/120-day review heuristic added to planner state.
* Per-process agent cache plus periodic checkpoints.

The 1/7/30/120 schedule is a configurable modelling assumption, not a universal
law of human memory.  The manuscript must describe it as a simulation policy.
"""
from __future__ import annotations

import csv
import json
import math
import inspect
import os
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from django.contrib.auth.models import User
from django.db.models import Count, Max, Q


@dataclass
class RuntimeConfig:
    mode: str = "full"  # full | static
    history_window_days: int = 120
    activity_lookback_days: int = 60
    review_intervals_days: Tuple[int, int, int, int] = (1, 7, 30, 120)
    max_custom_actions: int = 12
    fixed_action_dim: int = 20
    base_seed: int = 14040504
    batch_size: int = 64
    replay_capacity: int = 20000
    initial_updates: int = 64
    daily_updates: int = 8
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.01
    gradient_clip_norm: float = 5.0
    alpha_min: float = 0.05
    alpha_max: float = 0.25
    alpha_static: float = 0.20
    alpha_decay_days: int = 200
    checkpoint_every_days: int = 7
    stale_refresh_days: int = 7
    checkpoint_root: str = "dma_dsac_checkpoints"
    training_log_path: str = "paper_exports_final/dma_dsac_training_log_v26.csv"
    enable_checkpoints: bool = True


CONFIG = RuntimeConfig()
AGENT_CACHE: Dict[Tuple[str, int], Any] = {}
PATCH_NAMESPACE: Optional[MutableMapping[str, Any]] = None
ORIGINAL_SAC = None


def configure_runtime(**kwargs: Any) -> RuntimeConfig:
    """Update runtime options from the management command."""
    global CONFIG
    for key, value in kwargs.items():
        if value is None:
            continue
        if not hasattr(CONFIG, key):
            raise ValueError(f"Unknown DMA-DSAC runtime option: {key}")
        setattr(CONFIG, key, value)
    if CONFIG.mode not in {"full", "static"}:
        raise ValueError("mode must be 'full' or 'static'")
    if CONFIG.history_window_days < 30:
        raise ValueError("history_window_days must be at least 30")
    if CONFIG.activity_lookback_days < 14:
        raise ValueError("activity_lookback_days must be at least 14")
    if CONFIG.fixed_action_dim < 8:
        raise ValueError("fixed_action_dim must be at least 8")
    return CONFIG


def clear_runtime_agent_cache(save: bool = True) -> None:
    for agent in list(AGENT_CACHE.values()):
        if save:
            try:
                agent.save_checkpoint(force=True)
            except Exception:
                pass
    AGENT_CACHE.clear()


def _ns(name: str) -> Any:
    if PATCH_NAMESPACE is None:
        raise RuntimeError("DMA-DSAC patch has not been applied")
    return PATCH_NAMESPACE[name]


def _target_date() -> date:
    value = _ns("SIMULATION_TARGET_DATE")
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    return value


def _history_bounds(target_date: Optional[date] = None, days: Optional[int] = None) -> Tuple[date, date]:
    target_date = target_date or _target_date()
    days = int(days or CONFIG.history_window_days)
    # Planning for target_date uses realised data through target_date - 1.
    end_inclusive = target_date - timedelta(days=1)
    start_inclusive = end_inclusive - timedelta(days=days - 1)
    return start_inclusive, end_inclusive


def _user_number(user: User) -> int:
    m = re.search(r"(\d+)$", str(user.username))
    return int(m.group(1)) if m else int(user.pk)


def _paired_seed(user: User, target_date: date) -> int:
    # Same formula in Full and Static projects; independent of database PK.
    return int(CONFIG.base_seed) + _user_number(user) * 100_003 + target_date.toordinal()


def _set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@lru_cache(maxsize=512)
def _book_rows_cached(user_id: int) -> Tuple[Tuple[int, str, int], ...]:
    books_model = _ns("books_model")
    return tuple(
        (int(row["id"]), str(row["name"]), max(1, int(row["tedad_safahat"])))
        for row in books_model.objects.filter(user_id=user_id).order_by("id").values("id", "name", "tedad_safahat")
    )


def _book_maps(user: User) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, int]]:
    books = _book_rows_cached(int(user.id))
    name_to_idx = {name: i + 1 for i, (_, name, _) in enumerate(books)}
    name_to_pages = {name: pages for _, name, pages in books}
    idx_to_db = {i + 1: db_id for i, (db_id, _, _) in enumerate(books)}
    return name_to_idx, name_to_pages, idx_to_db


def _duration_fraction(start: Any, end: Any) -> float:
    if start is None or end is None:
        return 0.0
    start_minutes = start.hour * 60 + start.minute
    end_minutes = end.hour * 60 + end.minute
    if end_minutes < start_minutes:
        end_minutes += 24 * 60
    return max(0.0, (end_minutes - start_minutes) / (24.0 * 60.0))


def _start_fraction(value: Any) -> float:
    if value is None:
        return 0.0
    return (value.hour * 60 + value.minute) / (24.0 * 60.0)


def _stable_custom_maps(user: User) -> Tuple[Dict[int, int], List[Any]]:
    """Return DB-id -> RL-id stable mapping and selected custom objects."""
    type_model = _ns("type_Faliat_model")
    custom = list(
        type_model.objects.filter(user=user)
        .exclude(id__lte=6)
        .order_by("id")[: CONFIG.max_custom_actions]
    )
    return {int(obj.id): 9 + i for i, obj in enumerate(custom)}, custom


def get_user_action_space(user: User, target_date: Optional[date] = None, include_mask: bool = False):
    """Return a fixed-width action map with a dynamic feasibility mask.

    The neural output width is kept constant (20 by default) so adding a custom
    activity does not recreate the actor/critics or discard learned weights.
    Slots 1..8 are reserved for fixed activities, study and exam; slots 9..20
    are stable custom-activity slots.  Empty/stale custom slots remain mapped to
    inert sentinels and are masked during both training and inference.
    """
    Faliat_model = _ns("Faliat_model")
    target_date = target_date or _target_date()
    cutoff = target_date - timedelta(days=CONFIG.activity_lookback_days)

    custom_map, custom = _stable_custom_maps(user)
    custom_ids = [int(obj.id) for obj in custom]
    tracked_activity_ids = [5, 6] + custom_ids
    last_seen_by_id: Dict[int, date] = {
        int(row["type_faliat_id"]): row["last_seen"]
        for row in (
            Faliat_model.objects.filter(user=user, type_faliat_id__in=tracked_activity_ids)
            .values("type_faliat_id")
            .annotate(last_seen=Max("tarikh_en_user"))
        )
        if row["last_seen"] is not None
    }
    never_used = [obj.id for obj in custom if int(obj.id) not in last_seen_by_id]
    newest_unused = set(int(x) for x in never_used[-2:])

    fixed_dim = max(8, int(CONFIG.fixed_action_dim))
    max_custom_slots = max(0, fixed_dim - 8)
    rl_id_map: Dict[int, Any] = {i: i for i in range(1, 7)}
    rl_id_map[7] = "__STUDY__"
    rl_id_map[8] = "__EXAM__"
    for slot in range(9, fixed_dim + 1):
        rl_id_map[slot] = f"__INACTIVE_CUSTOM_{slot}__"
    for obj in custom[:max_custom_slots]:
        rl_id_map[custom_map[int(obj.id)]] = int(obj.id)

    rl_ids = list(range(1, fixed_dim + 1))
    idx_to_act = {i: rl_id for i, rl_id in enumerate(rl_ids)}
    act_to_idx = {rl_id: i for i, rl_id in idx_to_act.items()}

    # Essential actions remain available. Optional fixed actions 5 (class) and
    # 6 (other) are active only when they occurred within the same recency window.
    active_rl_ids = {1, 2, 3, 4, 7, 8}
    for optional_fixed in (5, 6):
        last_seen = last_seen_by_id.get(optional_fixed)
        if last_seen is not None and last_seen >= cutoff:
            active_rl_ids.add(optional_fixed)
    for obj in custom[:max_custom_slots]:
        obj_id = int(obj.id)
        last_seen = last_seen_by_id.get(obj_id)
        if (last_seen is not None and last_seen >= cutoff) or obj_id in newest_unused:
            active_rl_ids.add(custom_map[obj_id])
    active_mask = np.asarray([1.0 if idx_to_act[i] in active_rl_ids else 0.0 for i in range(fixed_dim)], dtype=np.float32)

    result = (fixed_dim, idx_to_act, act_to_idx, rl_id_map)
    return result + (active_mask,) if include_mask else result


def create_dfstat(user: User, target_date: Optional[date] = None, history_days: Optional[int] = None) -> pd.DataFrame:
    study_model = _ns("study_model")
    Faliat_model = _ns("Faliat_model")
    noe_motaleh = _ns("noe_motaleh")

    start_date, end_date = _history_bounds(target_date, history_days)
    columns = [
        "date", "actiontype", "bookname", "startpage", "endpage",
        "starttime", "duringtime", "typestudy", "oxijen", "estres",
        "dificulty", "repitnumber", "dutimerev", "typenutrition",
        "typefood", "typeentertinment", "real_date",
    ]

    name_to_idx, name_to_pages, _ = _book_maps(user)
    study_type_names = list(noe_motaleh.objects.order_by("id").values_list("name", flat=True))
    study_type_map = {name: i + 1 for i, name in enumerate(study_type_names)}
    difficulty_map = {"easy": 0.25, "medium": 0.50, "hard": 0.75}

    srows = list(
        study_model.objects.filter(
            user=user,
            tarikh_en_user__gte=start_date,
            tarikh_en_user__lte=end_date,
        ).select_related("book", "noeMotale").values(
            "tarikh_en_user", "book__name", "az_safhe", "ta_safhe",
            "az_saat", "ta_saat", "noeMotale__name", "mizan_rezaiat",
            "mizan_esters", "difficulty",
        )
    )
    study_records: List[Dict[str, Any]] = []
    for row in srows:
        d = row["tarikh_en_user"]
        book_name = row["book__name"]
        pages = max(1, int(name_to_pages.get(book_name, 1)))
        study_records.append({
            "date": (d - start_date).days + 1,
            "actiontype": 7,
            "bookname": int(name_to_idx.get(book_name, 0)),
            "startpage": float(row["az_safhe"] or 0) / pages,
            "endpage": float(row["ta_safhe"] or 0) / pages,
            "starttime": _start_fraction(row["az_saat"]),
            "duringtime": _duration_fraction(row["az_saat"], row["ta_saat"]),
            "typestudy": int(study_type_map.get(row["noeMotale__name"], 0)),
            "oxijen": float(row["mizan_rezaiat"] or 0),
            "estres": float(row["mizan_esters"] or 0),
            "dificulty": float(difficulty_map.get(row["difficulty"], 0.5)),
            "repitnumber": 0.0,
            "dutimerev": 0.0,
            "typenutrition": 0.0,
            "typefood": 0.0,
            "typeentertinment": 0.0,
            "real_date": pd.Timestamp(d),
        })

    custom_map, _ = _stable_custom_maps(user)
    arows = list(
        Faliat_model.objects.filter(
            user=user,
            tarikh_en_user__gte=start_date,
            tarikh_en_user__lte=end_date,
        ).select_related("type_faliat").values(
            "tarikh_en_user", "type_faliat_id", "az_saat", "ta_saat",
            "mizan_rezaiat", "mizan_esters", "tozihat",
        )
    )
    activity_records: List[Dict[str, Any]] = []
    for row in arows:
        db_id = int(row["type_faliat_id"])
        if 1 <= db_id <= 6:
            action_type = db_id
        else:
            action_type = custom_map.get(db_id)
        if action_type is None:
            continue
        d = row["tarikh_en_user"]
        activity_records.append({
            "date": (d - start_date).days + 1,
            "actiontype": int(action_type),
            "bookname": 0.0,
            "startpage": 0.0,
            "endpage": 0.0,
            "starttime": _start_fraction(row["az_saat"]),
            "duringtime": _duration_fraction(row["az_saat"], row["ta_saat"]),
            "typestudy": 0.0,
            "oxijen": float(row["mizan_rezaiat"] or 0),
            "estres": float(row["mizan_esters"] or 0),
            "dificulty": 0.0,
            "repitnumber": 0.0,
            "dutimerev": 0.0,
            "typenutrition": 0.0,
            "typefood": 0.0,
            "typeentertinment": 0.0,
            "real_date": pd.Timestamp(d),
        })

    df = pd.DataFrame(study_records + activity_records, columns=columns)
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df.sort_values(["date", "starttime"]).reset_index(drop=True).fillna(0)


def _calculate_exam_reward(exam_type_value: Any, score: float, previous_score: float, gap: float) -> float:
    weights = {1: 1.0, 2: 0.9, 3: 0.75, 4: 0.5}
    weight = weights.get(int(exam_type_value or 0), 0.5)
    score_100 = float(score) * 100.0
    previous_100 = float(previous_score) * 100.0
    progress = ((score_100 + previous_100) / 2.0) * (score_100 - previous_100)
    main = weight * score_100 * (1.0 + 0.1 * float(gap))
    return float(np.clip((main + progress) / 1500.0, -1.0, 1.0))


def create_dfexam(user: User, max_reward_exam: float = 1500.0, target_date: Optional[date] = None, history_days: Optional[int] = None) -> pd.DataFrame:
    azmoon_model = _ns("azmoon_model")
    type_azmoon_model = _ns("type_azmoon_model")
    start_date, end_date = _history_bounds(target_date, history_days)
    columns = [
        "date", "actiontype", "bookname", "startpage", "endpage",
        "starttime", "duringtime", "typestudy", "oxijen", "estres",
        "dificulty", "repitnumber", "dutimerev", "typenutrition",
        "typefood", "typeentertinment", "exam_type", "previous_score",
        "score", "timebetweenstudy", "R_Exam", "real_date",
    ]
    name_to_idx, name_to_pages, _ = _book_maps(user)
    exam_types = {
        name: i + 1
        for i, name in enumerate(type_azmoon_model.objects.order_by("id").values_list("name", flat=True))
    }
    rows = list(
        azmoon_model.objects.filter(
            user=user,
            tarikh_en_user__gte=start_date,
            tarikh_en_user__lte=end_date,
        ).select_related("book", "type_azmoon").values(
            "tarikh_en_user", "book__name", "az_safhe", "ta_safhe",
            "az_saat", "ta_saat", "type_azmoon__name", "javab_darsad",
        )
    )
    records: List[Dict[str, Any]] = []
    for row in rows:
        d = row["tarikh_en_user"]
        book_name = row["book__name"]
        pages = max(1, int(name_to_pages.get(book_name, 1)))
        records.append({
            "date": (d - start_date).days + 1,
            "actiontype": 8,
            "bookname": int(name_to_idx.get(book_name, 0)),
            "startpage": float(row["az_safhe"] or 0) / pages,
            "endpage": float(row["ta_safhe"] or 0) / pages,
            "starttime": _start_fraction(row["az_saat"]),
            "duringtime": _duration_fraction(row["az_saat"], row["ta_saat"]),
            "typestudy": 0.0,
            "oxijen": 0.0,
            "estres": 0.0,
            "dificulty": 0.0,
            "repitnumber": 0.0,
            "dutimerev": 0.0,
            "typenutrition": 0.0,
            "typefood": 0.0,
            "typeentertinment": 0.0,
            "exam_type": int(exam_types.get(row["type_azmoon__name"], 4)),
            "score": float(row["javab_darsad"] or 0) / 100.0,
            "real_date": pd.Timestamp(d),
        })
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df.sort_values(["bookname", "startpage", "date", "starttime"]).reset_index(drop=True)
    df["previous_score"] = df.groupby(["bookname", "startpage"])["score"].shift(1).fillna(0.0)
    df["timebetweenstudy"] = df.groupby(["bookname", "startpage"])["date"].diff().fillna(0.0)
    df["R_Exam"] = [
        _calculate_exam_reward(t, s, p, g)
        for t, s, p, g in zip(df["exam_type"], df["score"], df["previous_score"], df["timebetweenstudy"])
    ]
    return df.reindex(columns=columns, fill_value=0).fillna(0)


def _days_to_exam_vector(real_dates: pd.Series) -> np.ndarray:
    values = []
    for value in pd.to_datetime(real_dates):
        d = value.date()
        exam = date(d.year, 7, 1)
        if d > exam:
            exam = date(d.year + 1, 7, 1)
        values.append(max(0, (exam - d).days))
    return np.asarray(values, dtype=np.float32)


def _add_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    gap = np.zeros(n, dtype=np.float32)
    stage = np.zeros(n, dtype=np.float32)
    retention = np.ones(n, dtype=np.float32)
    due = np.zeros(n, dtype=np.float32)
    relearn = np.zeros(n, dtype=np.float32)

    learning_mask = df["actiontype"].isin([7, 8]) & (pd.to_numeric(df["bookname"], errors="coerce").fillna(0) > 0)
    if learning_mask.any():
        work = df.loc[learning_mask, ["bookname", "startpage", "endpage", "date"]].copy()
        work["start_key"] = pd.to_numeric(work["startpage"], errors="coerce").fillna(0).round(3)
        work["end_key"] = pd.to_numeric(work["endpage"], errors="coerce").fillna(0).round(3)
        keys = ["bookname", "start_key", "end_key"]
        for _, group in work.groupby(keys, sort=False):
            idxs = group.index.to_list()
            last_day: Optional[float] = None
            distinct_days: List[float] = []
            for idx in idxs:
                current_day = float(df.at[idx, "date"])
                if last_day is None:
                    current_gap = 0.0
                else:
                    current_gap = max(0.0, current_day - last_day)
                if current_day not in distinct_days:
                    distinct_days.append(current_day)
                prior_events = max(1, len(distinct_days) - 1)
                current_stage = min(prior_events, 4)
                interval = float(CONFIG.review_intervals_days[current_stage - 1])
                gap[idx] = current_gap
                stage[idx] = current_stage / 4.0
                retention[idx] = float(2.0 ** (-current_gap / max(interval, 1.0))) if last_day is not None else 1.0
                due[idx] = 1.0 if last_day is not None and current_gap >= interval else 0.0
                relearn[idx] = 1.0 if last_day is not None and current_gap >= CONFIG.review_intervals_days[-1] else 0.0
                last_day = current_day

    df["memory_gap_days"] = gap
    df["review_stage_norm"] = stage
    df["retention_proxy"] = retention
    df["due_review_flag"] = due
    df["relearn_flag"] = relearn
    return df


def _next_exam_signal(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.zeros(len(df), dtype=np.float32)
    gaps = np.zeros(len(df), dtype=np.float32)
    for _, group in df.groupby("bookname", sort=False):
        next_reward = 0.0
        next_day: Optional[float] = None
        for idx in reversed(group.index.to_list()):
            action = int(df.at[idx, "actiontype"])
            day_value = float(df.at[idx, "date"])
            if action == 8:
                next_reward = float(df.at[idx, "R_Exam_Score"])
                next_day = day_value
            elif action == 7 and next_day is not None:
                gap = max(0.0, next_day - day_value)
                if gap <= 30.0:
                    rewards[idx] = next_reward * math.exp(-gap / 30.0)
                    gaps[idx] = gap
    return rewards, gaps


def prepare_rl_data(
    df_stat: pd.DataFrame,
    df_exam: pd.DataFrame,
    user: User,
    min_date: Any,
    rl_id_to_db_id_map: Optional[Mapping[int, Any]] = None,
    max_reward_exam: float = 1500.0,
    max_reward_final_scale: float = 1.0,
) -> pd.DataFrame:
    state_cols = [
        "date", "actiontype", "bookname", "startpage", "endpage",
        "starttime", "duringtime", "typestudy", "oxijen", "estres",
        "dificulty", "repitnumber", "dutimerev", "typenutrition",
        "typefood", "typeentertinment",
    ]
    stat = (df_stat.copy() if df_stat is not None else pd.DataFrame())
    exam = (df_exam.copy() if df_exam is not None else pd.DataFrame())
    for col in state_cols:
        if col not in stat.columns:
            stat[col] = 0.0
        if col not in exam.columns:
            exam[col] = 0.0
    stat["R_Exam_Score"] = 0.0
    exam["R_Exam_Score"] = pd.to_numeric(exam.get("R_Exam", 0.0), errors="coerce").fillna(0.0)
    df = pd.concat([stat, exam], ignore_index=True, sort=False)
    if df.empty:
        return pd.DataFrame(columns=state_cols + ["R_Final"])
    if "real_date" not in df.columns:
        base = pd.Timestamp(min_date)
        df["real_date"] = base + pd.to_timedelta(pd.to_numeric(df["date"], errors="coerce").fillna(0), unit="D")
    df = df.sort_values(["real_date", "starttime", "actiontype"]).reset_index(drop=True).fillna(0)
    df["actiontype"] = pd.to_numeric(df["actiontype"], errors="coerce").fillna(0).astype(int)
    df = _add_memory_features(df)

    prev_stress = pd.to_numeric(df["estres"], errors="coerce").fillna(0).shift(1).fillna(0)
    prev_sat = pd.to_numeric(df["oxijen"], errors="coerce").fillna(0).shift(1).fillna(0)
    df["R_Stress_Change"] = (prev_stress - pd.to_numeric(df["estres"], errors="coerce").fillna(0)) / 60.0
    df["R_Satisfy_Change"] = (pd.to_numeric(df["oxijen"], errors="coerce").fillna(0) - prev_sat) / 60.0

    df["R_Study_Future"], df["future_exam_gap"] = _next_exam_signal(df)
    df["R_Exam_Future"] = np.where(df["actiontype"].eq(8), df["R_Exam_Score"], 0.0)

    days_left = _days_to_exam_vector(df["real_date"])
    urgency = np.clip(1.0 - days_left / 365.0, 0.0, 1.0) ** 1.20
    if CONFIG.mode == "static":
        w_study = np.full(len(df), 3.70, dtype=np.float32)
        w_exam = np.full(len(df), 2.75, dtype=np.float32)
        w_comfort = np.full(len(df), 0.16, dtype=np.float32)
        urgency_for_base = np.full(len(df), 0.50, dtype=np.float32)
    else:
        w_study = 1.40 + 4.60 * urgency
        w_exam = 1.00 + 3.50 * urgency
        w_comfort = 0.25 * (1.00 - 0.70 * urgency)
        urgency_for_base = urgency

    action = df["actiontype"].to_numpy(dtype=np.int32)
    base_reward = np.zeros(len(df), dtype=np.float32)
    base_values = {1: 0.20, 2: 0.30, 3: 0.30, 4: 0.10, 5: 0.05, 6: 0.05}
    for action_id, value in base_values.items():
        base_reward[action == action_id] = value
    base_reward[action == 7] = 0.85 + 0.75 * urgency_for_base[action == 7]
    base_reward[action == 8] = 0.05 * urgency_for_base[action == 8]

    memory_bonus = np.zeros(len(df), dtype=np.float32)
    study_mask = action == 7
    exam_mask = action == 8
    memory_bonus[study_mask] = (
        0.35 * df.loc[study_mask, "due_review_flag"].to_numpy(dtype=np.float32)
        + 0.25 * (1.0 - df.loc[study_mask, "retention_proxy"].to_numpy(dtype=np.float32))
        + 0.35 * df.loc[study_mask, "relearn_flag"].to_numpy(dtype=np.float32)
    )
    memory_bonus[exam_mask] = 0.15 * df.loc[exam_mask, "due_review_flag"].to_numpy(dtype=np.float32)

    future_component = w_exam * df["R_Exam_Future"].to_numpy(dtype=np.float32) + w_study * df["R_Study_Future"].to_numpy(dtype=np.float32)
    comfort_component = w_comfort * (
        df["R_Satisfy_Change"].to_numpy(dtype=np.float32)
        + df["R_Stress_Change"].to_numpy(dtype=np.float32)
    )

    # Linear-time repetition penalty.
    count_today = df.groupby(["date", "actiontype"]).cumcount().to_numpy(dtype=np.float32) + 1.0
    mean_count = (
        df.groupby(["date", "actiontype"]).size().groupby("actiontype").mean().to_dict()
    )
    mean_for_row = np.array([float(mean_count.get(int(a), 1.0)) for a in action], dtype=np.float32)
    excess = np.maximum(0.0, count_today - mean_for_row)
    repetition_penalty = np.where(action == 7, 0.0, -0.15 * excess)
    nonlearning_penalty = np.where((~np.isin(action, [7, 8])) & (df["due_review_flag"].to_numpy() > 0), -0.20, 0.0)

    raw = base_reward + df["R_Exam_Score"].to_numpy(dtype=np.float32) + future_component + comfort_component + memory_bonus + repetition_penalty + nonlearning_penalty
    df["R_Final"] = np.clip(raw / max(float(max_reward_final_scale), 1e-6), -1.0, 1.0)
    df["days_to_exam_norm"] = np.clip(days_left / 365.0, 0.0, 1.0)
    return df.reset_index(drop=True)



def _stale_refresh_date(target: date) -> date:
    refresh = max(1, int(CONFIG.stale_refresh_days))
    return target - timedelta(days=target.toordinal() % refresh)


@lru_cache(maxsize=1024)
def _stale_content_rows_cached(user_id: int, target_date_iso: str) -> Tuple[Tuple[int, int, int, str, int], ...]:
    """Return stale learning segments without loading complete history.

    Study rows nominate candidate segments.  A single grouped exam query then
    treats later assessment/retrieval of the same exact segment as a memory
    event, preventing recently tested material from being falsely relearned.
    """
    study_model = _ns("study_model")
    azmoon_model = _ns("azmoon_model")
    target = date.fromisoformat(target_date_iso)
    cutoff = target - timedelta(days=CONFIG.review_intervals_days[-1])
    candidates = list(
        study_model.objects.filter(user_id=user_id)
        .values("book_id", "az_safhe", "ta_safhe")
        .annotate(last_study=Max("tarikh_en_user"), study_count=Count("id"))
        .filter(last_study__lt=cutoff)
        .order_by("last_study")[:100]
    )
    if not candidates:
        return tuple()

    segment_q = Q()
    for row in candidates:
        segment_q |= Q(
            book_id=row["book_id"],
            az_safhe=row["az_safhe"],
            ta_safhe=row["ta_safhe"],
        )
    exam_rows = (
        azmoon_model.objects.filter(user_id=user_id)
        .filter(segment_q)
        .values("book_id", "az_safhe", "ta_safhe")
        .annotate(last_exam=Max("tarikh_en_user"), exam_count=Count("id"))
    )
    exam_map = {
        (int(r["book_id"]), int(r["az_safhe"]), int(r["ta_safhe"])): r
        for r in exam_rows
    }

    result = []
    for row in candidates:
        key = (int(row["book_id"]), int(row["az_safhe"]), int(row["ta_safhe"]))
        exam = exam_map.get(key)
        last_event = row["last_study"]
        event_count = int(row["study_count"])
        if exam and exam["last_exam"] is not None:
            last_event = max(last_event, exam["last_exam"])
            event_count += int(exam["exam_count"])
        if last_event >= cutoff:
            continue
        result.append((key[0], key[1], key[2], last_event.isoformat(), event_count))
        if len(result) >= 25:
            break
    return tuple(result)


def _planning_memory_summary(user: User, df_final: pd.DataFrame) -> Tuple[float, float, float]:
    """Return mean retention, due-review ratio and relearning ratio for planning."""
    mean_retention = 1.0
    due_ratio = 0.0
    recent_keys = set()
    learning = df_final[df_final["actiontype"].isin([7, 8])].copy() if df_final is not None and not df_final.empty else pd.DataFrame()
    if not learning.empty:
        learning["start_key"] = pd.to_numeric(learning["startpage"], errors="coerce").fillna(0).round(3)
        learning["end_key"] = pd.to_numeric(learning["endpage"], errors="coerce").fillna(0).round(3)
        latest = learning.sort_values(["date", "starttime"]).groupby(["bookname", "start_key", "end_key"], as_index=False).tail(1)
        mean_retention = float(pd.to_numeric(latest["retention_proxy"], errors="coerce").fillna(1.0).mean())
        due_ratio = float(pd.to_numeric(latest["due_review_flag"], errors="coerce").fillna(0.0).mean())
        recent_keys = set(zip(latest["bookname"].astype(int), latest["start_key"], latest["end_key"]))

    try:
        stale_count = len(_stale_content_rows_cached(int(user.id), _stale_refresh_date(_target_date()).isoformat()))
    except Exception:
        stale_count = 0
    total = max(1, len(recent_keys) + stale_count)
    relearn_ratio = float(stale_count / total)
    return float(np.clip(mean_retention, 0, 1)), float(np.clip(due_ratio, 0, 1)), float(np.clip(relearn_ratio, 0, 1))


class ReplayBuffer:
    def __init__(self, capacity: Optional[int] = None, batch_size: Optional[int] = None):
        self.capacity = int(capacity or CONFIG.replay_capacity)
        self.batch_size = int(batch_size or CONFIG.batch_size)
        self.clear()

    def clear(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[float] = []

    def add(self, state: Any, action: int, reward: float, next_state: Any, done: float = 0.0) -> None:
        if len(self.states) >= self.capacity:
            self.states.pop(0); self.actions.pop(0); self.rewards.pop(0); self.next_states.pop(0); self.dones.pop(0)
        self.states.append(np.asarray(state, dtype=np.float32).reshape(-1))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32).reshape(-1))
        self.dones.append(float(done))

    def sample(self):
        if not self.states:
            raise ValueError("Replay buffer is empty")
        k = min(len(self.states), self.batch_size)
        idx = np.random.choice(len(self.states), size=k, replace=False)
        return (
            tf.convert_to_tensor(np.stack([self.states[i] for i in idx]), dtype=tf.float32),
            tf.convert_to_tensor(np.asarray([self.actions[i] for i in idx]), dtype=tf.int32),
            tf.convert_to_tensor(np.asarray([self.rewards[i] for i in idx]), dtype=tf.float32),
            tf.convert_to_tensor(np.stack([self.next_states[i] for i in idx]), dtype=tf.float32),
            tf.convert_to_tensor(np.asarray([self.dones[i] for i in idx]), dtype=tf.float32),
        )

    def __len__(self) -> int:
        return len(self.states)


class Actor(tf.keras.Model):
    def __init__(self, state_dim: int, action_dim: int, name: str = "Actor"):
        super().__init__(name=name)
        self.fc1 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.fc2 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.logits_layer = tf.keras.layers.Dense(int(action_dim), kernel_initializer="glorot_uniform")

    def call(self, state, training: bool = False):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        if state.shape.rank == 1:
            state = state[None, :]
        x = self.fc1(state)
        x = self.fc2(x)
        logits = self.logits_layer(x)
        probs = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        return probs, log_probs, logits


class Critic(tf.keras.Model):
    def __init__(self, state_dim: int, num_columns: int, num_action_types: int, name: str = "Critic"):
        super().__init__(name=name)
        self.q1_1 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.q1_2 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.q1_out = tf.keras.layers.Dense(int(num_action_types))
        self.q2_1 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.q2_2 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.q2_out = tf.keras.layers.Dense(int(num_action_types))

    def call(self, state, training: bool = False):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        if state.shape.rank == 1:
            state = state[None, :]
        q1 = self.q1_out(self.q1_2(self.q1_1(state)))
        q2 = self.q2_out(self.q2_2(self.q2_1(state)))
        return q1, q2


class StudyEnvironment:
    BASE_DIM = 20

    def __init__(
        self,
        df_final_data: pd.DataFrame,
        df_stat_features: pd.DataFrame,
        df_exam_features: pd.DataFrame,
        min_date: Any,
        target_action_dim: Optional[int] = None,
        action_maps: Optional[Mapping[str, Any]] = None,
        active_action_mask: Optional[Sequence[float]] = None,
    ):
        self.user = User.objects.get(id=_ns("CURRENT_USER_ID"))
        self.dfstat = df_stat_features.copy()
        self.dfexam = df_exam_features.copy()
        self.df_final = df_final_data.copy().reset_index(drop=True)
        self.data = self.df_final.to_dict("records")
        if action_maps is None:
            _, idx_to, act_to, _, mask = get_user_action_space(self.user, include_mask=True)
        else:
            idx_to = dict(action_maps["index_to_actiontype"])
            act_to = dict(action_maps["actiontype_to_index"])
            mask = np.asarray(active_action_mask, dtype=np.float32)
        self.index_to_actiontype = idx_to
        self.actiontype_to_index = act_to
        self.n_actions = len(idx_to)
        self.active_action_mask = np.asarray(mask, dtype=np.float32)
        self.state_dim = self.BASE_DIM + 2 * self.n_actions
        self.max_steps = 50
        self.current_time = 0.0
        self.current_step = 0
        self.counts_today = np.zeros(self.n_actions, dtype=np.float32)
        self.base_state = self._row_to_base(self.data[-1]) if self.data else np.zeros(self.BASE_DIM, dtype=np.float32)
        self.memory_summary = _planning_memory_summary(self.user, self.df_final)

        utility = self.df_final.groupby("actiontype")["R_Final"].mean().to_dict() if not self.df_final.empty else {}
        self.action_rl_reward_map = {
            self.actiontype_to_index[int(a)]: float(r)
            for a, r in utility.items() if int(a) in self.actiontype_to_index
        }

    def _row_to_base(self, row: Mapping[str, Any]) -> np.ndarray:
        def f(key: str, default: float = 0.0) -> float:
            try:
                value = row.get(key, default)
                return default if value is None or pd.isna(value) else float(value)
            except Exception:
                return default
        raw_action = int(f("actiontype", 0))
        action_idx = float(self.actiontype_to_index.get(raw_action, 0))
        return np.asarray([
            f("date"), action_idx, f("bookname"), f("startpage"), f("endpage"),
            f("starttime"), f("duringtime"), f("typestudy"), f("oxijen") / 100.0,
            f("estres") / 100.0, f("dificulty"), f("repitnumber"), f("dutimerev"),
            f("typenutrition"), f("typefood"), f("typeentertinment"),
            f("retention_proxy", 1.0), f("due_review_flag"), f("relearn_flag"),
            f("days_to_exam_norm", 1.0),
        ], dtype=np.float32)

    def _compose_state(self, base: np.ndarray, counts: np.ndarray) -> np.ndarray:
        normalized = counts / float(self.max_steps)
        repeated = (counts > 1.0).astype(np.float32)
        return np.concatenate([base, normalized, repeated]).astype(np.float32)

    def reset(self):
        self.current_time = 0.0
        self.current_step = 0
        self.counts_today = np.zeros(self.n_actions, dtype=np.float32)
        self.base_state = self._row_to_base(self.data[-1]) if self.data else np.zeros(self.BASE_DIM, dtype=np.float32)
        self.base_state[5] = 0.0
        self.base_state[16], self.base_state[17], self.base_state[18] = self.memory_summary
        exam = date(_target_date().year, 7, 1)
        if _target_date() > exam:
            exam = date(_target_date().year + 1, 7, 1)
        self.base_state[19] = min(1.0, max(0.0, (exam - _target_date()).days / 365.0))
        return self.get_state()

    def get_state(self):
        return self._compose_state(self.base_state, self.counts_today).reshape(1, -1)

    def step(self, action_index: int):
        action_index = int(action_index)
        if action_index < 0 or action_index >= self.n_actions or self.active_action_mask[action_index] <= 0:
            return self.get_state(), -1.0, True, {"invalid_action": True}
        actiontype = int(self.index_to_actiontype[action_index])
        self.counts_today[action_index] += 1.0
        count = self.counts_today[action_index]
        repetition_penalty = 0.0 if actiontype == 7 else -0.15 * max(0.0, count - 1.0)
        reward = float(np.clip(self.action_rl_reward_map.get(action_index, 0.0) + repetition_penalty, -1.0, 1.0))
        self.current_step += 1
        self.base_state[1] = float(action_index)
        self.base_state[5] = float(np.clip(self.current_time, 0.0, 1.0))
        self.base_state[6] = 1.0 / self.max_steps
        done = self.current_step >= self.max_steps or self.current_time >= 1.0
        return self.get_state(), reward, done, {"action": actiontype, "count": count}

    def build_offline_transitions(self) -> List[Tuple[np.ndarray, int, float, np.ndarray, float]]:
        if self.df_final.empty:
            return []
        transitions = []
        counts = np.zeros(self.n_actions, dtype=np.float32)
        previous_day = None
        rows = self.df_final.to_dict("records")
        for pos, row in enumerate(rows):
            actiontype = int(row.get("actiontype", 0) or 0)
            if actiontype not in self.actiontype_to_index:
                continue
            current_day = int(float(row.get("date", 0) or 0))
            if previous_day is None or current_day != previous_day:
                counts = np.zeros(self.n_actions, dtype=np.float32)
            base = self._row_to_base(row)
            state = self._compose_state(base, counts)
            action_index = int(self.actiontype_to_index[actiontype])
            counts_after = counts.copy(); counts_after[action_index] += 1.0
            if pos + 1 < len(rows):
                next_base = self._row_to_base(rows[pos + 1])
                terminal = float(int(float(rows[pos + 1].get("date", current_day))) != current_day)
            else:
                next_base = base.copy()
                terminal = 1.0
            next_state = self._compose_state(next_base, counts_after)
            reward = float(row.get("R_Final", 0.0) or 0.0)
            transitions.append((state, action_index, reward, next_state, terminal))
            counts = counts_after
            previous_day = current_day
        return transitions


def _alpha_for_date(target_date: date) -> float:
    if CONFIG.mode == "static":
        return float(CONFIG.alpha_static)
    exam = date(target_date.year, 7, 1)
    if target_date > exam:
        exam = date(target_date.year + 1, 7, 1)
    days_left = max(0, (exam - target_date).days)
    if days_left >= CONFIG.alpha_decay_days:
        return float(CONFIG.alpha_max)
    ratio = days_left / float(CONFIG.alpha_decay_days)
    return float(CONFIG.alpha_min + (CONFIG.alpha_max - CONFIG.alpha_min) * ratio)


def _masked_policy(actor: Actor, states: tf.Tensor, active_mask: tf.Tensor, training: bool):
    _, _, logits = actor(states, training=training)
    mask = tf.reshape(tf.cast(active_mask, tf.float32), [1, -1])
    masked_logits = logits + (1.0 - mask) * tf.constant(-1e9, dtype=tf.float32)
    probs = tf.nn.softmax(masked_logits, axis=-1)
    log_probs = tf.nn.log_softmax(masked_logits, axis=-1)
    return probs, log_probs, masked_logits


def _checkpoint_paths(user: User) -> Tuple[Path, Path, Path, Path]:
    root = Path(CONFIG.checkpoint_root) / CONFIG.mode / user.username
    root.mkdir(parents=True, exist_ok=True)
    return root / "actor.weights.h5", root / "critic.weights.h5", root / "target.weights.h5", root / "metadata.json"


def _append_training_log(row: Mapping[str, Any]) -> None:
    path = Path(CONFIG.training_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _build_patched_sac_class(original_sac):
    class PatchedSAC(original_sac):
        def __init__(self, state_dim, action_dim, max_action, num_columns, study_env, action_maps, sac_index_to_db_id, rl_id_to_db_id_map=None, active_action_mask=None):
            # Do not call the broken original __init__; initialise all attributes used
            # by inherited schedule-materialisation methods.
            self.env = study_env
            self.study_env = study_env
            self.dfstat = study_env.dfstat
            self.dfexam = study_env.dfexam
            self.rl_id_to_db_id_map = dict(rl_id_to_db_id_map or {})
            self.db_id_to_rl_id_map = {v: k for k, v in self.rl_id_to_db_id_map.items()}
            self.index_to_actiontype = dict(action_maps["index_to_actiontype"])
            self.actiontype_to_index = dict(action_maps["actiontype_to_index"])
            self.action_dim = int(action_dim)
            self.n_features = int(num_columns)
            self.state_dim = (1, self.n_features)
            self.num_columns = self.n_features
            self.max_action = max_action
            self.sac_index_to_db_id = sac_index_to_db_id
            if active_action_mask is None:
                active_action_mask = np.ones(self.action_dim, dtype=np.float32)
            self.active_action_mask = tf.convert_to_tensor(np.asarray(active_action_mask, dtype=np.float32), dtype=tf.float32)

            self.actor = Actor(self.n_features, self.action_dim)
            self.critic = Critic(self.n_features, self.n_features, self.action_dim)
            self.target_critic = Critic(self.n_features, self.n_features, self.action_dim)
            dummy = tf.zeros((1, self.n_features), dtype=tf.float32)
            self.actor(dummy); self.critic(dummy); self.target_critic(dummy)
            self.target_critic.set_weights(self.critic.get_weights())

            self.actor_optimizer = tf.keras.optimizers.Adam(CONFIG.actor_lr)
            self.critic_optimizer = tf.keras.optimizers.Adam(CONFIG.critic_lr)
            self.gamma = float(CONFIG.gamma)
            self.tau = float(CONFIG.tau)
            self.alpha = float(CONFIG.alpha_static if CONFIG.mode == "static" else CONFIG.alpha_max)
            self.replay_buffer = ReplayBuffer()
            self.loaded_checkpoint = False
            self.last_train_stats: Dict[str, float] = {}

            books_model = _ns("books_model")
            self.book_queryset = books_model.objects.filter(user=study_env.user).order_by("id").values_list("id", "name", "tedad_safahat")
            self.books = {name: db_id for db_id, name, _ in self.book_queryset}
            self.books_name_to_idx = {name: idx + 1 for idx, (_, name, _) in enumerate(self.book_queryset)}
            self.books_idx_to_name = {v: k for k, v in self.books_name_to_idx.items()}
            self.books_db_id_to_name = {v: k for k, v in self.books.items()}
            self.books_idx_to_db_id = {self.books_name_to_idx[name]: db_id for db_id, name, _ in self.book_queryset}
            self.book_info_dict = {idx + 1: pages for idx, (_, _, pages) in enumerate(self.book_queryset)}

            self.load_checkpoint()

        def set_environment(self, env, action_maps, rl_id_map, active_mask):
            self.env = env; self.study_env = env; self.dfstat = env.dfstat; self.dfexam = env.dfexam
            self.index_to_actiontype = dict(action_maps["index_to_actiontype"])
            self.actiontype_to_index = dict(action_maps["actiontype_to_index"])
            self.rl_id_to_db_id_map = dict(rl_id_map)
            self.db_id_to_rl_id_map = {v: k for k, v in self.rl_id_to_db_id_map.items()}
            self.active_action_mask = tf.convert_to_tensor(np.asarray(active_mask, dtype=np.float32), dtype=tf.float32)

        def load_checkpoint(self):
            if not CONFIG.enable_checkpoints:
                return False
            actor_path, critic_path, target_path, meta_path = _checkpoint_paths(self.env.user)
            if not meta_path.exists() or not actor_path.exists() or not critic_path.exists():
                return False
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if int(meta.get("state_dim", -1)) != self.n_features or int(meta.get("action_dim", -1)) != self.action_dim:
                    return False
                self.actor.load_weights(actor_path)
                self.critic.load_weights(critic_path)
                if target_path.exists():
                    self.target_critic.load_weights(target_path)
                else:
                    self.target_critic.set_weights(self.critic.get_weights())
                self.loaded_checkpoint = True
                return True
            except Exception:
                return False

        def save_checkpoint(self, force: bool = False):
            if not CONFIG.enable_checkpoints:
                return
            target = _target_date()
            if not force and CONFIG.checkpoint_every_days > 1 and target.toordinal() % CONFIG.checkpoint_every_days != 0:
                return
            actor_path, critic_path, target_path, meta_path = _checkpoint_paths(self.env.user)
            self.actor.save_weights(actor_path)
            self.critic.save_weights(critic_path)
            self.target_critic.save_weights(target_path)
            meta_path.write_text(json.dumps({
                "mode": CONFIG.mode,
                "state_dim": self.n_features,
                "action_dim": self.action_dim,
                "last_date": target.isoformat(),
                "history_window_days": CONFIG.history_window_days,
            }, indent=2), encoding="utf-8")

        @tf.function(reduce_retracing=True)
        def _update_impl(self, states, actions, rewards, next_states, dones, alpha_value, active_mask):
            alpha_value = tf.cast(alpha_value, tf.float32)
            active_mask = tf.cast(active_mask, tf.float32)
            rewards = tf.reshape(tf.cast(rewards, tf.float32), [-1, 1])
            dones = tf.reshape(tf.cast(dones, tf.float32), [-1, 1])
            actions = tf.reshape(tf.cast(actions, tf.int32), [-1])
            with tf.GradientTape() as tape:
                next_probs, next_log_probs, _ = _masked_policy(self.actor, next_states, active_mask, False)
                tq1, tq2 = self.target_critic(next_states, training=False)
                tq = tf.minimum(tq1, tq2)
                next_v = tf.reduce_sum(next_probs * (tq - alpha_value * next_log_probs), axis=1, keepdims=True)
                target = tf.stop_gradient(rewards + (1.0 - dones) * self.gamma * next_v)
                q1, q2 = self.critic(states, training=True)
                batch = tf.range(tf.shape(actions)[0], dtype=tf.int32)
                indices = tf.stack([batch, actions], axis=1)
                q1_taken = tf.reshape(tf.gather_nd(q1, indices), [-1, 1])
                q2_taken = tf.reshape(tf.gather_nd(q2, indices), [-1, 1])
                critic_loss = tf.reduce_mean(tf.square(q1_taken - target) + tf.square(q2_taken - target))
            grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, CONFIG.gradient_clip_norm)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                probs, log_probs, _ = _masked_policy(self.actor, states, active_mask, True)
                q1, q2 = self.critic(states, training=False)
                q = tf.minimum(q1, q2)
                actor_loss = -tf.reduce_mean(tf.reduce_sum(probs * (q - alpha_value * log_probs), axis=1))
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, CONFIG.gradient_clip_norm)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            for target_var, source_var in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
                target_var.assign((1.0 - self.tau) * target_var + self.tau * source_var)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=1))
            return critic_loss, actor_loss, entropy

        def masked_action_probabilities(self, state):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            if state.shape.rank == 1:
                state = state[None, :]
            elif state.shape.rank and state.shape.rank > 2:
                state = tf.reshape(state, [1, -1])
            probs, _, _ = _masked_policy(self.actor, state, self.active_action_mask, False)
            return probs.numpy().reshape(-1)

        def select_action(self, state):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            if state.shape.rank == 1:
                state = state[None, :]
            elif state.shape.rank and state.shape.rank > 2:
                state = tf.reshape(state, [1, -1])
            _, log_probs, _ = _masked_policy(self.actor, state, self.active_action_mask, False)
            return tf.random.categorical(log_probs, 1)

        def train_offline(self, updates: int):
            transitions = self.env.build_offline_transitions()
            self.replay_buffer.clear()
            for transition in transitions[-CONFIG.replay_capacity:]:
                self.replay_buffer.add(*transition)
            if len(self.replay_buffer) < 2:
                self.last_train_stats = {"critic_loss": np.nan, "actor_loss": np.nan, "entropy": np.nan, "updates": 0}
                return self.last_train_stats
            losses = []
            for _ in range(max(1, int(updates))):
                batch = self.replay_buffer.sample()
                values = self._update_impl(*batch, tf.constant(float(self.alpha), dtype=tf.float32), self.active_action_mask)
                losses.append([float(v.numpy()) for v in values])
            arr = np.asarray(losses, dtype=float)
            self.last_train_stats = {
                "critic_loss": float(np.mean(arr[:, 0])),
                "actor_loss": float(np.mean(arr[:, 1])),
                "entropy": float(np.mean(arr[:, 2])),
                "updates": int(len(losses)),
                "transitions": int(len(self.replay_buffer)),
            }
            return self.last_train_stats

        def suggest_study_type(self, book_name, start_page, end_page):
            """Choose a study mode with correct page normalisation and stale reset."""
            try:
                book_idx = int(book_name)
                total = max(1, int(self.book_info_dict.get(book_idx, 100)))
                start_page = int(start_page); end_page = int(end_page)
                studies = self.dfstat[self.dfstat["actiontype"] == 7].copy()
                if not studies.empty:
                    studies = studies[studies["bookname"].astype(int) == book_idx]
                    studies["start_real"] = studies["startpage"].apply(lambda x: int(round(float(x) * total)) if 0 < float(x) <= 1 else int(round(float(x))))
                    studies["end_real"] = studies["endpage"].apply(lambda x: int(round(float(x) * total)) if 0 < float(x) <= 1 else int(round(float(x))))
                    studies = studies[(studies["end_real"] >= start_page) & (studies["start_real"] <= end_page)]
                if studies.empty:
                    return 2

                # A >120-day gap is treated as relearning, not as a late review.
                last_relative_day = float(studies["date"].max())
                current_relative_day = float(self.dfstat["date"].max())
                if current_relative_day - last_relative_day >= CONFIG.review_intervals_days[-1]:
                    return 2

                exams = self.dfexam.copy()
                if exams is not None and not exams.empty:
                    exams = exams[exams["bookname"].astype(int) == book_idx]
                    exams["start_real"] = exams["startpage"].apply(lambda x: int(round(float(x) * total)) if 0 < float(x) <= 1 else int(round(float(x))))
                    exams["end_real"] = exams["endpage"].apply(lambda x: int(round(float(x) * total)) if 0 < float(x) <= 1 else int(round(float(x))))
                    exams = exams[(exams["end_real"] >= start_page) & (exams["start_real"] <= end_page)]
                    if not exams.empty:
                        score = float(exams.sort_values("date")["score"].iloc[-1])
                        if score >= 0.80:
                            return 1
                        return 2
                return 2
            except Exception:
                return 2

        def get_review_priority(self):
            """Review queue from rolling data plus a bounded stale-content DB query."""
            candidates = []
            seen = set()
            dataset = self.dfstat
            if dataset is not None and not dataset.empty:
                studies = dataset[dataset["actiontype"] == 7]
                if not studies.empty:
                    current_day = float(studies["date"].max())
                    grouped = studies.groupby(["bookname", "startpage", "endpage"])
                    for (book_id, s_p, e_p), group in grouped:
                        distinct_dates = sorted(set(float(x) for x in group["date"].tolist()))
                        stage = min(max(1, len(distinct_dates)), 4)
                        interval = CONFIG.review_intervals_days[stage - 1]
                        gap = current_day - max(distinct_dates)
                        if gap >= interval:
                            key = (int(book_id), round(float(s_p), 3), round(float(e_p), 3))
                            seen.add(key)
                            candidates.append((0 if gap >= 120 else stage, -gap, int(book_id), self._to_real_page(book_id, s_p), self._to_real_page(book_id, e_p)))

            # Stale material older than the rolling window is re-entered as relearning.
            study_model = _ns("study_model")
            cutoff = _target_date() - timedelta(days=CONFIG.review_intervals_days[-1])
            stale = _stale_content_rows_cached(int(self.env.user.id), _stale_refresh_date(_target_date()).isoformat())
            db_to_network = {db_id: idx for idx, db_id in self.books_idx_to_db_id.items()}
            for book_id, az_safhe, ta_safhe, last_seen_iso, event_count in stale:
                network_book = db_to_network.get(int(book_id))
                if network_book is None:
                    continue
                total = max(1, int(self.book_info_dict.get(network_book, 100)))
                s_norm = float(az_safhe) / total
                e_norm = float(ta_safhe) / total
                key = (network_book, round(s_norm, 3), round(e_norm, 3))
                if key in seen:
                    continue
                gap = (_target_date() - date.fromisoformat(last_seen_iso)).days
                candidates.append((0, -float(gap), network_book, int(az_safhe), int(ta_safhe)))
                seen.add(key)
            candidates.sort(key=lambda x: (x[0], x[1]))
            return [(x[2], (x[3], x[4])) for x in candidates]

    return PatchedSAC


def _get_or_create_agent(env, action_maps, rl_id_map, active_mask, sac_index_to_db_id):
    key = (CONFIG.mode, int(env.user.id))
    agent = AGENT_CACHE.get(key)
    if agent is not None and agent.action_dim == env.n_actions and agent.n_features == env.state_dim:
        agent.set_environment(env, action_maps, rl_id_map, active_mask)
        return agent, False
    PatchedSAC = _ns("SAC")
    agent = PatchedSAC(
        state_dim=env.state_dim,
        action_dim=env.n_actions,
        max_action=1,
        num_columns=env.state_dim,
        study_env=env,
        action_maps=action_maps,
        sac_index_to_db_id=sac_index_to_db_id,
        rl_id_to_db_id_map=rl_id_map,
        active_action_mask=active_mask,
    )
    AGENT_CACHE[key] = agent
    return agent, True



def _v272_next_exam_date(day: date) -> date:
    exam = date(day.year, 7, 1)
    if day > exam:
        exam = date(day.year + 1, 7, 1)
    return exam


def _v272_phase_limits(day: date) -> Dict[str, float]:
    remaining = max(0, (_v272_next_exam_date(day) - day).days)
    if remaining <= 30:
        return {
            "block_cap": 3.0,
            "daily_cap": 15.0,
            "minimum": 3.0,
            "max_blocks": 7,
        }
    if remaining <= 90:
        return {
            "block_cap": 2.5,
            "daily_cap": 13.0,
            "minimum": 2.0,
            "max_blocks": 7,
        }
    if remaining <= 180:
        return {
            "block_cap": 2.0,
            "daily_cap": 11.5,
            "minimum": 1.0,
            "max_blocks": 6,
        }
    return {
        "block_cap": 2.0,
        "daily_cap": 10.0,
        "minimum": 1.0,
        "max_blocks": 6,
    }


def _v272_duration_hours(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, timedelta):
        return max(0.0, value.total_seconds() / 3600.0)
    if isinstance(value, time):
        return max(
            0.0,
            float(value.hour)
            + float(value.minute) / 60.0
            + float(value.second) / 3600.0,
        )
    if isinstance(value, (int, float, np.integer, np.floating)):
        return max(0.0, float(value))

    raw = str(value).strip()
    if not raw:
        return 0.0
    if ":" in raw:
        parts = raw.split(":")
        hours = float(parts[0] or 0)
        minutes = float(parts[1] or 0) if len(parts) > 1 else 0.0
        seconds = float(parts[2] or 0) if len(parts) > 2 else 0.0
        return max(0.0, hours + minutes / 60.0 + seconds / 3600.0)
    return max(0.0, float(raw))


def verify_persisted_v272_schedule(user: User, target_date: date) -> Dict[str, float]:
    """
    Validate the materialised DailySchedule inside the caller's transaction.

    Every simulated learner must have at least one study block every day.
    Recovery rules may lower the upper cap, but never remove this daily floor.
    Any failure raises before checkpoint/training-log finalisation so the
    management command's transaction rolls the user's schedule back.
    """
    DailySchedule = _ns("DailySchedule")
    rows = list(
        DailySchedule.objects.filter(
            user=user,
            date=target_date,
        ).values(
            "actiontype",
            "duringtime",
            "bookname",
            "startpage",
            "endpage",
        )
    )

    studies = [
        row
        for row in rows
        if str(row.get("actiontype") or "").strip() == "مطالعه"
    ]
    limits = _v272_phase_limits(target_date)
    hours = [_v272_duration_hours(row.get("duringtime")) for row in studies]
    total_hours = float(sum(hours))
    keys = [
        (
            str(row.get("bookname") or ""),
            int(row.get("startpage") or 0),
            int(row.get("endpage") or 0),
        )
        for row in studies
    ]
    duplicate_count = len(keys) - len(set(keys))
    invalid_range_count = sum(
        (not key[0]) or key[1] < 1 or key[2] < key[1]
        for key in keys
    )
    over_block_count = sum(
        value > float(limits["block_cap"]) + 1.0e-6
        for value in hours
    )

    errors: List[str] = []
    if not rows:
        errors.append("missing_daily_schedule=1")
    if not studies:
        errors.append("missing_daily_study=1")
    if total_hours + 1.0e-6 < float(limits["minimum"]):
        errors.append(
            f"study_floor={total_hours:.3f}<{float(limits['minimum']):.3f}"
        )
    if len(studies) > int(limits["max_blocks"]):
        errors.append(
            f"study_blocks={len(studies)}>{int(limits['max_blocks'])}"
        )
    if total_hours > float(limits["daily_cap"]) + 1.0e-6:
        errors.append(
            f"daily_study_hours={total_hours:.3f}>"
            f"{float(limits['daily_cap']):.3f}"
        )
    if over_block_count:
        errors.append(f"blocks_above_cap={over_block_count}")
    if duplicate_count:
        errors.append(f"duplicate_study_ranges={duplicate_count}")
    if invalid_range_count:
        errors.append(f"invalid_study_ranges={invalid_range_count}")

    if errors:
        raise RuntimeError(
            "v27.2 persisted scheduler verification failed for "
            f"user={user.username}, date={target_date}: "
            + "; ".join(errors)
        )

    return {
        "schedule_rows": float(len(rows)),
        "study_blocks": float(len(studies)),
        "study_hours": total_hours,
        "duplicate_study_ranges": float(duplicate_count),
    }



def run_algorithm_for_user_id(student_id: int, schedule_date: Optional[date] = None, clear_existing_schedule: bool = True):
    global PATCH_NAMESPACE
    if schedule_date is not None:
        _ns("set_simulation_target_date")(schedule_date)
    target_date = schedule_date or _target_date()
    user = User.objects.get(id=student_id)
    PATCH_NAMESPACE["CURRENT_USER_ID"] = student_id
    _set_all_seeds(_paired_seed(user, target_date))

    DailySchedule = _ns("DailySchedule")
    if clear_existing_schedule:
        DailySchedule.objects.filter(user=user, date=target_date).delete()

    df_stat = create_dfstat(user, target_date=target_date)
    df_exam = create_dfexam(user, target_date=target_date)
    if df_stat.empty:
        _ns("logger").warning("No rolling-window study/activity history for %s", user.username)
        return False

    action_dim, idx_to, act_to, rl_map, active_mask = get_user_action_space(user, target_date, include_mask=True)
    window_start, _ = _history_bounds(target_date)
    final_df = prepare_rl_data(df_stat, df_exam, user, window_start, rl_map)
    if final_df.empty:
        return False

    env = StudyEnvironment(
        final_df, df_stat, df_exam, window_start,
        target_action_dim=action_dim,
        action_maps={"index_to_actiontype": idx_to, "actiontype_to_index": act_to},
        active_action_mask=active_mask,
    )
    PATCH_NAMESPACE["NUM_COLUMNS"] = env.state_dim
    PATCH_NAMESPACE["column_names"] = [f"state_{i}" for i in range(env.state_dim)]

    _, _, sac_index_to_db_id = _book_maps(user)
    action_maps = {"index_to_actiontype": idx_to, "actiontype_to_index": act_to}
    agent, is_new = _get_or_create_agent(env, action_maps, rl_map, active_mask, sac_index_to_db_id)
    agent.alpha = _alpha_for_date(target_date)
    training_alpha = float(agent.alpha)
    updates = CONFIG.initial_updates if is_new and not agent.loaded_checkpoint else CONFIG.daily_updates
    actor_before = [w.numpy().copy() for w in agent.actor.trainable_variables]
    critic_before = [w.numpy().copy() for w in agent.critic.trainable_variables]
    stats = agent.train_offline(updates)
    actor_weight_delta = float(math.sqrt(sum(float(np.sum((w.numpy() - b) ** 2)) for w, b in zip(agent.actor.trainable_variables, actor_before))))
    critic_weight_delta = float(math.sqrt(sum(float(np.sum((w.numpy() - b) ** 2)) for w, b in zip(agent.critic.trainable_variables, critic_before))))
    agent.generate_daily_schedule()
    persisted_summary = verify_persisted_v272_schedule(user, target_date)
    _ns("logger").info(
        "v27.2 persisted schedule verified | user=%s | date=%s | "
        "study_blocks=%s | study_hours=%.3f",
        user.username,
        target_date,
        int(persisted_summary["study_blocks"]),
        float(persisted_summary["study_hours"]),
    )
    # The legacy materialisation method contains an old alpha assignment.
    # Restore the value actually used by the SAC updates before logging/saving.
    agent.alpha = training_alpha
    agent.save_checkpoint()

    _append_training_log({
        "mode": CONFIG.mode,
        "username": user.username,
        "paired_seed": _paired_seed(user, target_date),
        "date": target_date.isoformat(),
        "history_rows": len(final_df),
        "transitions": stats.get("transitions", 0),
        "updates": stats.get("updates", 0),
        "critic_loss": stats.get("critic_loss", np.nan),
        "actor_loss": stats.get("actor_loss", np.nan),
        "entropy": stats.get("entropy", np.nan),
        "actor_weight_delta": actor_weight_delta,
        "critic_weight_delta": critic_weight_delta,
        "alpha": training_alpha,
        "action_dim": action_dim,
        "active_actions": int(np.sum(active_mask)),
        "checkpoint_loaded": int(agent.loaded_checkpoint),
        "history_window_days": CONFIG.history_window_days,
        "activity_lookback_days": CONFIG.activity_lookback_days,
    })
    return True



class StudyPriority:
    """Bounded review/relearning queue used by the inherited scheduler."""
    STUDY_ACTIONTYPE = 7

    def __init__(self, dataset_file, book_info_dict):
        self.dataset = dataset_file.copy() if dataset_file is not None else pd.DataFrame()
        self.book_info = {int(k): int(v) for k, v in (book_info_dict or {}).items()}

    def _to_real_page(self, book_id, value):
        try:
            total = max(1, int(self.book_info.get(int(book_id), 100)))
            value = float(value)
            return max(1, int(round(value * total))) if 0 < value <= 1 else max(1, int(round(value)))
        except Exception:
            return 1

    def get_review_priority(self):
        candidates = []
        seen = set()
        if self.dataset is not None and not self.dataset.empty:
            learning_events = self.dataset[self.dataset["actiontype"].isin([7, 8])].copy()
            if not learning_events.empty:
                current_day = float(self.dataset["date"].max())
                for (book_id, s_p, e_p), group in learning_events.groupby(["bookname", "startpage", "endpage"]):
                    distinct_dates = sorted(set(float(x) for x in group["date"].tolist()))
                    stage = min(max(1, len(distinct_dates)), 4)
                    interval = CONFIG.review_intervals_days[stage - 1]
                    gap = current_day - max(distinct_dates)
                    if gap >= interval:
                        key = (int(book_id), round(float(s_p), 3), round(float(e_p), 3))
                        seen.add(key)
                        candidates.append((0 if gap >= 120 else stage, -gap, int(book_id), self._to_real_page(book_id, s_p), self._to_real_page(book_id, e_p)))

        # Re-introduce material whose last study is older than 120 days without
        # loading the complete history into pandas.  The grouped query is bounded.
        try:
            user = User.objects.get(id=_ns("CURRENT_USER_ID"))
            books_model = _ns("books_model")
            study_model = _ns("study_model")
            ordered_books = list(books_model.objects.filter(user=user).order_by("id").values_list("id", flat=True))
            db_to_network = {int(db_id): i + 1 for i, db_id in enumerate(ordered_books)}
            stale = _stale_content_rows_cached(int(user.id), _stale_refresh_date(_target_date()).isoformat())
            for book_id, az_safhe, ta_safhe, last_seen_iso, event_count in stale:
                network_book = db_to_network.get(int(book_id))
                if network_book is None:
                    continue
                total = max(1, int(self.book_info.get(network_book, 100)))
                key = (network_book, round(float(az_safhe) / total, 3), round(float(ta_safhe) / total, 3))
                if key in seen:
                    continue
                gap = (_target_date() - date.fromisoformat(last_seen_iso)).days
                candidates.append((0, -float(gap), network_book, int(az_safhe), int(ta_safhe)))
                seen.add(key)
        except Exception:
            pass
        candidates.sort(key=lambda x: (x[0], x[1]))
        return [(x[2], (x[3], x[4])) for x in candidates]

    def suggest_study(self):
        suggestions = []
        seen_books = set()
        for book_id, pages in self.get_review_priority():
            suggestions.append((book_id, pages))
            seen_books.add(int(book_id))
        if self.dataset is not None and not self.dataset.empty:
            studies = self.dataset[self.dataset["actiontype"] == self.STUDY_ACTIONTYPE]
            if not studies.empty:
                for book_id, end_page in studies.groupby("bookname")["endpage"].max().items():
                    book_id = int(book_id)
                    if book_id in seen_books:
                        continue
                    current = self._to_real_page(book_id, end_page)
                    total = max(1, int(self.book_info.get(book_id, 100)))
                    if current < total:
                        suggestions.append((book_id, (current + 1, min(current + 10, total))))
                        seen_books.add(book_id)
        for book_id in sorted(set(self.book_info) - seen_books):
            suggestions.append((book_id, (1, min(10, self.book_info[book_id]))))
        return suggestions

def apply_patch(namespace: MutableMapping[str, Any], mode: str = "full") -> None:
    """
    Bind the audited RL runtime without replacing the v27.2 scheduler class.

    v27/v27.1 accidentally replaced the enhanced StudyPriority class with the
    older runtime helper. That erased the planning_day API and all same-day
    shadow-state protections. v27.2 explicitly preserves the scheduler class
    defined in shoan_v27_2.py and fails at import time if the API is absent.
    """
    global PATCH_NAMESPACE, ORIGINAL_SAC
    PATCH_NAMESPACE = namespace

    scheduler_cls = namespace.get("StudyPriority")
    if scheduler_cls is None:
        raise RuntimeError("v27.2 StudyPriority is missing before runtime binding")

    scheduler_signature = inspect.signature(scheduler_cls.__init__)
    if "planning_day" not in scheduler_signature.parameters:
        raise RuntimeError(
            "v27.2 runtime binding refused an incompatible StudyPriority "
            "without the planning_day parameter"
        )

    ORIGINAL_SAC = namespace["SAC"]
    configure_runtime(mode=mode)
    patched_sac = _build_patched_sac_class(ORIGINAL_SAC)

    namespace.update({
        "DMA_DSAC_RUNTIME_CONFIG": CONFIG,
        "configure_runtime": configure_runtime,
        "clear_runtime_agent_cache": clear_runtime_agent_cache,
        "create_dfstat": create_dfstat,
        "create_dfexam": create_dfexam,
        "get_user_action_space": get_user_action_space,
        "prepare_rl_data": prepare_rl_data,
        "ReplayBuffer": ReplayBuffer,
        "Actor": Actor,
        "Critic": Critic,
        "StudyEnvironment": StudyEnvironment,
        # Intentionally DO NOT overwrite namespace["StudyPriority"].
        "SAC": patched_sac,
        "verify_persisted_v272_schedule": verify_persisted_v272_schedule,
        "run_algorithm_for_user_id": run_algorithm_for_user_id,
    })

