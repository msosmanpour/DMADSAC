"""Memory-Aware Urgency Heuristic (MAU-Heuristic) v1.0.3.

This module adds a deterministic, non-RL external benchmark to the locked
v27.2.1a simulation. It intentionally reuses the audited scheduler
materialisation, feasibility limits, execution model, database schema, rolling
history window, action mask, study-review queue, and read-only assessment.

It does NOT instantiate, train, load, or save Actor/Critic networks.
The heuristic uses only information available at planning time.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from django.contrib.auth.models import User

from study import shoan_v27_2_1a as base
from study.models import DailySchedule, books_model


@dataclass
class HeuristicConfig:
    base_seed: int = 14040504
    history_window_days: int = 120
    activity_lookback_days: int = 60
    review_intervals_days: tuple[int, int, int, int] = (1, 7, 30, 120)
    audit_log_path: str = "paper_exports_final/mau_heuristic_v1_audit.csv"
    protocol_version: str = "MAU_HEURISTIC_V1_LOCKED"


CONFIG = HeuristicConfig()


def configure_runtime(**kwargs: Any) -> HeuristicConfig:
    """Accept the management-command interface without enabling RL training."""
    aliases = {
        "seed": "base_seed",
        "history_window": "history_window_days",
        "activity_lookback": "activity_lookback_days",
    }
    for key, value in kwargs.items():
        if value is None:
            continue
        key = aliases.get(key, key)
        if key in {"mode", "initial_updates", "daily_updates", "checkpoint_root", "training_log_path"}:
            continue
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)

    if CONFIG.history_window_days < 30:
        raise ValueError("history_window_days must be at least 30")
    if CONFIG.activity_lookback_days < 14:
        raise ValueError("activity_lookback_days must be at least 14")
    return CONFIG


def clear_runtime_agent_cache(save: bool = True) -> None:
    """Compatibility no-op: the heuristic has no learned agent or checkpoint."""
    return None


# Expose cache handles expected by the existing command.
get_cached_dfstat = base.get_cached_dfstat
get_cached_dfexam = base.get_cached_dfexam


def _next_exam_date(day: date) -> date:
    exam = date(day.year, 7, 1)
    if day > exam:
        exam = date(day.year + 1, 7, 1)
    return exam


def _urgency(day: date) -> float:
    days_left = max(0, (_next_exam_date(day) - day).days)
    return float(np.clip(1.0 - days_left / 365.0, 0.0, 1.0) ** 1.20)


def _priority_coefficients(day: date) -> Dict[str, float]:
    u = _urgency(day)
    return {
        "urgency": u,
        "study": 1.40 + 4.60 * u,
        "assessment": 1.00 + 3.50 * u,
        "wellbeing": 0.25 * (1.00 - 0.70 * u),
    }


def _stable_epsilon(*parts: Any) -> float:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return value * 1.0e-7


def _append_audit(row: Mapping[str, Any]) -> None:
    path = Path(CONFIG.audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol_version", "username", "master_seed", "date",
        "urgency", "study_priority", "assessment_priority", "wellbeing_priority",
        "schedule_rows", "study_blocks", "study_hours", "exam_blocks",
        "activity_blocks", "actor_used", "critic_used", "checkpoint_used",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


class MAUHeuristicAgent(base.SAC):
    """Lightweight scheduler shell that reuses materialisation but not SAC."""

    def __init__(
        self,
        env,
        action_maps: Mapping[str, Mapping[int, int]],
        sac_index_to_db_id: Mapping[int, int],
        rl_id_to_db_id_map: Optional[Mapping[int, int]],
        active_action_mask: Sequence[float],
        master_seed: int,
        target_date: date,
    ):
        # Deliberately do not call super().__init__(): that would build
        # Actor/Critic networks and replay buffers.
        self.env = env
        self.dfstat = env.dfstat
        self.dfexam = env.dfexam
        self.study_env = env
        self.index_to_actiontype = dict(action_maps["index_to_actiontype"])
        self.actiontype_to_index = dict(action_maps["actiontype_to_index"])
        self.action_dim = len(self.index_to_actiontype)
        self.state_dim = (1, int(env.state_dim))
        self.max_action = 1
        self.num_columns = int(env.state_dim)
        self.sac_index_to_db_id = dict(sac_index_to_db_id)
        self.rl_id_to_db_id_map = dict(rl_id_to_db_id_map or {})
        self.db_id_to_rl_id_map = {v: k for k, v in self.rl_id_to_db_id_map.items()}
        self.active_action_mask = tf.convert_to_tensor(
            np.asarray(active_action_mask, dtype=np.float32), dtype=tf.float32
        )
        self.alpha = 0.0  # No entropy-temperature mechanism in this baseline.
        self.master_seed = int(master_seed)
        self.target_date = target_date

        self.book_queryset = books_model.objects.filter(
            user=env.user
        ).order_by("id").values_list("id", "name", "tedad_safahat")
        rows = list(self.book_queryset)
        self.book_queryset = rows
        self.books = {name: db_id for db_id, name, _ in rows}
        self.books_name_to_idx = {name: i + 1 for i, (_, name, _) in enumerate(rows)}
        self.books_idx_to_name = {i: name for name, i in self.books_name_to_idx.items()}
        self.books_db_id_to_name = {db_id: name for db_id, name, _ in rows}
        self.books_idx_to_db_id = {
            self.books_name_to_idx[name]: db_id for db_id, name, _ in rows
        }
        self.book_info_dict = {
            i + 1: max(1, int(pages)) for i, (_, _, pages) in enumerate(rows)
        }
        self.exam_suggestions = []
        self.exam_index = 0
        self.base_category_rates = self._historical_category_rates()

    def _historical_category_rates(self) -> Dict[str, float]:
        """Mean observed blocks/day in the rolling planning history.

        This calibration prevents the objective coefficients from being
        misinterpreted as literal action proportions. Small fixed floors only
        keep a category mathematically available when its rolling window is
        sparse; they are not tuned from heuristic outcomes.
        """
        df = self.dfstat.copy() if self.dfstat is not None else pd.DataFrame()
        if df.empty or "actiontype" not in df.columns or "date" not in df.columns:
            return {"study": 1.0, "assessment": 0.10, "wellbeing": 3.0}
        action = pd.to_numeric(df["actiontype"], errors="coerce").fillna(0).astype(int)
        day = pd.to_numeric(df["date"], errors="coerce")
        valid_days = max(1, int(day.dropna().nunique()))
        study = float((action == 7).sum()) / valid_days
        assessment = float((action == 8).sum()) / valid_days
        wellbeing = float((~action.isin([7, 8])).sum()) / valid_days
        return {
            "study": max(0.25, study),
            "assessment": max(0.05, assessment),
            "wellbeing": max(0.50, wellbeing),
        }

    def _category_counts(self) -> Dict[str, float]:
        counts = np.asarray(self.env.counts_today, dtype=np.float64)
        study_idx = self.actiontype_to_index.get(7)
        exam_idx = self.actiontype_to_index.get(8)
        study = float(counts[study_idx]) if study_idx is not None else 0.0
        exam = float(counts[exam_idx]) if exam_idx is not None else 0.0
        wellbeing = float(np.sum(counts) - study - exam)
        return {"study": study, "assessment": exam, "wellbeing": wellbeing}

    def _wellbeing_score(self, rl_id: int, count: float, current_hour: float) -> float:
        # Time-of-day suitability is used only inside the non-study category.
        if rl_id == 1:  # sleep
            base_score = 0.10
            if current_hour >= 22.0 or current_hour < 6.0:
                base_score += 4.0
            elif 13.0 <= current_hour <= 15.0:
                base_score += 0.35
        elif rl_id == 2:  # nutrition
            base_score = 0.45
            if (7.0 <= current_hour <= 9.5) or (12.0 <= current_hour <= 14.5) or (18.0 <= current_hour <= 21.0):
                base_score += 2.2
        elif rl_id == 3:  # sport
            base_score = 0.70 + (0.30 if 16.0 <= current_hour <= 20.5 else 0.0)
        elif rl_id == 4:  # entertainment
            base_score = 0.55
        elif rl_id == 5:  # class/extracurricular
            base_score = 0.50
        elif rl_id == 6:  # other
            base_score = 0.35
        else:  # active custom activity
            base_score = 0.40
        return base_score - 0.55 * max(0.0, count)

    @staticmethod
    def _parse_schedule_time(value: Any) -> dt_time:
        """Convert the inherited materialiser's time representation strictly."""
        if isinstance(value, dt_time):
            return value.replace(microsecond=0)
        if isinstance(value, datetime):
            return value.time().replace(microsecond=0)
        if isinstance(value, (int, float, np.integer, np.floating)):
            total_hours = max(0.0, min(23.9833333333, float(value)))
            hour = int(total_hours)
            minute = int(round((total_hours - hour) * 60.0))
            if minute >= 60:
                hour = min(23, hour + 1)
                minute = 0
            return dt_time(hour=hour, minute=minute)
        text = str(value or "").strip()
        if not text:
            raise ValueError("empty starttime")
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"invalid starttime format: {text!r}")
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0

        # The inherited v27.2.1a formatter rounds fractional hours to the
        # nearest minute but does not carry a rounded minute value of 60.
        # Thus a mathematically valid boundary such as 18.999... hours can be
        # serialised as ``18:60`` instead of ``19:00``.  Canonicalise only
        # this one-minute carry edge (and the analogous second carry) before
        # applying strict range checks.  Larger overflows remain errors.
        if second == 60:
            second = 0
            minute += 1
        elif not (0 <= second <= 59):
            raise ValueError(f"out-of-range starttime: {text!r}")

        if minute == 60:
            minute = 0
            hour += 1
        elif not (0 <= minute <= 59):
            raise ValueError(f"out-of-range starttime: {text!r}")

        # Preserve the inherited v27.2.1a persistence semantics at the exact
        # end-of-day rounding boundary. The shared scheduler can serialise a
        # value just below 24.0 hours as ``23:60``; its audited persistence
        # path canonicalises that exact carry to 23:59 rather than wrapping
        # the row to the next date. Apply the same rule only to exact carries.
        # Literal ``24:00`` and larger or irregular overflows remain invalid.
        carried_to_boundary = (
            hour == 24 and minute == 0 and second == 0
            and (
                (len(parts) >= 2 and int(parts[0]) == 23 and int(parts[1]) == 60)
                or (
                    len(parts) == 3
                    and int(parts[0]) == 23
                    and int(parts[1]) == 59
                    and int(parts[2]) == 60
                )
            )
        )
        if carried_to_boundary:
            base.logger.warning(
                "MAU-Heuristic canonicalised exact end-of-day starttime "
                "boundary %r to 23:59:00",
                text,
            )
            return dt_time(hour=23, minute=59, second=0)

        if not (0 <= hour <= 23):
            raise ValueError(f"out-of-range starttime: {text!r}")

        return dt_time(hour=hour, minute=minute, second=second)

    @staticmethod
    def _parse_duration_hours(value: Any) -> float:
        """Convert the inherited whole-minute duration into model hours."""
        if isinstance(value, (int, float, np.integer, np.floating)):
            result = float(value)
        else:
            text = str(value or "").strip()
            if not text:
                raise ValueError("empty duration")
            if ":" in text:
                parts = text.split(":")
                hours = float(parts[0] or 0)
                minutes = float(parts[1] or 0) if len(parts) > 1 else 0.0
                seconds = float(parts[2] or 0) if len(parts) > 2 else 0.0
                result = hours + minutes / 60.0 + seconds / 3600.0
            else:
                result = float(text)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"invalid duration: {value!r}")
        return result

    @staticmethod
    def _parse_schedule_date(value: Any) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value).strip())

    @staticmethod
    def _study_type_name(value: Any) -> str:
        mapping = {
            1: "مطالعه ساده",
            2: "مطالعه با خلاصه نویسی",
            3: "مطالعه با حل مساله",
            4: "حل مساله",
            5: "حل تست",
            6: "مطالعه خلاصه مطالب",
            7: "بررسی آزمون های داده شده",
        }
        try:
            return mapping.get(int(float(value)), "مطالعه ساده")
        except (TypeError, ValueError):
            text = str(value or "").strip()
            return text or "مطالعه ساده"

    def save_schedule_to_django(self, daily_schedule) -> None:
        """Persist the inherited schedule atomically and fail loudly.

        The original shared persistence method catches per-row exceptions and
        continues. That behavior can turn a real persistence failure into a
        later, less-informative ``missing_daily_study`` error.  The external
        heuristic benchmark instead validates every materialised row first,
        then uses one bulk insert.  This changes no planning/ranking rule; it
        is a persistence-only reliability hotfix made before any official
        heuristic outcome was generated.
        """
        user_obj = self.env.user
        nutrition_map = {1: "صبحانه", 2: "نهار", 3: "شام", 4: "میان وعده"}
        food_map = {
            1: "غذاهای پروتئینی",
            2: "غذاهای شیرین",
            3: "لبنیات",
            4: "غذاهای کامل",
            5: "میوه",
            6: "سایر",
        }
        objects = []
        study_count = 0

        for row_number, entry in enumerate(daily_schedule, start=1):
            raw_action = entry.get("actiontype")
            if raw_action == "__STUDY__":
                rl_id = 7
                action_name = "مطالعه"
            elif raw_action == "__EXAM__":
                rl_id = 8
                action_name = "آزمون"
            else:
                try:
                    numeric_action = int(float(raw_action))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"row {row_number}: invalid actiontype={raw_action!r}"
                    ) from exc
                # The inherited materialiser stores custom activities as raw
                # DB ids. get_activity_name supports both RL ids and raw ids.
                rl_id = numeric_action
                action_name = self.get_activity_name(numeric_action, user_obj)

            if not action_name or "نامعتبر" in str(action_name):
                raise ValueError(
                    f"row {row_number}: unresolved action name for {raw_action!r}"
                )

            bookname = entry.get("bookname")
            bookname = str(bookname).strip() if bookname not in (None, "") else None
            startpage = entry.get("startpage")
            endpage = entry.get("endpage")
            startpage = int(float(startpage)) if startpage is not None else None
            endpage = int(float(endpage)) if endpage is not None else None

            if rl_id == 7:
                study_count += 1
                if not bookname or startpage is None or endpage is None:
                    raise ValueError(
                        f"row {row_number}: incomplete study range "
                        f"book={bookname!r}, start={startpage!r}, end={endpage!r}"
                    )
                if startpage < 1 or endpage < startpage:
                    raise ValueError(
                        f"row {row_number}: invalid study range {startpage}-{endpage}"
                    )
                typestudy = self._study_type_name(entry.get("typestudy"))
            elif rl_id == 8:
                typestudy = "آزمون"
            else:
                typestudy = None

            nutrition_value = None
            food_value = None
            if action_name.strip() == "تغذیه":
                try:
                    nutrition_value = nutrition_map.get(
                        int(float(entry.get("typenutrition")))
                    )
                except (TypeError, ValueError):
                    nutrition_value = None
                try:
                    food_value = food_map.get(int(float(entry.get("typefood"))))
                except (TypeError, ValueError):
                    food_value = None

            objects.append(
                DailySchedule(
                    user=user_obj,
                    date=self._parse_schedule_date(entry.get("date")),
                    actiontype=str(action_name).strip(),
                    starttime=self._parse_schedule_time(entry.get("starttime")),
                    duringtime=self._parse_duration_hours(entry.get("duringtime")),
                    bookname=bookname,
                    startpage=startpage,
                    endpage=endpage,
                    typestudy=typestudy,
                    typenutrition=nutrition_value,
                    typefood=food_value,
                )
            )

        if not objects:
            raise RuntimeError("MAU-Heuristic materialised an empty daily schedule")
        if study_count < 1:
            raise RuntimeError("MAU-Heuristic materialised no study row")

        DailySchedule.objects.bulk_create(objects)
        persisted_total = DailySchedule.objects.filter(
            user=user_obj, date=self.target_date
        ).count()
        persisted_study = DailySchedule.objects.filter(
            user=user_obj, date=self.target_date, actiontype="مطالعه"
        ).count()
        if persisted_total != len(objects) or persisted_study != study_count:
            raise RuntimeError(
                "MAU-Heuristic strict persistence mismatch: "
                f"expected_rows={len(objects)}, persisted_rows={persisted_total}, "
                f"expected_study={study_count}, persisted_study={persisted_study}"
            )

    def _score_vector(self) -> np.ndarray:
        coeff = _priority_coefficients(self.target_date)
        adjusted = {
            "study": self.base_category_rates["study"] * coeff["study"],
            "assessment": self.base_category_rates["assessment"] * coeff["assessment"],
            "wellbeing": self.base_category_rates["wellbeing"] * coeff["wellbeing"],
        }
        total = sum(adjusted.values())
        shares = {name: adjusted[name] / total for name in adjusted}
        counts = self._category_counts()
        step = float(np.sum(self.env.counts_today)) + 1.0
        deficits = {
            name: shares[name] * step - counts[name]
            for name in shares
        }

        scores = np.full(self.action_dim, -1.0e9, dtype=np.float64)
        active = np.asarray(self.active_action_mask.numpy(), dtype=np.float64)
        current_hour = float(np.clip(self.env.current_time, 0.0, 1.0) * 24.0)

        exam_available = bool(
            getattr(self, "exam_suggestions", None)
            and int(getattr(self, "exam_index", 0)) < len(self.exam_suggestions)
        )

        for idx in range(self.action_dim):
            if active[idx] <= 0:
                continue
            rl_id = int(self.index_to_actiontype[idx])
            action_count = float(self.env.counts_today[idx])
            tie = _stable_epsilon(
                self.master_seed,
                self.env.user.username,
                self.target_date.isoformat(),
                int(np.sum(self.env.counts_today)),
                rl_id,
            )
            if rl_id == 7:
                score = deficits["study"] + 0.15
            elif rl_id == 8:
                score = deficits["assessment"] + 0.10 if exam_available else -1.0e6
            else:
                score = deficits["wellbeing"] + 0.05 * self._wellbeing_score(
                    rl_id, action_count, current_hour
                )
            scores[idx] = score + tie
        return scores

    def masked_action_probabilities(self, state) -> np.ndarray:
        scores = self._score_vector()
        finite = scores > -1.0e8
        if not np.any(finite):
            raise RuntimeError("MAU-Heuristic found no active action.")
        shifted = scores.copy()
        shifted[finite] -= np.max(shifted[finite])
        probs = np.zeros_like(shifted, dtype=np.float64)
        probs[finite] = np.exp(np.clip(shifted[finite], -50.0, 0.0))
        denom = float(np.sum(probs))
        if denom <= 0.0:
            probs[finite] = 1.0 / float(np.sum(finite))
        else:
            probs /= denom
        return probs.astype(np.float32)

    def select_action(self, state):
        scores = self._score_vector()
        return int(np.argmax(scores))



def _window_start(target_date: date) -> date:
    return target_date - timedelta(days=int(CONFIG.history_window_days))


def run_algorithm_for_user_id(
    student_id: int,
    schedule_date: Optional[date] = None,
    clear_existing_schedule: bool = True,
):
    target_date = schedule_date or date.today()
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    base.set_simulation_target_date(target_date)
    base.CURRENT_USER_ID = int(student_id)
    user = User.objects.get(id=student_id)

    # Deterministic process seeds. No stochastic policy draw is used, but the
    # shared scheduler and helper functions may rely on these libraries.
    paired = int(CONFIG.base_seed) + int(student_id) * 100_003 + target_date.toordinal()
    os.environ["PYTHONHASHSEED"] = str(paired)
    random.seed(paired)
    np.random.seed(paired % (2**32 - 1))
    tf.keras.utils.set_random_seed(paired)

    if clear_existing_schedule:
        DailySchedule.objects.filter(user=user, date=target_date).delete()

    df_stat = base.create_dfstat(user, target_date=target_date)
    df_exam = base.create_dfexam(user, target_date=target_date)
    if df_stat.empty:
        base.logger.warning("No rolling-window history for %s", user.username)
        return False

    action_dim, idx_to, act_to, rl_map, active_mask = base.get_user_action_space(
        user, target_date, include_mask=True
    )
    final_df = base.prepare_rl_data(
        df_stat,
        df_exam,
        user,
        _window_start(target_date),
        rl_id_to_db_id_map=rl_map,
    )
    if final_df.empty:
        return False

    env = base.StudyEnvironment(
        final_df,
        df_stat,
        df_exam,
        _window_start(target_date),
        target_action_dim=action_dim,
        action_maps={"index_to_actiontype": idx_to, "actiontype_to_index": act_to},
        active_action_mask=active_mask,
    )
    base.NUM_COLUMNS = env.state_dim
    base.column_names = [f"state_{i}" for i in range(env.state_dim)]

    book_rows = list(books_model.objects.filter(user=user).order_by("id").values_list("id", flat=True))
    sac_index_to_db_id = {i + 1: int(db_id) for i, db_id in enumerate(book_rows)}

    agent = MAUHeuristicAgent(
        env=env,
        action_maps={"index_to_actiontype": idx_to, "actiontype_to_index": act_to},
        sac_index_to_db_id=sac_index_to_db_id,
        rl_id_to_db_id_map=rl_map,
        active_action_mask=active_mask,
        master_seed=CONFIG.base_seed,
        target_date=target_date,
    )
    agent.generate_daily_schedule()
    persisted = base.verify_persisted_v272_schedule(user, target_date)

    rows = list(DailySchedule.objects.filter(user=user, date=target_date).values("actiontype"))
    exam_blocks = sum(str(r["actiontype"] or "").strip() == "آزمون" for r in rows)
    activity_blocks = len(rows) - int(persisted["study_blocks"]) - exam_blocks
    coeff = _priority_coefficients(target_date)
    _append_audit({
        "protocol_version": CONFIG.protocol_version,
        "username": user.username,
        "master_seed": CONFIG.base_seed,
        "date": target_date.isoformat(),
        "urgency": f"{coeff['urgency']:.10f}",
        "study_priority": f"{coeff['study']:.10f}",
        "assessment_priority": f"{coeff['assessment']:.10f}",
        "wellbeing_priority": f"{coeff['wellbeing']:.10f}",
        "schedule_rows": int(persisted["schedule_rows"]),
        "study_blocks": int(persisted["study_blocks"]),
        "study_hours": f"{float(persisted['study_hours']):.6f}",
        "exam_blocks": int(exam_blocks),
        "activity_blocks": int(activity_blocks),
        "actor_used": 0,
        "critic_used": 0,
        "checkpoint_used": 0,
    })
    base.logger.info(
        "MAU-Heuristic schedule verified | user=%s | date=%s | study_blocks=%s | study_hours=%.3f",
        user.username,
        target_date,
        int(persisted["study_blocks"]),
        float(persisted["study_hours"]),
    )
    return True
