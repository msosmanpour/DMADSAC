# V27.2.1A_EXACT_PROGRESS_MARKERS = True
# simulator/management/commands/run_shoan_300days.py
import os
try:
    import tensorflow as tf
except Exception:
    tf = None
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import gc
try:
    import tensorflow as tf
except Exception:
    tf = None
import random
import math
from datetime import datetime, date, time, timedelta

import jdatetime
import numpy as np

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import Q

from study.models import (
    DailySchedule,
    study_model,
    Faliat_model,
    azmoon_model,
    books_model,
    noe_motaleh,
    type_Faliat_model,
    type_azmoon_model,
)

# اگر shoan.py داخل اپ study است، همین درست است
from study import shoan_v27_2_1a as shoan

from simulator.state import PsychologicalState
from simulator.rl_env.behavior import HumanBehaviorModel
try:
    from simulator.parameters import SimParams
    SIM_PARAMS = SimParams()
except Exception:
    SIM_PARAMS = None


def P(name: str, default: float) -> float:
    """
    خواندن امن پارامتر از SimParams.
    اگر پارامتر وجود نداشت، مقدار پیش‌فرض برمی‌گردد.
    """

    if SIM_PARAMS is None:
        return float(default)

    return float(getattr(SIM_PARAMS, name, default))


# ---------------------------------------------------------
# تنظیمات اصلی شبیه‌سازی
# ---------------------------------------------------------

START_DATE = date(2025, 7, 26)  # 4 مرداد 1404
DAYS = 330

FIRST_USER_INDEX = 1
LAST_USER_INDEX = 100

MAX_TOTAL_RL_ACTIONS = 20
FIXED_RL_ACTIONS = 8
MAX_CUSTOM_ACTIVITIES_PER_USER = MAX_TOTAL_RL_ACTIONS - FIXED_RL_ACTIONS

DEFAULT_ACTIVITY_NAMES = {
    1: "خواب",
    2: "تغذیه",
    3: "ورزش",
    4: "سرگرمی",
    5: "کلاس آموزشی یا فوق برنامه",
    6: "سایر",
}
NUTRITION_TYPE_MAP = {
    1: "صبحانه",
    2: "نهار",
    3: "شام",
    4: "میان وعده",
}

FOOD_TYPE_MAP = {
    1: "غذاهای پروتئینی",
    2: "غذاهای شیرین",
    3: "لبنیات",
    4: "غذاهای کامل",
    5: "میوه",
    6: "سایر",
}

ENTERTAINMENT_TYPE_MAP = {
    1: "بازی رایانه ای",
    2: "گوشی و شبکه های اجتماعی",
    3: "گردش با خانواده",
    4: "خواندن کتاب های غیر درسی",
    5: "فیلم و سریال",
    6: "سایر",
}


# ---------------------------------------------------------
# ابزارهای تاریخ و زمان
# ---------------------------------------------------------

def gregorian_to_jalali_str(g_date: date) -> str:
    j = jdatetime.date.fromgregorian(date=g_date)
    return f"{j.year:04d}-{j.month:02d}-{j.day:02d}"


def add_hours_to_time(start: time, duration_hours: float) -> time:
    base_dt = datetime.combine(date.today(), start)
    end_dt = base_dt + timedelta(hours=float(duration_hours))

    if end_dt.date() > base_dt.date():
        return time(23, 59)

    return end_dt.time().replace(second=0, microsecond=0)


def clamp_feeling(value: float) -> int:
    """
    تبدیل مقدار 0..1 یا 0..100 به یکی از مقادیر مجاز مدل:
    20, 40, 60, 80, 100
    """

    if value is None:
        return 60

    value = float(value)

    if 0.0 <= value <= 1.0:
        value = value * 100.0

    choices = [20, 40, 60, 80, 100]
    return min(choices, key=lambda x: abs(x - value))

def realistic_satisfaction_score(
    user: User,
    psy,
    rng: random.Random,
    action_kind: str,
    executed: bool = True,
    score: int | None = None,
) -> int:
    """
    تولید رضایت واقعی‌تر برای دانش‌آموز شبیه‌سازی‌شده.

    خروجی فقط یکی از مقادیر مجاز مدل است:
    20, 40, 60, 80, 100

    هدف:
    - رضایت مطالعه همیشه 100 نشود
    - رضایت فعالیت‌ها بیشتر حول 40/60/80 باشد
    - دانش‌آموزان گروه‌های مختلف رفتار متفاوت داشته باشند
    """

    group = get_behavior_group(user)

    base_by_group = {
        "high_adherence": 62,
        "normal": 52,
        "unstable": 45,
        "low_adherence": 38,
    }

    base = base_by_group.get(group, 50)

    # اثر نوع فعالیت
    if action_kind == "study":
        base += 3
        base += 10 * float(getattr(psy, "motivation", 0.5))
        base -= 10 * float(getattr(psy, "stress", 0.5))
        base -= 8 * float(getattr(psy, "fatigue", 0.0))

    elif action_kind == "exam":
        if score is None:
            score = 50

        if score >= 80:
            base += 18
        elif score >= 60:
            base += 8
        elif score >= 40:
            base -= 4
        else:
            base -= 15

        base -= 8 * float(getattr(psy, "stress", 0.5))

    elif action_kind == "sleep":
        base += 12
        base -= 5 * float(getattr(psy, "stress", 0.5))

    elif action_kind == "nutrition":
        base += 6

    elif action_kind == "sport":
        base += 5

    elif action_kind == "entertainment":
        base += 10
        # سرگرمی زیاد همیشه رضایت علمی/روزانه را خیلی بالا نبرد
        base -= 4 * float(getattr(psy, "recent_failure_rate", 0.0))

    else:
        base += 0

    if not executed:
        base -= 18

    # نویز فردی
    base += rng.normalvariate(0, 12)

    # محدودیت برای اینکه زیاد 100 تولید نشود
    base = max(20, min(92, base))

    return clamp_feeling(base)

def clean_schedule_text(value):
    """
    مقدارهای خالی، None و '0' را به None تبدیل می‌کند.
    """

    if value is None:
        return None

    value = str(value).strip()

    if value == "" or value == "0" or value.lower() == "none":
        return None

    return value


def choose_entertainment_type(user: User, rng: random.Random) -> str:
    """
    انتخاب نوع سرگرمی برای ثبت در tozihat براساس گروه تبعیت از برنامه.
    """

    label = get_adherence_profile(user)["label"]

    if label == "perfect_adherence":
        weights = [0.05, 0.08, 0.28, 0.30, 0.14, 0.15]

    elif label == "high_adherence":
        weights = [0.10, 0.15, 0.25, 0.25, 0.15, 0.10]

    elif label == "medium_high_adherence":
        weights = [0.15, 0.25, 0.20, 0.15, 0.15, 0.10]

    elif label == "low_medium_adherence":
        weights = [0.20, 0.35, 0.12, 0.08, 0.18, 0.07]

    else:
        # very_low_adherence
        weights = [0.25, 0.40, 0.08, 0.05, 0.17, 0.05]

    choices = list(ENTERTAINMENT_TYPE_MAP.values())
    return rng.choices(choices, weights=weights, k=1)[0]

def build_activity_description(user: User, schedule: DailySchedule, activity_name: str, rng: random.Random) -> str:
    """
    ساخت مقدار درست برای فیلد tozihat در Faliat_model.

    برای تغذیه:
        نوع تغذیه: شام | نوع غذا: غذاهای پروتئینی

    برای سرگرمی:
        گوشی و شبکه های اجتماعی

    برای بقیه فعالیت‌ها:
        توضیح کوتاه اختصاصی یا خالی
    """

    activity_name = str(activity_name or "").strip()

    if "تغذیه" in activity_name:
        nutrition_text = clean_schedule_text(schedule.typenutrition)
        food_text = clean_schedule_text(schedule.typefood)

        if nutrition_text is None:
            # اگر از DailySchedule چیزی نیامده بود، از ساعت شروع حدس می‌زنیم
            h = schedule.starttime.hour
            if h < 11:
                nutrition_text = "صبحانه"
            elif h < 17:
                nutrition_text = "نهار"
            elif h < 22:
                nutrition_text = "شام"
            else:
                nutrition_text = "میان وعده"

        if food_text is None:
            food_text = rng.choice(list(FOOD_TYPE_MAP.values()))

        return f"نوع تغذیه: {nutrition_text} | نوع غذا: {food_text}"

    if "سرگرمی" in activity_name:
        return choose_entertainment_type(user, rng)

    if "ورزش" in activity_name:
        return rng.choice([
            "پیاده‌روی سبک",
            "حرکات کششی",
            "ورزش کوتاه خانگی",
            "تمرین هوازی سبک",
        ])

    if "خواب" in activity_name:
        return rng.choice([
            "خواب شبانه",
            "استراحت کوتاه",
            "چرت کوتاه",
        ])

    if "کلاس" in activity_name:
        return "کلاس آموزشی یا فوق برنامه"

    return ""
def stress_score_from_state(psy, rng: random.Random, action_kind: str) -> int:
    """
    استرس را کمی طبیعی‌تر و همراه با نویز ذخیره می‌کند.
    """

    base = 100 * float(getattr(psy, "stress", 0.5))

    if action_kind == "exam":
        base += 10
    elif action_kind == "sleep":
        base -= 8
    elif action_kind == "entertainment":
        base -= 5
    elif action_kind == "study":
        base += 3

    base += rng.normalvariate(0, 8)
    base = max(20, min(100, base))

    return clamp_feeling(base)
# ---------------------------------------------------------
# انتخاب کاربران شبیه‌سازی‌شده
# ---------------------------------------------------------

def get_simulated_users():
    usernames = [f"USim{i}" for i in range(FIRST_USER_INDEX, LAST_USER_INDEX + 1)]

    users = list(
        User.objects
        .filter(username__in=usernames)
        .order_by("username")
    )

    # مرتب‌سازی عددی، چون order_by("username") ممکن است USim100 را قبل USim2 بیاورد
    def user_number(u):
        try:
            return int(u.username.replace("USim", ""))
        except Exception:
            return 10**9

    users.sort(key=user_number)
    return users


def get_user_number(user: User) -> int:
    try:
        return int(user.username.replace("USim", ""))
    except Exception:
        return user.id

#===========================برای ذخیره داده ها 
def append_exam_scoring_trace(local_vars):
    """
    ثبت مؤلفه‌های فرمول نمره آزمون برای تحلیل علمی.
    این تابع هیچ اثری روی simulation ندارد؛ فقط CSV تشخیصی می‌سازد.
    """

    try:
        import csv
        from pathlib import Path

        out_path = Path("paper_exports_final/exam_scoring_trace.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "username",
            "date",
            "book",
            "start_page",
            "end_page",
            "duration_hours",
            "real_duration",
            "score",
            "memory_value",
            "normalized_memory",
            "days_since_last",
            "stability",
            "retention_factor",
            "coverage",
            "review_bonus",
            "mastery_signal",
            "effective_mastery",
            "motivation",
            "stress",
            "fatigue",
            "difficulty",
            "study_quality",
            "source_gap_days",
            "source_min_gap_hours",
            "source_review_count",
            "source_coverage",
        ]

        schedule = local_vars.get("schedule")
        user = local_vars.get("user")
        book = local_vars.get("book")
        source_features = local_vars.get("source_features") or {}

        row = {
            "username": getattr(user, "username", ""),
            "date": local_vars.get("target_date", ""),
            "book": getattr(book, "name", "") if book is not None else getattr(schedule, "bookname", ""),
            "start_page": local_vars.get("start_page", ""),
            "end_page": local_vars.get("end_page", ""),
            "duration_hours": local_vars.get("duration_hours", ""),
            "real_duration": local_vars.get("real_duration", ""),
            "score": local_vars.get("score", ""),
            "memory_value": local_vars.get("memory_value", ""),
            "normalized_memory": local_vars.get("normalized_memory", ""),
            "days_since_last": local_vars.get("days_since_last", ""),
            "stability": local_vars.get("stability", ""),
            "retention_factor": local_vars.get("retention_factor", ""),
            "coverage": local_vars.get("coverage", ""),
            "review_bonus": local_vars.get("review_bonus", ""),
            "mastery_signal": local_vars.get("mastery_signal", ""),
            "effective_mastery": local_vars.get("effective_mastery", ""),
            "motivation": local_vars.get("motivation", ""),
            "stress": local_vars.get("stress", ""),
            "fatigue": local_vars.get("fatigue", ""),
            "difficulty": local_vars.get("difficulty", ""),
            "study_quality": local_vars.get("study_quality", ""),
            "source_gap_days": source_features.get("gap_days", ""),
            "source_min_gap_hours": source_features.get("min_gap_hours", ""),
            "source_review_count": source_features.get("review_count", ""),
            "source_coverage": source_features.get("coverage", ""),
        }

        write_header = not out_path.exists()

        with out_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    except Exception as e:
        print("exam scoring trace failed:", repr(e))
# ---------------------------------------------------------
# دسته‌بندی رفتاری دانش‌آموزان
# ---------------------------------------------------------
def get_user_sim_index(user: User) -> int:
    """
    از username مثل USim1 یا usim1 عدد 1 را استخراج می‌کند.
    اگر نشد، از id استفاده می‌کند.
    """

    try:
        return int(
            user.username.lower().replace("usim", "")
        )
    except Exception:
        return int(user.id)


def get_adherence_profile(user: User) -> dict:
    """
    تقسیم‌بندی دقیق 1000 دانش‌آموز به 10 دسته 100 نفری.

    در هر دسته 100 نفری:
    1-25    : 0 تا 20 درصد اجرای برنامه
    26-50   : 20 تا 50 درصد اجرای برنامه
    51-75   : 50 تا 80 درصد اجرای برنامه
    76-95   : 80 تا 98 درصد اجرای برنامه
    96-100  : اجرای دقیق برنامه، یعنی 100 درصد
    """

    n = get_user_sim_index(user)

    # دسته 1 تا 10
    batch_index = ((n - 1) // 100) + 1

    # موقعیت داخل دسته 100 نفری: 1 تا 100
    position_in_batch = ((n - 1) % 100) + 1

    if 1 <= position_in_batch <= 25:
        return {
            "batch": batch_index,
            "position": position_in_batch,
            "label": "very_low_adherence",
            "min_rate": 0.00,
            "max_rate": 0.20,
            "exact": False,
        }

    if 26 <= position_in_batch <= 50:
        return {
            "batch": batch_index,
            "position": position_in_batch,
            "label": "low_medium_adherence",
            "min_rate": 0.20,
            "max_rate": 0.50,
            "exact": False,
        }

    if 51 <= position_in_batch <= 75:
        return {
            "batch": batch_index,
            "position": position_in_batch,
            "label": "medium_high_adherence",
            "min_rate": 0.50,
            "max_rate": 0.80,
            "exact": False,
        }

    if 76 <= position_in_batch <= 95:
        return {
            "batch": batch_index,
            "position": position_in_batch,
            "label": "high_adherence",
            "min_rate": 0.80,
            "max_rate": 0.98,
            "exact": False,
        }

    # نفرهای 96 تا 100 هر دسته
    return {
        "batch": batch_index,
        "position": position_in_batch,
        "label": "perfect_adherence",
        "min_rate": 1.00,
        "max_rate": 1.00,
        "exact": True,
    }
def get_behavior_group(user: User) -> str:
    """
    سازگاری با کدهای قدیمی.
    از این به بعد مبنای اصلی دسته‌بندی get_adherence_profile است.
    """

    return get_adherence_profile(user)["label"]

def target_adherence_for_user_day(user: User, day_index: int) -> float:
    """
    نرخ هدف اجرای برنامه برای یک دانش‌آموز در یک روز خاص.

    برای نفرهای 96 تا 100 هر دسته، همیشه 1.0 است.
    برای بقیه، مقدار داخل بازه گروه خودش است، با تغییرات کوچک روزانه.
    """

    profile = get_adherence_profile(user)

    if profile["exact"]:
        return 1.0

    min_rate = profile["min_rate"]
    max_rate = profile["max_rate"]

    # مقدار پایه پایدار برای هر کاربر
    stable_rng = random.Random(500000 + get_user_sim_index(user) * 13)
    base_rate = stable_rng.uniform(min_rate, max_rate)

    # نویز روزانه کوچک
    day_rng = random.Random(700000 + get_user_sim_index(user) * 17 + day_index * 31)
    day_noise = day_rng.normalvariate(0, 0.04)

    rate = base_rate + day_noise

    # حتماً داخل بازه گروه خودش نگه می‌داریم
    rate = max(min_rate, min(max_rate, rate))

    return float(rate)

def build_execution_plan_for_day(user: User, schedules: list, day_index: int, rng: random.Random):
    """
    برای برنامه‌های یک روز، مشخص می‌کند کدام آیتم‌ها طبق برنامه اجرا شوند.

    خروجی:
        dict {schedule_id: True/False}

    True یعنی دانش‌آموز دقیقاً طبق برنامه پیشنهادی عمل می‌کند.
    False یعنی دانش‌آموز از برنامه پیشنهادی پیروی نمی‌کند.
    """

    if not schedules:
        return {}

    target_rate = target_adherence_for_user_day(user, day_index)
    profile = get_adherence_profile(user)

    # نفرهای 96 تا 100 هر دسته: اجرای دقیق کل برنامه
    if profile["exact"]:
        return {s.id: True for s in schedules}

    total_items = len(schedules)

    # تعداد آیتم‌هایی که باید واقعاً طبق برنامه اجرا شود
    n_follow = int(round(total_items * target_rate))
    n_follow = max(0, min(total_items, n_follow))

    indices = list(range(total_items))
    rng.shuffle(indices)

    follow_indices = set(indices[:n_follow])

    execution_map = {}

    for i, schedule in enumerate(schedules):
        execution_map[schedule.id] = i in follow_indices

    return execution_map

def build_psychological_state(user: User, seed: int) -> PsychologicalState:
    """
    ساخت حالت روانشناسی اولیه براساس دسته اجرای برنامه.
    مقدارها نباید از ابتدا خیلی افراطی باشند، چون در 300 روز سریع اشباع می‌شوند.
    """

    profile = get_adherence_profile(user)
    label = profile["label"]

    if label == "very_low_adherence":
        return PsychologicalState(
            stress_base=0.62,
            hope_base=0.42,
            motivation_base=0.42,
            stress_sensitivity=1.20,
            noise=0.08,
            seed=seed,
        )

    if label == "low_medium_adherence":
        return PsychologicalState(
            stress_base=0.55,
            hope_base=0.50,
            motivation_base=0.52,
            stress_sensitivity=1.08,
            noise=0.07,
            seed=seed,
        )

    if label == "medium_high_adherence":
        return PsychologicalState(
            stress_base=0.48,
            hope_base=0.60,
            motivation_base=0.63,
            stress_sensitivity=0.98,
            noise=0.06,
            seed=seed,
        )

    if label == "high_adherence":
        return PsychologicalState(
            stress_base=0.42,
            hope_base=0.70,
            motivation_base=0.72,
            stress_sensitivity=0.88,
            noise=0.05,
            seed=seed,
        )

    return PsychologicalState(
        stress_base=0.38,
        hope_base=0.76,
        motivation_base=0.78,
        stress_sensitivity=0.80,
        noise=0.04,
        seed=seed,
    )

#========================
def warm_start_psychological_state_from_history(user, psy, start_date):
    """
    حافظه و وضعیت روان‌شناختی شبیه‌ساز را از رکوردهای قبل از start_date مقداردهی اولیه می‌کند.

    چرا لازم است؟
    چون الگوریتم شوان هنگام برنامه‌ریزی، رکوردهای قبلی کاربر را در دیتابیس می‌بیند؛
    بنابراین شبیه‌ساز هم باید همان سابقه را تا حدی در memory_strength و وضعیت روانی خود داشته باشد.
    """

    history_studies = (
        study_model.objects
        .filter(user=user, tarikh_en_user__lt=start_date)
        .order_by("tarikh_en_user", "az_saat", "id")
    )

    history_exams = (
        azmoon_model.objects
        .filter(user=user, tarikh_en_user__lt=start_date)
        .order_by("tarikh_en_user", "az_saat", "id")
    )

    study_count = 0
    stress_values = []
    satisfaction_values = []

    for st in history_studies:
        if not st.book_id:
            continue

        book_id = st.book_id

        try:
            pages = max(1, int((st.ta_safhe or st.az_safhe or 1) - (st.az_safhe or 1) + 1))
        except Exception:
            pages = 1

        try:
            stress_raw = float(st.mizan_esters or 60) / 100.0
        except Exception:
            stress_raw = 0.60

        try:
            satisfaction_raw = float(st.mizan_rezaiat or 60) / 100.0
        except Exception:
            satisfaction_raw = 0.60

        stress_values.append(stress_raw)
        satisfaction_values.append(satisfaction_raw)

        # کیفیت تقریبی مطالعه‌های قبلی.
        # اگر رضایت بالاتر و استرس کمتر باشد، اثر حافظه بیشتر می‌شود.
        quality = float(np.clip(
            0.45 + 0.35 * satisfaction_raw + 0.20 * (1.0 - stress_raw),
            0.20,
            0.95,
        ))

        # مطالعه طولانی‌تر کمی اثر بیشتر دارد، ولی محدود نگه داشته می‌شود.
        page_factor = float(np.clip(pages / 10.0, 0.30, 1.20))

        memory_gain = 0.35 * quality * page_factor

        current_memory = float(psy.memory_strength[book_id])
        psy.memory_strength[book_id] = float(np.clip(current_memory + memory_gain, 0.20, 5.00))

        psy.repetitions[book_id] += 1

        try:
            days_before_start = (start_date - st.tarikh_en_user).days
            psy.last_study_day[book_id] = -float(max(0, days_before_start))
        except Exception:
            pass

        study_count += 1

    exam_scores = []
    for ex in history_exams:
        try:
            if ex.javab_darsad is not None:
                exam_scores.append(float(ex.javab_darsad) / 100.0)
        except Exception:
            pass

    # وضعیت روانی اولیه را هم کمی با سابقه واقعی هماهنگ می‌کنیم.
    if stress_values:
        avg_stress = sum(stress_values) / len(stress_values)
        psy.stress = float(np.clip(0.70 * float(psy.stress) + 0.30 * avg_stress, 0.05, 0.95))

    if satisfaction_values:
        avg_satisfaction = sum(satisfaction_values) / len(satisfaction_values)
        psy.satisfaction = float(np.clip(0.70 * float(psy.satisfaction) + 0.30 * avg_satisfaction, 0.05, 0.95))
    #=========== حذف شده
    # فعلاً از آزمون‌های قدیمی برای تغییر motivation / failure استفاده نمی‌کنیم.
    # دلیل: آزمون‌های قدیمی ممکن است با نسخه‌های قبلی و خطادار شبیه‌ساز ساخته شده باشند.
    # history در این مرحله فقط برای warm-start حافظه استفاده می‌شود.
    '''
    if exam_scores:
        avg_exam = sum(exam_scores) / len(exam_scores)
        psy.motivation = float(np.clip(0.75 * float(psy.motivation) + 0.25 * avg_exam, 0.05, 0.95))

        # اگر کاربر سابقه آزمون خیلی بد داشته باشد، کمی recent_failure_rate را بالا می‌بریم.
        low_exam_rate = sum(1 for x in exam_scores if x < 0.40) / max(1, len(exam_scores))
        psy.recent_failure_rate = float(np.clip(low_exam_rate, 0.0, 1.0))
    '''
    return {
        "history_study_count": study_count,
        "history_exam_count": len(exam_scores),
    }
def build_behavior_model(user: User, seed: int) -> HumanBehaviorModel:
    """
    ساخت مدل رفتار انسانی براساس دسته اجرای برنامه.

    نکته:
    نرخ اجرای نهایی برنامه با build_execution_plan_for_day کنترل می‌شود،
    اما این مدل هنوز روی شدت اجرا، کیفیت، خستگی، انگیزه و پیامدهای روانشناختی اثر دارد.
    """

    profile = get_adherence_profile(user)
    label = profile["label"]

    # مقدار میانگین بازه برای تنظیم شخصیت پایه
    target_mid = (profile["min_rate"] + profile["max_rate"]) / 2.0

    if label == "perfect_adherence":
        target_mid = 0.98

    if label == "very_low_adherence":
        return HumanBehaviorModel(
            adherence_base=-1.20,
            stress_impact=1.10,
            motivation_impact=0.40,
            randomness=0.12,
            self_regulation=0.25,
            habit_strength=0.20,
            fatigue_sensitivity=1.25,
            reward_responsiveness=0.75,
            seed=seed,
        )

    if label == "low_medium_adherence":
        return HumanBehaviorModel(
            adherence_base=-0.45,
            stress_impact=0.90,
            motivation_impact=0.55,
            randomness=0.09,
            self_regulation=0.45,
            habit_strength=0.38,
            fatigue_sensitivity=1.05,
            reward_responsiveness=0.85,
            seed=seed,
        )

    if label == "medium_high_adherence":
        return HumanBehaviorModel(
            adherence_base=0.15,
            stress_impact=0.70,
            motivation_impact=0.70,
            randomness=0.06,
            self_regulation=0.65,
            habit_strength=0.58,
            fatigue_sensitivity=0.85,
            reward_responsiveness=1.00,
            seed=seed,
        )

    if label == "high_adherence":
        return HumanBehaviorModel(
            adherence_base=0.75,
            stress_impact=0.50,
            motivation_impact=0.85,
            randomness=0.04,
            self_regulation=0.82,
            habit_strength=0.78,
            fatigue_sensitivity=0.65,
            reward_responsiveness=1.10,
            seed=seed,
        )

    # perfect_adherence
    return HumanBehaviorModel(
        adherence_base=1.25,
        stress_impact=0.35,
        motivation_impact=0.95,
        randomness=0.01,
        self_regulation=0.95,
        habit_strength=0.95,
        fatigue_sensitivity=0.45,
        reward_responsiveness=1.20,
        seed=seed,
    )

# ---------------------------------------------------------
# فعالیت‌های اختصاصی کاربر در طول زمان
# ---------------------------------------------------------

CUSTOM_ACTIVITY_POOL = [
    "پیاده‌روی کوتاه",
    "تمرین تنفس",
    "مرور فلش‌کارت",
    "تماس با مشاور",
    "مرتب‌سازی اتاق مطالعه",
    "گوش دادن به پادکست آموزشی",
    "مدیتیشن کوتاه",
    "تمرین تست زمان‌دار",
    "استراحت فعال",
    "مرور دفترچه اشتباهات",
    "برنامه‌ریزی هفتگی",
    "تماشای ویدئوی آموزشی",
    "تمرین تمرکز",
    "مطالعه آزاد",
    "حل تمرین اضافه",
    "مرور نکات کلاس",
    "یادداشت‌برداری",
    "تمرین سرعت تست",
    "جمع‌بندی روزانه",
    "آماده‌سازی وسایل فردا",
]

def maybe_add_custom_activity(user: User, day_index: int, rng: random.Random):
    """
    بعضی کاربران در طول زمان فعالیت اختصاصی اضافه می‌کنند.

    سقف اکشن‌ها:
    1 تا 6 فعالیت ثابت
    7 مطالعه
    8 آزمون
    9 تا 20 فعالیت اختصاصی
    """

    current_custom_count = type_Faliat_model.objects.filter(
        user=user
    ).exclude(id__lte=6).count()

    if current_custom_count >= MAX_CUSTOM_ACTIVITIES_PER_USER:
        return None

    label = get_adherence_profile(user)["label"]

    if day_index == 0:
        probability_by_label = {
            "very_low_adherence": 0.10,
            "low_medium_adherence": 0.08,
            "medium_high_adherence": 0.05,
            "high_adherence": 0.03,
            "perfect_adherence": 0.01,
        }

    elif day_index % 21 == 0:
        probability_by_label = {
            "very_low_adherence": 0.12,
            "low_medium_adherence": 0.10,
            "medium_high_adherence": 0.07,
            "high_adherence": 0.04,
            "perfect_adherence": 0.02,
        }

    elif day_index % 45 == 0:
        probability_by_label = {
            "very_low_adherence": 0.15,
            "low_medium_adherence": 0.12,
            "medium_high_adherence": 0.08,
            "high_adherence": 0.06,
            "perfect_adherence": 0.02,
        }

    else:
        probability_by_label = {
            "very_low_adherence": 0.008,
            "low_medium_adherence": 0.006,
            "medium_high_adherence": 0.004,
            "high_adherence": 0.003,
            "perfect_adherence": 0.001,
        }

    probability = probability_by_label.get(label, 0.005)

    if rng.random() > probability:
        return None

    existing_names = set(
        type_Faliat_model.objects.filter(user=user).values_list("name", flat=True)
    )

    candidates = [x for x in CUSTOM_ACTIVITY_POOL if x not in existing_names]

    if not candidates:
        return None

    name = rng.choice(candidates)

    obj = type_Faliat_model.objects.create(
        user=user,
        name=name,
        noeFaliat=None,
    )

    return obj

# ---------------------------------------------------------
# پیدا کردن آبجکت‌های لازم برای ثبت داده واقعی
# ---------------------------------------------------------

def get_or_create_activity_type_by_name(user: User, activity_name: str):
    """
    برای ذخیره در Faliat_model باید type_Faliat_model داشته باشیم.
    اگر فعالیت ثابت یا اختصاصی پیدا نشود، ساخته می‌شود.
    """

    activity_name = (activity_name or "سایر").strip()

    obj = type_Faliat_model.objects.filter(
        Q(user=user) | Q(user__isnull=True),
        name__iexact=activity_name
    ).first()

    if obj:
        return obj

    return type_Faliat_model.objects.create(
        user=user,
        name=activity_name,
        noeFaliat=None,
    )


def get_default_study_type():
    obj = noe_motaleh.objects.filter(name__icontains="مطالعه ساده").first()
    if obj:
        return obj

    obj = noe_motaleh.objects.first()
    if obj:
        return obj

    return noe_motaleh.objects.create(name="مطالعه ساده")


def get_study_type_by_text(text_value):
    if text_value:
        obj = noe_motaleh.objects.filter(name__icontains=str(text_value).strip()).first()
        if obj:
            return obj

    return get_default_study_type()


def get_default_exam_type():
    obj = type_azmoon_model.objects.first()
    if obj:
        return obj

    return type_azmoon_model.objects.create(
        name="آزمون شبیه‌سازی",
        is_testy=True,
    )


def get_book_for_schedule(user: User, schedule: DailySchedule):
    """
    DailySchedule در مدل فعلی bookname را به صورت متن ذخیره می‌کند.
    پس باید از روی نام کتاب، books_model را پیدا کنیم.
    """

    if schedule.bookname:
        book = books_model.objects.filter(
            user=user,
            name__iexact=str(schedule.bookname).strip()
        ).first()
        if book:
            return book

    return books_model.objects.filter(user=user).order_by("id").first()



def snap_report_value(value: float) -> int:
    """
    تبدیل مقدار پیوسته به مقیاس گزارش‌شده 20/40/60/80/100.
    """
    value = float(np.clip(value, 20.0, 100.0))
    return int(np.clip(round(value / 20.0) * 20, 20, 100))


def apply_between_day_recovery(psy, day_index: int):
    """
    بازیابی روزانه stress/fatigue به سمت baseline.

    منطق علمی:
    fatigue نسبت به stress حالت کوتاه‌مدت‌تر و فیزیولوژیک‌تر است؛
    بنابراین باید بعد از خواب شبانه/ریکاوری روزانه سریع‌تر از stress کاهش یابد.
    stress کندتر تعدیل می‌شود، چون به فشارهای تحصیلی و روانی تجمعی وابسته است.
    """

    stress_baseline = 0.42
    fatigue_baseline = 0.18
    satisfaction_baseline = 0.62

    current_stress = float(getattr(psy, "stress", stress_baseline))
    current_fatigue = float(getattr(psy, "fatigue", fatigue_baseline))
    current_satisfaction = float(getattr(psy, "satisfaction", satisfaction_baseline))

    # stress کندتر از fatigue بازیابی می‌شود.
    psy.stress = float(np.clip(
        0.62 * current_stress + 0.38 * stress_baseline,
        0.0,
        1.0,
    ))

    # fatigue باید سریع‌تر افت کند؛ در غیر این صورت در چند روز کوتاه به 1 می‌چسبد.
    psy.fatigue = float(np.clip(
        0.25 * current_fatigue + 0.75 * fatigue_baseline,
        0.0,
        1.0,
    ))

    psy.satisfaction = float(np.clip(
        0.82 * current_satisfaction + 0.18 * satisfaction_baseline,
        0.0,
        1.0,
    ))


def adjust_reported_activity_feelings(feelings: dict, action_kind: str, followed_plan: bool) -> dict:
    """
    تعدیل احساسات گزارش‌شده پس از فعالیت.

    تفاوت مهم:
    - psy.stress وضعیت پنهان/تجمعی دانش‌آموز است.
    - mizan_esters در Faliat_model باید stress لحظه‌ای پس از فعالیت را هم منعکس کند.

    بنابراین خواب، ورزش، تغذیه و استراحت فعال حتی اگر latent stress بالا باشد،
    باید در گزارش لحظه‌ای اثر کاهنده نشان دهند.
    """

    action_kind = str(action_kind or "activity")

    stress = float(feelings.get("stress", 60))
    satisfaction = float(feelings.get("satisfaction", 60))

    if action_kind == "sleep":
        stress_shift = -22
        sat_shift = +6

    elif action_kind == "sport":
        stress_shift = -18
        sat_shift = +6

    elif action_kind == "active_rest":
        stress_shift = -20
        sat_shift = +5

    elif action_kind == "nutrition":
        stress_shift = -12
        sat_shift = +5

    elif action_kind == "relaxing_entertainment":
        stress_shift = -14
        sat_shift = +4

    elif action_kind == "general_entertainment":
        stress_shift = -6
        sat_shift = +3

    elif action_kind == "screen_entertainment":
        # رضایت فوری ممکن است بالا باشد، اما stress را کاهش نمی‌دهد.
        stress_shift = +6
        sat_shift = +2

    elif action_kind == "class_activity":
        stress_shift = +4
        sat_shift = +2

    else:
        stress_shift = 0
        sat_shift = 0

    if followed_plan:
        # تبعیت از برنامه حس کنترل و رضایت می‌دهد.
        stress_shift += -4
        sat_shift += +5
    else:
        # عدم تبعیت هزینه کوچکی دارد، اما اثر مثبت خواب/ورزش را نابود نمی‌کند.
        if action_kind in ("sleep", "sport", "active_rest", "nutrition", "relaxing_entertainment"):
            stress_shift += +3
            sat_shift += -2
        elif action_kind == "screen_entertainment":
            stress_shift += +8
            sat_shift += -3
        else:
            stress_shift += +5
            sat_shift += -3

    out = dict(feelings)
    out["stress"] = snap_report_value(stress + stress_shift)
    out["satisfaction"] = snap_report_value(satisfaction + sat_shift)
    return out



def adjust_reported_study_feelings(feelings: dict, reward: float, intensity: float) -> dict:
    """
    تعدیل stress/satisfaction گزارش‌شده پس از مطالعه.

    مطالعه موفق، مخصوصاً وقتی intensity و reward بالاتر باشد، علاوه بر خستگی شناختی،
    حس پیشرفت ایجاد می‌کند و می‌تواند stress گزارش‌شده بعد از فعالیت را کاهش دهد.
    """

    reward = float(np.clip(reward, 0.0, 1.0))
    intensity = float(np.clip(intensity, 0.0, 1.0))

    stress = float(feelings.get("stress", 60))
    satisfaction = float(feelings.get("satisfaction", 60))

    # مطالعه کم‌کیفیت/اجباری می‌تواند کمی stress را بالا ببرد؛
    # مطالعه موفق و با reward بالا stress را پایین می‌آورد.
    stress_shift = +4 * intensity - 18 * reward
    satisfaction_shift = +18 * reward + 4 * intensity

    out = dict(feelings)
    out["stress"] = snap_report_value(stress + stress_shift)
    out["satisfaction"] = snap_report_value(satisfaction + satisfaction_shift)
    return out


def classify_activity_effect_kind(activity_name: str, activity_description: str = "", duration_hours: float = 0.5) -> str:
    """
    طبقه‌بندی اثر روان‌شناختی فعالیت واقعی.

    منطق علمی:
    - خواب، ورزش، تغذیه سالم و استراحت فعال معمولاً stress را کاهش می‌دهند.
    - سرگرمی‌های صفحه‌ای مثل گوشی، شبکه اجتماعی، بازی رایانه‌ای و فیلم/سریال
      می‌توانند رضایت کوتاه‌مدت ایجاد کنند، اما در صورت جایگزینی با فعالیت‌های مهم
      یا استفاده زیاد، stress/fatigue را افزایش دهند.
    - سرگرمی‌های آرام مثل کتاب غیردرسی یا گردش خانوادگی اثر آرام‌کننده‌تر دارند.
    """

    text = f"{activity_name or ''} {activity_description or ''}".strip().lower()

    if "خواب" in text or "چرت" in text:
        return "sleep"

    if "ورزش" in text or "پیاده" in text or "کششی" in text or "هوازی" in text:
        return "sport"

    if "تغذیه" in text or "صبحانه" in text or "نهار" in text or "شام" in text or "میان وعده" in text:
        return "nutrition"

    if "استراحت فعال" in text or "تنفس" in text or "مدیتیشن" in text:
        return "active_rest"

    if (
        "گوشی" in text
        or "شبکه" in text
        or "بازی رایانه" in text
        or "فیلم" in text
        or "سریال" in text
    ):
        return "screen_entertainment"

    if "کتاب" in text or "گردش" in text or "خانواده" in text:
        return "relaxing_entertainment"

    if "سرگرمی" in text:
        return "general_entertainment"

    if "کلاس" in text:
        return "class_activity"

    return "activity"


def apply_psychological_effect_from_params(
    psy,
    action_kind: str,
    intensity: float = 1.0,
    reward: float = 0.0,
    followed_plan: bool = True,
    exam_score: float | None = None,
):
    """
    مدل دو‌لایه و homeostatic اثر روان‌شناختی فعالیت‌ها.

    اصل علمی/مدلسازی:
    1. اثر خود فعالیت واقعی از اثر تبعیت از برنامه جداست.
    2. خواب، ورزش، تغذیه و استراحت فعال معمولاً stress/fatigue را کاهش می‌دهند.
    3. مطالعه و آزمون ممکن است فشار بسازند، اما موفقیت و تبعیت از برنامه رضایت/انگیزه را بالا می‌برد.
    4. اثر stress و fatigue خطی و بی‌نهایت تجمعی نیست؛ وقتی state نزدیک 1 است،
       افزایش‌های بعدی ضعیف‌تر و کاهش‌های تنظیم‌کننده مؤثرتر می‌شوند.
    """

    intensity = float(np.clip(intensity, 0.0, 1.0))
    reward = float(np.clip(reward, 0.0, 1.0))
    action_kind = str(action_kind or "activity")

    EVENT_SCALE = 0.25

    def S(value: float) -> float:
        return float(value) * EVENT_SCALE

    missed_academic = False

    if not followed_plan and action_kind == "study":
        missed_academic = True
        delta_stress = 0.018
        delta_hope = -0.010
        delta_motivation = -0.014
        delta_satisfaction = -0.016
        delta_fatigue = 0.002
        reward_value = 0.0

    elif not followed_plan and action_kind == "exam":
        missed_academic = True
        delta_stress = 0.024
        delta_hope = -0.014
        delta_motivation = -0.018
        delta_satisfaction = -0.020
        delta_fatigue = 0.003
        reward_value = 0.0

    elif action_kind == "study":
        delta_stress = 0.020 * intensity - 0.060 * reward
        delta_hope = 0.026 * intensity
        delta_motivation = P("k_mot_success", 0.040) * reward
        delta_satisfaction = P("k_sat_progress", 0.060) * reward
        delta_fatigue = P("k_fatigue_study", 0.026) * intensity
        reward_value = reward

    elif action_kind == "exam":
        score = 0.5 if exam_score is None else float(np.clip(exam_score, 0.0, 1.0))
        delta_stress = 0.034 - 0.130 * (score - 0.50)
        delta_hope = 0.035 if score >= 0.75 else (0.018 if score >= 0.60 else -0.026)
        delta_motivation = P("k_mot_success", 0.045) * (score - 0.50)
        delta_satisfaction = P("k_sat_progress", 0.065) * (score - 0.50)
        delta_fatigue = 0.014
        reward_value = score

    elif action_kind == "sleep":
        delta_stress = -abs(P("beta_sleep", 0.080)) * intensity
        delta_hope = 0.014
        delta_motivation = P("k_mot_sleep", 0.022) * intensity
        delta_satisfaction = P("k_sat_sleep", 0.042) * intensity
        delta_fatigue = -abs(P("k_fatigue_sleep", 0.090)) * intensity
        reward_value = 0.25

    elif action_kind == "sport":
        delta_stress = -abs(P("beta_exercise", 0.055)) * intensity
        delta_hope = 0.020 * intensity
        delta_motivation = 0.026 * intensity
        delta_satisfaction = 0.030 * intensity
        delta_fatigue = 0.006 * intensity
        reward_value = 0.22

    elif action_kind == "nutrition":
        delta_stress = -abs(P("beta_breakfast", 0.026)) * intensity
        delta_hope = 0.006
        delta_motivation = P("k_mot_breakfast", 0.016) * intensity
        delta_satisfaction = 0.024 * intensity
        delta_fatigue = -0.012 * intensity
        reward_value = 0.16

    elif action_kind == "active_rest":
        delta_stress = -0.042 * intensity
        delta_hope = 0.010
        delta_motivation = 0.004
        delta_satisfaction = 0.022 * intensity
        delta_fatigue = -0.030 * intensity
        reward_value = 0.18

    elif action_kind == "relaxing_entertainment":
        delta_stress = -0.026 * intensity
        delta_hope = 0.008
        delta_motivation = 0.002
        delta_satisfaction = 0.025 * intensity
        delta_fatigue = -0.010 * intensity
        reward_value = 0.16

    elif action_kind == "screen_entertainment":
        delta_stress = 0.006 * intensity
        delta_hope = -0.002
        delta_motivation = -0.008 * intensity
        delta_satisfaction = 0.016 * intensity
        delta_fatigue = 0.006 * intensity
        reward_value = 0.10

    elif action_kind == "general_entertainment":
        delta_stress = -0.010 * intensity
        delta_hope = 0.004
        delta_motivation = -0.002 * intensity
        delta_satisfaction = 0.018 * intensity
        delta_fatigue = -0.002 * intensity
        reward_value = 0.12

    elif action_kind == "class_activity":
        delta_stress = 0.006 * intensity
        delta_hope = 0.010 * intensity
        delta_motivation = 0.014 * intensity
        delta_satisfaction = 0.010 * intensity
        delta_fatigue = 0.008 * intensity
        reward_value = 0.12

    else:
        delta_stress = -0.002
        delta_hope = 0.004
        delta_motivation = 0.004
        delta_satisfaction = 0.006
        delta_fatigue = 0.000
        reward_value = 0.08

    if not missed_academic:
        if followed_plan:
            if action_kind == "screen_entertainment":
                delta_stress += -0.002
            elif action_kind in ("sleep", "sport", "nutrition", "active_rest", "relaxing_entertainment"):
                delta_stress += -0.010
            else:
                delta_stress += -0.006

            delta_hope += 0.004
            delta_motivation += 0.006
            delta_satisfaction += 0.012

        else:
            if action_kind in ("sleep", "sport", "nutrition", "active_rest", "relaxing_entertainment"):
                delta_stress += 0.003
                delta_motivation += -0.004
                delta_satisfaction += -0.003
            elif action_kind == "screen_entertainment":
                delta_stress += 0.010
                delta_motivation += -0.010
                delta_satisfaction += -0.006
            else:
                delta_stress += 0.006
                delta_motivation += -0.006
                delta_satisfaction += -0.005

    current_stress = float(np.clip(getattr(psy, "stress", 0.5), 0.0, 1.0))
    current_fatigue = float(np.clip(getattr(psy, "fatigue", 0.0), 0.0, 1.0))

    if delta_stress > 0:
        delta_stress *= max(0.15, 1.0 - current_stress)
    elif delta_stress < 0:
        delta_stress *= max(0.15, current_stress)

    if delta_fatigue > 0:
        delta_fatigue *= max(0.15, 1.0 - current_fatigue)
    elif delta_fatigue < 0:
        delta_fatigue *= max(0.15, current_fatigue)

    psy.apply_activity_effect(
        delta_stress=S(delta_stress),
        delta_hope=S(delta_hope),
        delta_motivation=S(delta_motivation),
        delta_satisfaction=S(delta_satisfaction),
        delta_fatigue=S(delta_fatigue),
        reward=reward_value,
        followed_plan=followed_plan,
    )

    if action_kind == "sleep" and hasattr(psy, "sleep_debt"):
        psy.sleep_debt = psy._clip01(
            float(psy.sleep_debt) + S(-0.080 * intensity)
        )

    return


# ---------------------------------------------------------
# ثبت داده واقعی براساس برنامه ذخیره‌شده
# ---------------------------------------------------------
def simulate_study(user, schedule, target_date, psy, behavior, rng, force_execute=None):
    book = get_book_for_schedule(user, schedule)

    if not book:
        return False

    start_page = schedule.startpage or 1
    end_page = schedule.endpage or min(start_page + 10, book.tedad_safahat)

    start_page = max(1, int(start_page))
    end_page = max(start_page, int(end_page))
    end_page = min(end_page, book.tedad_safahat)

    duration_hours = max(float(schedule.duringtime or 1.0), 0.1)

    executed, intensity, prob = behavior.execute_action(
        stress=psy.stress,
        motivation=psy.motivation,
        satisfaction=psy.satisfaction,
        fatigue=psy.fatigue,
        habit=psy.habit_strength,
        recent_failure_rate=psy.recent_failure_rate,
    )
    if force_execute is not None:
        executed = bool(force_execute)

        if executed and intensity <= 0:
            intensity = float(np.clip(
                prob + rng.normalvariate(0, 0.08),
                0.30,
                1.00
            ))

        if not executed:
            intensity = 0.0

    if not executed:
        apply_psychological_effect_from_params(
            psy=psy,
            action_kind="study",
            intensity=0.0,
            reward=0.0,
            followed_plan=False,
        )
        return False

    # اگر مطالعه اجرا شده، نباید مدت واقعی به چند دقیقه سقوط کند.
# intensity روی کیفیت/کامل‌بودن اثر می‌گذارد، نه اینکه کل زمان ثبت‌شده را نابود کند.
    real_duration = max(
        duration_hours * max(intensity, 0.60),
        min(duration_hours, 0.25),
    )
    end_time = add_hours_to_time(schedule.starttime, real_duration)

    previous_daily_study_hours = float(
        getattr(psy, "daily_study_hours", 0.0)
    )
    projected_daily_study_hours = (
        previous_daily_study_hours + real_duration
    )

    # v27 overload model:
    # - the first 8 realized hours are treated as a normal high-load day;
    # - load above 8 hours progressively reduces quality;
    # - load above 12 hours produces an additional severe-load penalty.
    overload_hours = max(
        0.0,
        projected_daily_study_hours - 8.0,
    )
    severe_overload_hours = max(
        0.0,
        projected_daily_study_hours - 12.0,
    )
    cognitive_load_penalty = min(
        0.25,
        0.025 * overload_hours
        + 0.035 * severe_overload_hours,
    )

    study_quality = float(np.clip(
        (
            0.50 * intensity
            + 0.30 * psy.motivation
            + 0.20 * (1.0 - psy.stress)
        )
        * (1.0 - cognitive_load_penalty),
        0.0,
        1.0,
    ))

    current_day = float(getattr(psy, "current_day_index", 0.0))
    previous_day = psy.last_study_day.get(int(book.id), None)

    if previous_day is None:
        dt_days = 1.0
    else:
        # اگر همان روز چند بار مطالعه/مرور شود، dt_days صفر نشود.
        dt_days = max(0.15, current_day - float(previous_day))

    is_review = "مرور" in str(schedule.typestudy or "")

    psy.update_memory(
        subject_id=book.id,
        study_quality=study_quality,
        dt_days=dt_days,
        is_review=is_review
    )

    psy.last_study_day[int(book.id)] = current_day

    apply_psychological_effect_from_params(
        psy=psy,
        action_kind="study",
        intensity=intensity,
        reward=study_quality,
        followed_plan=True,
    )

    psy.daily_study_hours = projected_daily_study_hours

    # Additional latent load after long study days. The homeostatic clipping
    # prevents unbounded accumulation, while still making a 13-15 hour day
    # visibly more stressful and fatiguing than an 8-hour day.
    psy.stress = float(np.clip(
        float(getattr(psy, "stress", 0.5))
        + 0.005 * overload_hours
        + 0.010 * severe_overload_hours,
        0.0,
        1.0,
    ))
    psy.fatigue = float(np.clip(
        float(getattr(psy, "fatigue", 0.2))
        + 0.010 * overload_hours
        + 0.015 * severe_overload_hours,
        0.0,
        1.0,
    ))

    feelings = psy.export_reported_feelings(
        action_kind="study",
        executed=True,
        performance_score=study_quality,
        rng=psy.rng,
    )
    
    feelings = adjust_reported_study_feelings(
        feelings=feelings,
        reward=study_quality,
        intensity=intensity,
    )

    if overload_hours > 0.0:
        feelings["stress"] = snap_report_value(
            float(feelings["stress"])
            + 2.0 * overload_hours
            + 3.0 * severe_overload_hours
        )
        feelings["satisfaction"] = snap_report_value(
            float(feelings["satisfaction"])
            - 1.0 * overload_hours
            - 1.5 * severe_overload_hours
        )

    study_model.objects.create(
            user=user,
            book=book,
            noeMotale=get_study_type_by_text(schedule.typestudy),
            difficulty=rng.choice(["easy", "medium", "hard"]),
            az_safhe=start_page,
            ta_safhe=end_page,
            az_saat=schedule.starttime,
            ta_saat=end_time,
            tarikh_per=gregorian_to_jalali_str(target_date),
            tarikh_en_user=target_date,
            tarikh_per_user=gregorian_to_jalali_str(target_date),
            sum_safhe=max(1, end_page - start_page + 1),
            mizan_esters=feelings["stress"],
            mizan_rezaiat=feelings["satisfaction"],
        )

    return True

#=========helper
def page_overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    a_start = int(a_start or 1)
    a_end = int(a_end or a_start)
    b_start = int(b_start or 1)
    b_end = int(b_end or b_start)

    left = max(a_start, b_start)
    right = min(a_end, b_end)

    if right < left:
        return 0.0

    exam_pages = max(1, a_end - a_start + 1)
    overlap_pages = right - left + 1
    return float(np.clip(overlap_pages / exam_pages, 0.0, 1.0))


def duration_hours_between(start_t, end_t) -> float:
    try:
        base_day = datetime.today().date()
        start_dt = datetime.combine(base_day, start_t)
        end_dt = datetime.combine(base_day, end_t)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
        return max(0.05, (end_dt - start_dt).total_seconds() / 3600.0)
    except Exception:
        return 1.0


def difficulty_to_penalty(value) -> float:
    text = str(value or "medium").lower()
    if "hard" in text or "سخت" in text:
        return 1.0
    if "easy" in text or "آسان" in text:
        return 0.0
    return 0.5


def get_exam_source_features(user, book, start_page, end_page, target_date, exam_starttime, rng):
    """
    استخراج ویژگی‌های واقعی از مطالعه‌های قبلی مرتبط با همین کتاب و همین بازه صفحات.
    خروجی‌ها در score آزمون استفاده می‌شوند:
    - coverage: آیا صفحات آزمون واقعاً قبلاً مطالعه/مرور شده‌اند؟
    - source_quality: کیفیت مطالعه‌های مرتبط
    - review_count: تعداد مرورهای مرتبط
    - difficulty_penalty: سختی میانگین مطالعه‌های مرتبط
    - days_since_last: فاصله از آخرین مطالعه مرتبط
    """
    qs = (
        study_model.objects
        .filter(
            user=user,
            book=book,
            tarikh_en_user__lte=target_date,
            ta_safhe__gte=start_page,
            az_safhe__lte=end_page,
        )
        .order_by("-tarikh_en_user", "-az_saat")[:25]
    )

    total_weight = 0.0
    weighted_quality = 0.0
    weighted_difficulty = 0.0
    coverage = 0.0
    review_count = 0
    days_since_last = None

    for st in qs:
        # اگر رکوردی در همان روز ولی بعد از آزمون باشد، نباید در نمره آزمون اثر بگذارد.
        try:
            if st.tarikh_en_user == target_date and st.az_saat and exam_starttime and st.az_saat >= exam_starttime:
                continue
        except Exception:
            pass

        overlap = page_overlap_ratio(start_page, end_page, st.az_safhe, st.ta_safhe)
        if overlap <= 0:
            continue

        try:
            gap_days = max(0, (target_date - st.tarikh_en_user).days)
        except Exception:
            gap_days = 3

        if days_since_last is None:
            days_since_last = gap_days
        else:
            days_since_last = min(days_since_last, gap_days)

        recency_weight = math.exp(-gap_days / 5.0)

        stress_norm = float(st.mizan_esters or 60) / 100.0
        sat_norm = float(st.mizan_rezaiat or 60) / 100.0
        duration_h = duration_hours_between(st.az_saat, st.ta_saat)

        pages = max(1, int((st.ta_safhe or st.az_safhe) - (st.az_safhe or 1) + 1))
        pages_per_hour = pages / max(duration_h, 0.05)

        # کیفیت مطالعه از رضایت، استرس، زمان کافی، و سرعت خیلی غیرطبیعی استخراج می‌شود.
        pace_quality = float(np.clip(1.0 - abs(pages_per_hour - 7.0) / 14.0, 0.30, 1.0))
        quality = (
            0.45
            + 0.25 * sat_norm
            - 0.20 * stress_norm
            + 0.20 * pace_quality
        )

        try:
            study_type_name = str(st.noeMotale.name or "")
        except Exception:
            study_type_name = ""

        if "مرور" in study_type_name:
            quality += 0.10
            review_count += 1

        difficulty_penalty = difficulty_to_penalty(getattr(st, "difficulty", "medium"))

        quality = float(np.clip(quality, 0.0, 1.0))
        weight = overlap * recency_weight

        coverage += overlap
        weighted_quality += quality * weight
        weighted_difficulty += difficulty_penalty * weight
        total_weight += weight

    if total_weight <= 0:
        # اگر overlap دقیق پیدا نشد، نباید فرض کنیم دانش‌آموز هیچ چیزی بلد نیست.
        # چون آزمون توسط شوان از برنامه پیشنهادی ساخته شده و در تست 3روزه فاصله واقعی زیاد نیست.
        # fallback محافظه‌کارانه اما غیرصفر:
        return {
            "coverage": 0.45,
            "source_quality": 0.55,
            "review_count": 0,
            "difficulty_penalty": 0.45,
            "days_since_last": 1.5,
            "gap_days": 1.5,
            "min_gap_hours": 36.0,
        }

    final_days_since_last = float(days_since_last if days_since_last is not None else 7.0)

    return {
        "coverage": float(np.clip(coverage, 0.0, 1.0)),
        "source_quality": float(np.clip(weighted_quality / total_weight, 0.0, 1.0)),
        "review_count": int(review_count),
        "difficulty_penalty": float(np.clip(weighted_difficulty / total_weight, 0.0, 1.0)),
        "days_since_last": final_days_since_last,
        "gap_days": final_days_since_last,
        "min_gap_hours": final_days_since_last * 24.0,
    }
def simulate_exam(user, schedule, target_date, psy, behavior, rng, force_execute=None):
    book = get_book_for_schedule(user, schedule)

    if not book:
        return False

    start_page = schedule.startpage or 1
    end_page = schedule.endpage or min(start_page + 20, book.tedad_safahat)

    start_page = max(1, int(start_page))
    end_page = max(start_page, int(end_page))
    end_page = min(end_page, book.tedad_safahat)

    duration_hours = max(float(schedule.duringtime or 0.5), 0.1)

    executed, intensity, prob = behavior.execute_action(
        stress=psy.stress,
        motivation=psy.motivation,
        satisfaction=psy.satisfaction,
        fatigue=psy.fatigue,
        habit=psy.habit_strength,
        recent_failure_rate=psy.recent_failure_rate,
    )

    if force_execute is not None:
        executed = bool(force_execute)

        if executed and intensity <= 0:
            intensity = float(np.clip(
                prob + rng.normalvariate(0, 0.08),
                0.30,
                1.00
            ))

        if not executed:
            intensity = 0.0

    if not executed:
        apply_psychological_effect_from_params(
            psy=psy,
            action_kind="exam",
            intensity=0.0,
            reward=0.0,
            followed_plan=False,
        )
        
        return False

    # اگر آزمون اجرا شده، مدت آزمون نباید 4 دقیقه شود.
    # intensity باید کیفیت اجرا را تغییر دهد، نه اینکه آزمون معتبر را از نظر زمانی نابود کند.
    real_duration = max(duration_hours * max(intensity, 0.70), min(duration_hours, 0.25))
    end_time = add_hours_to_time(schedule.starttime, real_duration)

    memory_value = float(psy.memory_strength[int(book.id)])
    # قبلاً تقسیم بر 5 باعث می‌شد اثر حافظه کم شود.
    # اینجا mastery حساس‌تر شده تا مطالعه/مرور موفق اثر واضح‌تری روی score بگذارد.
    normalized_memory = float(np.clip(memory_value / 3.5, 0.0, 1.0))

    source_features = get_exam_source_features(
        user=user,
        book=book,
        start_page=start_page,
        end_page=end_page,
        target_date=target_date,
        exam_starttime=schedule.starttime,
        rng=rng,
    )

    coverage = source_features["coverage"]
    source_quality = source_features["source_quality"]
    review_count = source_features["review_count"]
    difficulty_penalty = source_features["difficulty_penalty"]
    days_since_last = source_features["days_since_last"]

    repetition_count = int(psy.repetitions[int(book.id)])
    review_bonus = float(np.clip(review_count / 3.0, 0.0, 1.0))

    # ------------------------------------------------------------
    # Scientific scoring model
    # ------------------------------------------------------------
    # اصل مدل:
    #   R(t) = 2 ^ (-delta / h)
    # که در آن:
    #   delta = فاصله از آخرین مطالعه/مرور مرتبط
    #   h     = نیمه‌عمر حافظه، وابسته به کیفیت مطالعه، تکرار، مرور، استرس و خستگی
    #
    # نکته مهم:
    # دیگر memory_value decay‌شده را دوباره در retention ضرب نمی‌کنیم.
    # encoding_strength نشان‌دهنده قدرت یادگیری در زمان مطالعه است،
    # و retention_factor فقط یک بار افت زمانی را اعمال می‌کند.

    motivation = float(psy.motivation)
    stress = float(psy.stress)
    fatigue = float(psy.fatigue)
    habit_strength = float(psy.habit_strength)

    difficulty = difficulty_penalty
    study_quality = source_quality

    repetition_signal = float(np.clip(
        math.log1p(max(0, repetition_count)) / math.log(6.0),
        0.0,
        1.0,
    ))

    # قدرت رمزگذاری/یادگیری قبل از فراموشی.
    # coverage شرط محتوایی است، source_quality کیفیت مطالعه را می‌دهد،
    # repetition و review پایداری یادگیری را افزایش می‌دهند.
    encoding_strength = float(np.clip(
        0.50 * source_quality
        + 0.20 * coverage
        + 0.20 * repetition_signal
        + 0.10 * review_bonus,
        0.0,
        1.0,
    ))

    # برای سازگاری با trace قبلی، نام mastery_signal را نگه می‌داریم.
    mastery_signal = encoding_strength

    # نیمه‌عمر پایه یک پارامتر کالیبراسیون مدل شبیه‌سازی است، نه تغییر قانون آبینگهاوس.
    # این مقدار در مقاله باید به عنوان base half-life گزارش شود و بعداً می‌توان sensitivity analysis گرفت.
    base_half_life_days = 2.0

    quality_factor = 0.70 + 0.60 * source_quality
    repetition_factor = 1.00 + 0.30 * math.log1p(max(0, repetition_count))
    review_factor = 1.00 + 0.25 * max(0, review_count)
    stress_factor = 1.00 - 0.25 * stress
    fatigue_factor = 1.00 - 0.15 * fatigue

    half_life = (
        base_half_life_days
        * quality_factor
        * repetition_factor
        * review_factor
        * stress_factor
        * fatigue_factor
    )

    # برای جلوگیری از مقادیر غیرواقعی در شبیه‌سازی.
    # این کف score نیست؛ محدودسازی پارامتر نیمه‌عمر است.
    half_life = float(np.clip(half_life, 0.75, 14.0))

    # برای سازگاری با نام ستون trace قبلی، stability را معادل half_life ذخیره می‌کنیم.
    stability = half_life

    # تابع فراموشی نمایی/half-life.
    retention_factor = float(2.0 ** (-float(days_since_last) / half_life))

    # احتمال یادآوری در زمان آزمون.
    recall_probability = float(np.clip(
        coverage * encoding_strength * retention_factor,
        0.0,
        1.0,
    ))

    # برای سازگاری با trace قبلی.
    effective_mastery = recall_probability

    # اگر آزمون تستی است، baseline حدس‌زنی 25 درصد است.
    # این score_floor نیست؛ مؤلفه آزمون‌شناختی برای آزمون چهارگزینه‌ای است.
    exam_type = get_default_exam_type()
    guessing_baseline = 25.0 if bool(getattr(exam_type, "is_testy", False)) else 0.0

    cognitive_score = guessing_baseline + (100.0 - guessing_baseline) * recall_probability

    behavior_adjustment = (
        + 8.0 * (float(intensity) - 0.70)
        + 8.0 * (motivation - 0.50)
        + 4.0 * (habit_strength - 0.50)
        - 8.0 * (stress - 0.50)
        - 5.0 * (fatigue - 0.50)
        - 6.0 * difficulty_penalty
    )

    score = cognitive_score + behavior_adjustment + rng.normalvariate(0, 4)

    score = int(np.clip(round(score), 0, 100))
    append_exam_scoring_trace(locals())
    #===================
    apply_psychological_effect_from_params(
        psy=psy,
        action_kind="exam",
        intensity=intensity,
        reward=score / 100.0,
        followed_plan=True,
        exam_score=score / 100.0,
    )
    

    total_questions = 20
    correct = int(round(total_questions * score / 100.0))
    wrong = max(0, total_questions - correct)

    feelings = psy.export_reported_feelings(
        action_kind="exam",
        executed=True,
        performance_score=score / 100.0,
        rng=psy.rng,
    )

    azmoon_model.objects.create(
        user=user,
        book=book,
        type_azmoon=exam_type,
        #text_type_azmoon="آزمون شبیه‌سازی",
        text_type_azmoon=f"آزمون {book.name} از صفحه {start_page} تا {end_page}",
        az_saat=schedule.starttime,
        ta_saat=end_time,
        az_safhe=start_page,
        ta_safhe=end_page,
        tarikh_en_user=target_date,
        tarikh_per_user=gregorian_to_jalali_str(target_date),
        javab_dorost=correct,
        javab_ghalat=wrong,
        javab_darsad=score,
    )

    return True


def simulate_activity(user, schedule, target_date, psy, behavior, rng, force_execute=None):
    original_activity_name = str(schedule.actiontype or "سایر").strip()
    activity_name = original_activity_name
    duration_hours = max(float(schedule.duringtime or 0.5), 0.05)

    executed, intensity, prob = behavior.execute_action(
        stress=psy.stress,
        motivation=psy.motivation,
        satisfaction=psy.satisfaction,
        fatigue=psy.fatigue,
        habit=psy.habit_strength,
        recent_failure_rate=psy.recent_failure_rate,
    )

    if force_execute is not None:
        executed = bool(force_execute)

        if executed and intensity <= 0:
            intensity = float(np.clip(
                prob + rng.normalvariate(0, 0.08),
                0.30,
                1.00
            ))

        if not executed:
            intensity = 0.0

    followed_plan = bool(executed)

    # اگر طبق برنامه عمل نکرد، یک فعالیت واقعی جایگزین ثبت می‌کنیم.
    # جایگزین‌ها فقط سرگرمی نیستند؛ گاهی خواب/استراحت/ورزش هم ممکن است رخ دهد.
    if not followed_plan:
        activity_name = rng.choices(
            ["سرگرمی", "سایر", "استراحت فعال", "خواب", "ورزش"],
            weights=[0.45, 0.20, 0.18, 0.12, 0.05],
            k=1,
        )[0]

        duration_hours = min(duration_hours, 0.75)
        intensity = rng.uniform(0.35, 0.85)

    # توضیح فعالیت باید قبل از اثر روانی ساخته شود تا نوع سرگرمی مشخص شود.
    activity_description = build_activity_description(
        user=user,
        schedule=schedule,
        activity_name=activity_name,
        rng=rng,
    )

    action_kind = classify_activity_effect_kind(
        activity_name=activity_name,
        activity_description=activity_description,
        duration_hours=duration_hours,
    )

    # اثر خواب روی sleep_debt جداگانه هم ثبت شود.
    if action_kind == "sleep":
        psy.update_sleep(
            slept_hours=min(duration_hours, 8.0),
            sleep_need_hours=8.0
        )

    apply_psychological_effect_from_params(
        psy=psy,
        action_kind=action_kind,
        intensity=intensity,
        reward=0.2 if followed_plan else 0.0,
        followed_plan=followed_plan,
    )

    real_duration = max(duration_hours * max(intensity, 0.3), 0.05)
    end_time = add_hours_to_time(schedule.starttime, real_duration)

    activity_type = get_or_create_activity_type_by_name(user, activity_name)

    if not followed_plan:
        activity_description = (
            f"عدم تبعیت از برنامه پیشنهادی | "
            f"برنامه پیشنهادی: {original_activity_name} | "
            f"فعالیت واقعی: {activity_name} | "
            f"{activity_description}"
        )

    # برای export_reported_feelings از گروه کلی استفاده می‌کنیم تا اگر state.py
    # برای subtypeهای جدید branch نداشت، مشکلی پیش نیاید.
    reported_kind = action_kind
    if action_kind in ("screen_entertainment", "relaxing_entertainment", "general_entertainment"):
        reported_kind = "entertainment"
    elif action_kind == "active_rest":
        reported_kind = "activity"
    elif action_kind == "class_activity":
        reported_kind = "activity"

    feelings = psy.export_reported_feelings(
        action_kind=reported_kind,
        executed=followed_plan,
        performance_score=None,
        rng=psy.rng,
    )

    feelings = adjust_reported_activity_feelings(
        feelings=feelings,
        action_kind=action_kind,
        followed_plan=followed_plan,
    )

    Faliat_model.objects.create(
        user=user,
        type_faliat=activity_type,
        text_type_faliat=activity_name,
        az_saat=schedule.starttime,
        ta_saat=end_time,
        tarikh_per=gregorian_to_jalali_str(target_date),
        tarikh_en_user=target_date,
        tarikh_per_user=gregorian_to_jalali_str(target_date),
        mizan_esters=feelings["stress"],
        mizan_rezaiat=feelings["satisfaction"],
        tozihat=activity_description,
    )

    # True یعنی طبق برنامه پیشنهادی عمل کرده.
    # False یعنی فعالیت جایگزین انجام داده.
    return followed_plan


def simulate_one_user_day(user, target_date, day_index, psy, behavior, rng):
    psy.current_day_index = float(day_index)
    # v27: reset the cumulative realized study load at the start of each day.
    psy.daily_study_hours = 0.0
    schedules = list(
        DailySchedule.objects
        .filter(user=user, date=target_date)
        .order_by("starttime")
    )
    execution_map = build_execution_plan_for_day(
        user=user,
        schedules=schedules,
        day_index=day_index,
        rng=rng,
    )

    if not schedules:
        return {
            "study": 0,
            "exam": 0,
            "activity": 0,
            "skipped": 0,
        }

    counters = {
        "study": 0,
        "exam": 0,
        "activity": 0,
        "skipped": 0,
    }

    # برای جلوگیری از دوباره‌سازی داده‌ها در اجرای مجدد command
    study_model.objects.filter(user=user, tarikh_en_user=target_date).delete()
    Faliat_model.objects.filter(user=user, tarikh_en_user=target_date).delete()
    azmoon_model.objects.filter(user=user, tarikh_en_user=target_date).delete()

    for schedule in schedules:
        force_execute = execution_map.get(schedule.id, False)
        action_text = str(schedule.actiontype or "")

        try:
            if "مطالعه" in action_text:
                ok = simulate_study(
                    user=user,
                    schedule=schedule,
                    target_date=target_date,
                    psy=psy,
                    behavior=behavior,
                    rng=rng,
                    force_execute=force_execute,
                )
                counters["study" if ok else "skipped"] += 1

            elif "آزمون" in action_text:
                ok = simulate_exam(
                    user=user,
                    schedule=schedule,
                    target_date=target_date,
                    psy=psy,
                    behavior=behavior,
                    rng=rng,
                    force_execute=force_execute,
                )
                counters["exam" if ok else "skipped"] += 1

            else:
                ok = simulate_activity(
                    user=user,
                    schedule=schedule,
                    target_date=target_date,
                    psy=psy,
                    behavior=behavior,
                    rng=rng,
                    force_execute=force_execute,
                )
                counters["activity" if ok else "skipped"] += 1

        #except Exception as e:
        #    counters["skipped"] += 1
        except Exception as e:
            counters["skipped"] += 1

            import traceback
            print(
                "\nSIMULATION ITEM ERROR",
                "| user=", user.username,
                "| date=", target_date,
                "| schedule_id=", getattr(schedule, "id", None),
                "| actiontype=", getattr(schedule, "actiontype", None),
                "| bookname=", getattr(schedule, "bookname", None),
                "| pages=", getattr(schedule, "startpage", None), "-", getattr(schedule, "endpage", None),
                "| error=", repr(e),
            )
            traceback.print_exc()
        '''
        except Exception as e:
            import traceback

            counters["skipped"] += 1

            print(
                "\nSIMULATION ITEM ERROR",
                "| user=", user.username,
                "| date=", target_date,
                "| schedule_id=", getattr(schedule, "id", None),
                "| actiontype=", getattr(schedule, "actiontype", None),
                "| error=", repr(e),
            )
            traceback.print_exc()
        '''
        
    psy.daily_decay()

    return counters
# ---------------------------------------------------------
# Command اصلی
# ---------------------------------------------------------

class Command(BaseCommand):
    help = "Run shoan algorithm and psychological simulator for USim1..USim1000 for 330 days."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=DAYS,
            help="Number of simulation days. Default: 330"
        )

        parser.add_argument(
            "--start-date",
            type=str,
            default=START_DATE.strftime("%Y-%m-%d"),
            help="Gregorian start date. Default: 2025-07-26"
        )

        parser.add_argument(
            "--from-user",
            type=int,
            default=FIRST_USER_INDEX,
            help="First USim index. Default: 1"
        )

        parser.add_argument(
            "--to-user",
            type=int,
            default=LAST_USER_INDEX,
            help="Last USim index. Default: 100"
        )

        parser.add_argument(
            "--seed", type=int, default=14040504,
            help="Paired random seed used identically by Full and Static."
        )
        parser.add_argument(
            "--history-window", type=int, default=120,
            help="Rolling educational-history window in days. Default: 120"
        )
        parser.add_argument(
            "--activity-lookback", type=int, default=60,
            help="Custom-activity recency mask in days. Default: 60"
        )
        parser.add_argument(
            "--initial-updates", type=int, default=64,
            help="Offline SAC gradient updates when a user model is first built."
        )
        parser.add_argument(
            "--daily-updates", type=int, default=8,
            help="Offline SAC gradient updates on later nights."
        )
        parser.add_argument(
            "--checkpoint-dir", type=str, default="dma_dsac_checkpoints",
            help="Directory for per-user policy checkpoints."
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Only print what would be done, without writing data."
        )

    def handle(self, *args, **options):
        global DAYS, START_DATE, FIRST_USER_INDEX, LAST_USER_INDEX

        DAYS = int(options["days"])
        START_DATE = datetime.strptime(options["start_date"], "%Y-%m-%d").date()
        FIRST_USER_INDEX = int(options["from_user"])
        LAST_USER_INDEX = int(options["to_user"])

        dry_run = bool(options["dry_run"])

        shoan.configure_runtime(
            mode="static",
            base_seed=int(options["seed"]),
            history_window_days=int(options["history_window"]),
            activity_lookback_days=int(options["activity_lookback"]),
            initial_updates=int(options["initial_updates"]),
            daily_updates=int(options["daily_updates"]),
            checkpoint_root=str(options["checkpoint_dir"]),
            training_log_path=f"paper_exports_final/dma_dsac_training_log_v27_2_1a_static.csv",
        )

        users = get_simulated_users()

        if not users:
            run_days = int(options.get("days") or DAYS)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Shoan simulation completed successfully | days={run_days} | users={len(users)}"
                )
            )
            return

        self.stdout.write(
            self.style.SUCCESS(
                f"Found {len(users)} simulated users: "
                f"USim{FIRST_USER_INDEX}..USim{LAST_USER_INDEX}"
            )
        )

        #=======
        # state و behavior را در طول روزهای شبیه‌سازی در حافظه نگه می‌داریم.
        # نکته مهم:
        # build_psychological_state در این پروژه seed می‌گیرد، نه rng.
        # همچنین نباید بعد از warm-start دوباره psy_by_user را از صفر بسازیم.
        psy_by_user = {}
        behavior_by_user = {}
        warmstart_summary = {}

        base_seed = int(options.get("seed", 14040504) or 14040504)

        for user in users:
            user_seed = base_seed + get_user_number(user)

            psy = build_psychological_state(user, seed=user_seed)

            warm_info = warm_start_psychological_state_from_history(
                user=user,
                psy=psy,
                start_date=START_DATE,
            )

            psy_by_user[user.id] = psy
            behavior_by_user[user.id] = build_behavior_model(user, seed=user_seed)
            warmstart_summary[user.username] = warm_info

        print("Warm-start completed.")
        print(
            "Warm-start totals | studies=",
            sum(x["history_study_count"] for x in warmstart_summary.values()),
            "| exams=",
            sum(x["history_exam_count"] for x in warmstart_summary.values()),
        )
        #====================================
        for day_index in range(DAYS):
            target_date = START_DATE + timedelta(days=day_index)
            jalali_date = gregorian_to_jalali_str(target_date)

            self.stdout.write(
                self.style.WARNING(
                    f"\n===== DAY {day_index + 1}/{DAYS} | {target_date} | {jalali_date} ====="
                )
            )

            if dry_run:
                self.stdout.write("DRY RUN: no data will be written.")
                continue

            # -------------------------------------------------
            # بازیابی روزانه latent stress/fatigue قبل از شروع رویدادهای روز
            # -------------------------------------------------
            for user in users:
                apply_between_day_recovery(
                    psy=psy_by_user[user.id],
                    day_index=day_index,
                )

            # -------------------------------------------------
            # مرحله صفر: بعضی کاربران فعالیت اختصاصی اضافه می‌کنند
            # -------------------------------------------------
            added_custom_count = 0

            for user in users:
                user_rng = random.Random(900000 + get_user_number(user) * 17 + day_index)
                obj = maybe_add_custom_activity(user, day_index, user_rng)
                if obj:
                    added_custom_count += 1

            self.stdout.write(
                f"Custom activities added today: {added_custom_count}"
            )

            # پاک‌سازی کش shoan برای اینکه داده‌های دیروز وارد تصمیم امروز شود
            try:
                shoan.get_cached_dfstat.cache_clear()
                shoan.get_cached_dfexam.cache_clear()
            except Exception:
                pass

            # -------------------------------------------------
            # مرحله ۱: اجرای الگوریتم برای همه کاربران
            # -------------------------------------------------
            planned_count = 0
            failed_plan_count = 0

            for idx, user in enumerate(users, start=1):
                self.stdout.write(
                    f"ACTIVE | phase=planning | day={day_index + 1}/{DAYS} | "
                    f"date={target_date} | user={idx}/{len(users)} | "
                    f"username={user.username}"
                )
                try:
                    with transaction.atomic():
                        ok = shoan.run_algorithm_for_user_id(
                            student_id=user.id,
                            schedule_date=target_date,
                            clear_existing_schedule=True,
                        )

                    if ok:
                        planned_count += 1
                    else:
                        failed_plan_count += 1

                except Exception as e:
                    failed_plan_count += 1
                    self.stdout.write(
                        self.style.ERROR(
                            f"Algorithm failed for {user.username} on {target_date}: {e}"
                        )
                    )

                if idx % 50 == 0:
                    self.stdout.write(
                        f"Planning progress: {idx}/{len(users)}"
                    )

                if idx % 25 == 0:
                    # v27: keep cached per-user agents alive; only collect Python garbage.
                    gc.collect()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Planning finished | ok={planned_count} | failed={failed_plan_count}"
                )
            )

            # v27.2 fail-closed policy: a day is valid only when every selected
            # learner receives a verified schedule containing daily study.
            # Do not continue to the behavioural simulator with partial plans.
            if failed_plan_count:
                raise CommandError(
                    "v27.2.1a planning aborted because "
                    f"{failed_plan_count} user(s) failed on {target_date}."
                )

            # بعد از تولید برنامه‌ها، کش را پاک کن تا شبیه‌سازی و روز بعد داده تازه ببیند
            try:
                shoan.get_cached_dfstat.cache_clear()
                shoan.get_cached_dfexam.cache_clear()
            except Exception:
                pass

            # -------------------------------------------------
            # مرحله ۲: اجرای شبیه‌ساز براساس DailySchedule
            # -------------------------------------------------
            total_study = 0
            total_exam = 0
            total_activity = 0
            total_skipped = 0

            for idx, user in enumerate(users, start=1):
                self.stdout.write(
                    f"ACTIVE | phase=simulation | day={day_index + 1}/{DAYS} | "
                    f"date={target_date} | user={idx}/{len(users)} | "
                    f"username={user.username}"
                )
                user_rng = random.Random(1200000 + get_user_number(user) * 31 + day_index)

                try:
                    with transaction.atomic():
                        result = simulate_one_user_day(
                            user=user,
                            target_date=target_date,
                            day_index=day_index,
                            psy=psy_by_user[user.id],
                            behavior=behavior_by_user[user.id],
                            rng=user_rng,
                        )
                    total_study += result["study"]
                    total_exam += result["exam"]
                    total_activity += result["activity"]
                    total_skipped += result["skipped"]

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f"Simulation failed for {user.username} on {target_date}: {e}"
                        )
                    )

                if idx % 50 == 0:
                    self.stdout.write(
                        f"Simulation progress: {idx}/{len(users)}"
                    )

                if idx % 25 == 0:
                    # v27: keep cached per-user agents alive; only collect Python garbage.
                    gc.collect()

            self.stdout.write(
                self.style.SUCCESS(
                    f"Simulation finished | "
                    f"study={total_study} | exam={total_exam} | "
                    f"activity={total_activity} | skipped={total_skipped}"
                )
                
            )
                    # v27: keep cached per-user agents alive; only collect Python garbage.

            gc.collect()

            # روز بعد باید داده‌های امروز را ببیند
            try:
                shoan.get_cached_dfstat.cache_clear()
                shoan.get_cached_dfexam.cache_clear()
            except Exception:
                pass

        shoan.clear_runtime_agent_cache(save=True)

        self.stdout.write(
            self.style.SUCCESS("\n330-day DMA-DSAC v27.2.1a (static) simulation completed successfully.")
        )