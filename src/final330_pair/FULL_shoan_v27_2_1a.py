
# V27.2.1A_ATOMIC_DAILY_FLOOR = True
import os

V2721_DURATION_QUANTIZATION_HOTFIX = True
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import sys
from study.models import noe_motaleh,study_model,Faliat_model,azmoon_model,type_Faliat_model,books_model
from django.db.models import Min
import datetime
#---------------- kaka shoan code---------------------------------------
import pandas as pd
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # فقط خطاهای فATAL نمایش داده می‌شوند
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # خاموش کردن oneDNN warnings
import tensorflow as tf
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass
tf.get_logger().setLevel('ERROR')  # فقط خطاها
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from collections import deque
import random
from tensorflow.keras import layers
import datetime
from datetime import datetime, timedelta
import sys
import random
from collections import defaultdict
import logging
import os
import math
from study.models import type_azmoon_model
from study.models import DailySchedule
from django.contrib.auth.models import User
from .models import study_model, Faliat_model, books_model, type_Faliat_model
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import cProfile
import pstats
from functools import lru_cache
from django.db.models import Q
import diskcache as dc
# در ابتدای فایل بعد از imports اضافه کنید:
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': True,
    'constant_folding': True, 
    'shape_optimization': True,
    'remapping': True,
    'arithmetic_optimization': True,
    'dependency_optimization': True,
    'loop_optimization': True,
    'function_optimization': True,
    'debug_stripper': True,
    'scoped_allocator_optimization': True,
    'pin_to_host_optimization': True,
    'implementation_selector': True,
    'enable_grappler_remapping': False,
    'disable_meta_optimizer': True,  # غیرفعال کردن برخی بهینه‌سازها
})
TRAIN_SIGNATURE = (tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32))
# 🌟 تعریف UPDATE_SIGNATURE_SAC:
# این امضا برای update(self, states, actions, rewards, next_states) است
UPDATE_SIGNATURE_SAC = (
    tf.TensorSpec(shape=[None, 16], dtype=tf.float32), # 1. States (فرض بر 16 ویژگی)
    tf.TensorSpec(shape=[None, 1], dtype=tf.int32),    # 2. Actions (فرض بر 1D)
    tf.TensorSpec(shape=[None, 1], dtype=tf.float32),  # 3. Rewards
    tf.TensorSpec(shape=[None, 16], dtype=tf.float32)  # 4. Next States
)
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
# ساخت پوشه لاگ اگر وجود نداشت
log_dir = os.path.join(os.getcwd(), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# پیکربندی لاگر
logging.basicConfig(
    filename=os.path.join(log_dir, "shoan.log"),
    level=logging.ERROR,  # only log errors to reduce verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'  # اضافه کردن به لاگ
)

logger = logging.getLogger(__name__)
#---------پایان لاگ

# Utility decorators to add uniform logging for functions and class methods
import functools

def log_call(func):
    """Decorator to log function entry, exit, arguments and exceptions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Try to extract user context: look for 'user' kwarg, User instances in args,
            # or a .user attribute on the first arg (common for methods). Fall back to CURRENT_USER_ID.
            user_info = None
            try:
                # check kwargs
                if 'user' in kwargs:
                    u = kwargs.get('user')
                    if hasattr(u, 'id'):
                        user_info = f"id={getattr(u,'id', None)} username={getattr(u,'username', None)}"
                    else:
                        user_info = repr(u)
                else:
                    # scan args
                    for a in args:
                        try:
                            if isinstance(a, User):
                                user_info = f"id={a.id} username={a.username}"
                                break
                        except Exception:
                            pass
                        if hasattr(a, 'user'):
                            try:
                                u = getattr(a, 'user')
                                if isinstance(u, User):
                                    user_info = f"id={u.id} username={u.username}"
                                    break
                                elif isinstance(u, int):
                                    user_info = f"id={u}"
                                    break
                            except Exception:
                                pass
                if user_info is None and 'CURRENT_USER_ID' in globals() and globals().get('CURRENT_USER_ID'):
                    uid = globals().get('CURRENT_USER_ID')
                    try:
                        u = User.objects.get(id=uid)
                        user_info = f"id={uid} username={u.username}"
                    except Exception:
                        user_info = f"id={uid}"
            except Exception:
                user_info = 'unknown'

            # Log a concise error message with context and full traceback (single entry)
            logger.error(f"Exception in {func.__qualname__} | user={user_info} | error={e}", exc_info=True)
            raise
    return wrapper

def log_methods(cls):
    """Class decorator that wraps all callable attributes (methods) with log_call."""
    for attr_name, attr_value in list(cls.__dict__.items()):
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, log_call(attr_value))
    return cls


# باز کردن فایل برای نوشتن
#sys.stdout = open('outputnn.txt', 'w')
batch_size = 32 #1
daily_schedule = []
CURRENT_USER_ID = None  # پیش‌فرض برای جلوگیری از NameError در import time
NUM_COLUMNS = None  # پیش‌فرض برای جلوگیری از NameError در import time
# پیش‌فرض‌های امن برای جلوگیری از NameError در توابعی که قبل از تنظیم مقادیر واقعی فراخوانی می‌شوند
dfstat = pd.DataFrame()
dfexam = pd.DataFrame()
column_names = []
# کدهیی که برای شبیه سازی سیصد روز اضافه شده
SIMULATION_TARGET_DATE = None


def set_simulation_target_date(target_date):
    """
    تاریخ هدف برنامه‌ریزی را برای generate_daily_schedule تنظیم می‌کند.
    اگر None باشد، کد مثل قبل از تاریخ امروز استفاده می‌کند.
    """
    global SIMULATION_TARGET_DATE

    if target_date is None:
        SIMULATION_TARGET_DATE = None
        return

    if isinstance(target_date, str):
        SIMULATION_TARGET_DATE = datetime.strptime(target_date, "%Y-%m-%d").date()
    elif isinstance(target_date, datetime):
        SIMULATION_TARGET_DATE = target_date.date()
    else:
        SIMULATION_TARGET_DATE = target_date

#------------------- tabe ejraie barname ---------------------------------------------
# فرض می‌کنیم تمام مدل‌ها و کلاس‌های لازم (مثل User, SAC, StudyEnvironment, create_dfexam, get_user_action_space) import شده‌اند.

@log_call
def runalgoritm():
    # متغیرهای سراسری که نیاز به تعریف دارند
    global CURRENT_USER_ID, NUM_COLUMNS, column_names 
    logger.info("Starting runalgoritm")
    print("Starting runalgoritm")

    try:
        from account.models import user_reg_model

        # گرفتن تمام یوزر آی دی های دانش آموزان
        students = user_reg_model.objects.filter(
            moshaver__gte=0
        ).values_list("user_id", flat=True)

        for student_id in students:
            try:
                logger.info(f"Running algorithm for student {student_id}")
                CURRENT_USER_ID = student_id 
                user = User.objects.get(id=student_id)
                logger.error(
                    f"CHECK USER {student_id} | "
                    f"study={study_model.objects.filter(user=user).count()} | "
                    f"activity={Faliat_model.objects.filter(user=user).count()} | "
                    f"exam={azmoon_model.objects.filter(user=user).count()}"
                )

                #اگر یوزری فعالیت یا مطالعه یا آزمون ثبت نکرده باشد الگوریتم برای او اجرا نمی شود
                has_study = study_model.objects.filter(user=user).exists()
                has_activity = Faliat_model.objects.filter(user=user).exists()
                has_exam = azmoon_model.objects.filter(user=user).exists()

                missing = []
                if not has_study:
                    missing.append("study")
                if not has_activity:
                    missing.append("activity")
                if not has_exam:
                    missing.append("exam")

                if missing:
                    logger.error(
                        f"Skipping user {student_id}: missing data -> {', '.join(missing)}"
                    )
                    continue

                
                # 🔥 ساخت نقشه معکوس برای کتاب‌ها 🔥
                user_books = books_model.objects.filter(user=user).order_by('id') 
                sac_index_to_db_id = {}
                for index, book in enumerate(user_books, 1):
                    sac_index_to_db_id[index] = book.id

                # ۱. آماده‌سازی داده‌ها و بررسی‌های اولیه
                df = get_cached_dfstat(student_id)
                exam_file = get_cached_dfexam(student_id)

                if df.empty:
                    logger.info(f"Skipping user {student_id}: dfstat is empty")
                    continue
                # اگر exam_file خالی بود، باز هم ادامه می‌دهیم (طبق منطق شما)
                if exam_file.empty:
                    logger.info(f"User {student_id}: dfexam is empty, continuing without exam data.")
                
                # ----------------------------------------------------------------------------------
                # 🔥 ۲. محاسبه فضای عمل پویا (Discrete Action Space) - فراخوانی فقط یک بار و صحیح 🔥
                # ----------------------------------------------------------------------------------
                user_action_dim, user_idx_to_act, user_act_to_idx, rl_id_map = get_user_action_space(user)
                logger.debug(f"Sanity check - RL id map keys: {sorted(list(rl_id_map.keys()))}")
                if 1 not in user_idx_to_act.values() or 2 not in user_idx_to_act.values():
                    logger.warning("Warning: RL action space does not include mandatory actions 1 or 2 (study/exam). Check get_user_action_space.")
                # چک تکراری نبودن DB ids
                db_ids = list(rl_id_map.values())
                if len(db_ids) != len(set(db_ids)):
                    logger.error("Duplicate DB ids in rl_id_map! This will cause incorrect reverse mapping.")
                    raise RuntimeError("Duplicate DB ids detected in rl_id_map.")
                # 🔥 دیباگ ۱: بررسی لیست فعالیت‌های مجاز 🔥

                
                if user_action_dim == 0:
                    logger.warning(f"Skipping user {student_id}: No actions found (dim=0)")
                    continue
                    
                logger.info(f"User {student_id} has {user_action_dim} total actions.")
                
                # ... (بقیه کدهای محاسبه min_date_of_all_data) ...
                
                # دریافت حداقل تاریخ از هر مدل
                min_study_date = study_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
                min_activity_date = Faliat_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
                min_exam_date = azmoon_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
                
                # پیدا کردن کوچکترین تاریخ معتبر
                valid_dates = list(filter(None, [min_study_date, min_activity_date, min_exam_date]))
                if not valid_dates:
                    logger.error("No valid activity/study/exam data found for min_date calculation.")
                    min_date_of_all_data = None 
                else:
                    min_date_of_all_data = min(valid_dates)
                    
            except Exception as e:
                logger.error(f"Error generating action space for user {student_id}: {e}")
                continue   # *** این حیاتی است ***
                    
            # اگر min_date نامعتبر است، باید جلوی اجرای الگوریتم گرفته شود
            if min_date_of_all_data is None:
                logger.error("Cannot run algorithm: min_date_of_all_data is None.")
                continue # پایان دادن به تابع

            # ----------------------------------------------------------------------------------
            # --- ۲. فراخوانی prepare_rl_data با آرگومان جدید ---
            # ----------------------------------------------------------------------------------
            FINAL_DF = prepare_rl_data(
                df_stat=df,
                df_exam=exam_file, 
                user=user, 
                min_date=min_date_of_all_data,
                # 🔥 پاس دادن نگاشت جدید برای تبدیل IDهای دیتابیس به RL ID 🔥
                rl_id_to_db_id_map=rl_id_map, 
                max_reward_exam=1500.0, 
                max_reward_final_scale=1.0 
            )
            
            # 🌟🌟🌟 افزودن کدهای استانداردسازی داده‌ها (قبل از استفاده در مدل) 🌟🌟🌟
            # ... (کدهای استانداردسازی df) ...
            df = df.fillna(0) # پر کردن هر گونه NaN باقی مانده
            df = df.astype(np.float32) # تبدیل به نوع داده مورد انتظار Keras
            df = df.replace([np.inf, -np.inf], 0) # حذف مقادیر بی نهایت

            # ۳. تعریف ابعاد state و تنظیم متغیرهای سراسری
            num_columns = df.shape[1]
            state_dim = (1, num_columns) 
            action_dim = user_action_dim 
                
            NUM_COLUMNS = num_columns
            column_names = df.columns.tolist()
            # ۳. محاسبه ابعاد و متغیرهای مورد نیاز
            # نکته: num_actions باید از تعداد کل اکشن‌های RL که در مرحله قبل محاسبه کردید (user_action_dim) گرفته شود
            num_actions = user_action_dim             
            # ۴. ساخت محیط (Environment)
            study_env = StudyEnvironment(
                df_final_data=FINAL_DF,
                df_stat_features=df,
                df_exam_features=exam_file, 
                min_date=min_date_of_all_data,
                target_action_dim=num_actions  
            ) 

            # ۵. دریافت وضعیت اولیه برای تشخیص ابعاد نهایی (مثلاً ۲۵)
            initial_state = study_env.reset()
            real_state_dim = np.array(initial_state).shape[-1] 

            # ۶. ساخت ایجنت SAC با تمام آرگومان‌های درخواستی
            # توجه: آرگومان‌ها را دقیقاً به ترتیبی که در SAC.__init__ اصلاح کردیم می‌فرستیم
            # --- اصلاح در تابع runalgoritm ---
            # در تابع runalgoritm
            # ...
            
            # ۶. ساخت ایجنت SAC
            sac_agent = SAC(
                state_dim=real_state_dim,
                action_dim=num_actions,
                max_action=1,
                study_env=study_env,
                num_columns=real_state_dim,
                action_maps={
                    'index_to_actiontype': user_idx_to_act,
                    'actiontype_to_index': user_act_to_idx
                },
                sac_index_to_db_id=sac_index_to_db_id, # نگاشت کتاب‌ها
                
                # 🔥 خط اصلاح شده: حتماً این آرگومان را اضافه کنید
                rl_id_to_db_id_map=rl_id_map 
            )
                        
            sac_agent.generate_daily_schedule()
            logger.info(f"Algorithm completed successfully for student {student_id}")

    except Exception as e:
        logger.error(f"Error in runalgoritm: {str(e)}", exc_info=True)
        raise
#=============تابع برای اجرای شبیه سازی سیصد روز =========
@log_call
def run_algorithm_for_user_id(student_id, schedule_date=None, clear_existing_schedule=True):
    """
    اجرای الگوریتم shoan فقط برای یک دانش‌آموز.

    کاربرد در شبیه‌سازی:
    - برای هر روز
    - برای هر یوزر USim1 تا USim1000
    - برنامه همان روز هدف در DailySchedule ذخیره می‌شود
    """

    global CURRENT_USER_ID, NUM_COLUMNS, column_names

    if schedule_date is not None:
        set_simulation_target_date(schedule_date)

    CURRENT_USER_ID = student_id
    user = User.objects.get(id=student_id)

    if clear_existing_schedule and schedule_date is not None:
        DailySchedule.objects.filter(
            user=user,
            date=schedule_date
        ).delete()

    logger.info(f"Running shoan algorithm for simulated user {student_id} on {schedule_date}")

    # اگر یوزری فعالیت یا مطالعه یا آزمون نداشته باشد، الگوریتم برای او اجرا نمی‌شود
    has_study = study_model.objects.filter(user=user).exists()
    has_activity = Faliat_model.objects.filter(user=user).exists()
    has_exam = azmoon_model.objects.filter(user=user).exists()

    missing = []
    if not has_study:
        missing.append("study")
    if not has_activity:
        missing.append("activity")
    if not has_exam:
        missing.append("exam")

    if missing:
        logger.error(
            f"Skipping simulated user {student_id}: missing data -> {', '.join(missing)}"
        )
        return False

    # پاک‌کردن کش، چون در شبیه‌سازی هر روز داده‌های جدید ساخته می‌شود
    try:
        get_cached_dfstat.cache_clear()
        get_cached_dfexam.cache_clear()
    except Exception:
        pass

    # نگاشت کتاب‌ها
    user_books = books_model.objects.filter(user=user).order_by('id')
    sac_index_to_db_id = {}
    for index, book in enumerate(user_books, 1):
        sac_index_to_db_id[index] = book.id

    # ساخت دیتافریم‌ها
    df = get_cached_dfstat(student_id)
    exam_file = get_cached_dfexam(student_id)

    if df.empty:
        logger.info(f"Skipping simulated user {student_id}: dfstat is empty")
        return False

    # فضای عمل پویا
    user_action_dim, user_idx_to_act, user_act_to_idx, rl_id_map = get_user_action_space(user)

    if user_action_dim == 0:
        logger.warning(f"Skipping simulated user {student_id}: No actions found")
        return False

    # محاسبه حداقل تاریخ داده‌های کاربر
    min_study_date = study_model.objects.filter(user=user).aggregate(
        min_date=Min('tarikh_en_user')
    )['min_date']

    min_activity_date = Faliat_model.objects.filter(user=user).aggregate(
        min_date=Min('tarikh_en_user')
    )['min_date']

    min_exam_date = azmoon_model.objects.filter(user=user).aggregate(
        min_date=Min('tarikh_en_user')
    )['min_date']

    valid_dates = list(filter(None, [min_study_date, min_activity_date, min_exam_date]))

    if not valid_dates:
        logger.error(f"Cannot run algorithm for user {student_id}: no valid dates")
        return False

    min_date_of_all_data = min(valid_dates)

    # آماده‌سازی داده RL
    FINAL_DF = prepare_rl_data(
        df_stat=df,
        df_exam=exam_file,
        user=user,
        min_date=min_date_of_all_data,
        rl_id_to_db_id_map=rl_id_map,
        max_reward_exam=1500.0,
        max_reward_final_scale=1.0
    )

    df = df.fillna(0)
    df = df.astype(np.float32)
    df = df.replace([np.inf, -np.inf], 0)

    num_columns = df.shape[1]
    NUM_COLUMNS = num_columns
    column_names = df.columns.tolist()

    num_actions = user_action_dim

    study_env = StudyEnvironment(
        df_final_data=FINAL_DF,
        df_stat_features=df,
        df_exam_features=exam_file,
        min_date=min_date_of_all_data,
        target_action_dim=num_actions
    )

    initial_state = study_env.reset()
    real_state_dim = np.array(initial_state).shape[-1]

    sac_agent = SAC(
        state_dim=real_state_dim,
        action_dim=num_actions,
        max_action=1,
        study_env=study_env,
        num_columns=real_state_dim,
        action_maps={
            'index_to_actiontype': user_idx_to_act,
            'actiontype_to_index': user_act_to_idx
        },
        sac_index_to_db_id=sac_index_to_db_id,
        rl_id_to_db_id_map=rl_id_map
    )

    sac_agent.generate_daily_schedule()

    logger.info(
        f"Algorithm completed for simulated user {student_id} on {schedule_date}"
    )

    return True
#============تابع ادغام جدول ها و تشکیل دیتا فریم ====================
@log_call
def create_dfstat(user):
    logger.info(f"Starting create_dfstat for user {user.username}")
    
    columns = [
        'date', 'actiontype', 'bookname', 'startpage', 'endpage',
        'starttime', 'duringtime', 'typestudy', 'oxijen', 'estres',
        'dificulty', 'repitnumber', 'dutimerev', 'typenutrition',
        'typefood', 'typeentertinment'
    ]
    empty_df = pd.DataFrame(columns=columns)

    try:
        # 1. ساخت نگاشت برای فعالیت‌های کاربر (Custom Activities)
        # فعالیت‌های ۱ تا ۶ ثابت هستند. فعالیت‌های ID > 6 باید به 9, 10, ... نگاشت شوند.
        # نکته: فرض می‌کنیم در دیتابیس شما IDهای ۱ تا ۶ رزرو شده‌اند.
        custom_activities = type_Faliat_model.objects.filter(user=user).exclude(id__lte=6).order_by('id')
        
        # دیکشنری تبدیل DB_ID -> RL_ID برای فعالیت‌های کاستوم
        # مثلاً اگر کاربر فعالیتی با ID=12 دارد، می‌شود RL=9
        custom_map = {}
        start_rl_index = 9
        for act in custom_activities:
            custom_map[int(act.id)] = start_rl_index
            start_rl_index += 1
            
        # ---------------------------------------------------------

        has_study_data = study_model.objects.filter(user=user).exists()
        has_activity_data = Faliat_model.objects.filter(user=user).exists()

        if not has_study_data and not has_activity_data:
            return empty_df
        
        # پیدا کردن مینیمم تاریخ
        min_dates = []
        if has_study_data:
            min_dates.append(study_model.objects.filter(user=user).aggregate(m=Min('tarikh_en_user'))['m'])
        if has_activity_data:
            min_dates.append(Faliat_model.objects.filter(user=user).aggregate(m=Min('tarikh_en_user'))['m'])
        
        valid_dates = [d for d in min_dates if d is not None]
        if not valid_dates:
             return empty_df
        min_date = min(valid_dates)
        min_date = pd.to_datetime(min_date)

        # === بخش ۱: داده‌های مطالعه (Study) ===
        if has_study_data:
            study_data = list(study_model.objects.filter(user=user).values(
                'tarikh_en_user', 'book__name', 'az_safhe', 'ta_safhe',
                'az_saat', 'ta_saat', 'noeMotale__name', 'mizan_rezaiat', 'mizan_esters'
            ))
            df_study = pd.DataFrame(study_data)
            
            # نگاشت کتاب‌ها
            book_names = list(books_model.objects.filter(user=user).values_list('name', flat=True))
            books = {name: idx + 1 for idx, name in enumerate(book_names)}
            book_pages = {b.name: b.tedad_safahat for b in books_model.objects.filter(user=user)}

            df_study['tarikh_en_user'] = pd.to_datetime(df_study['tarikh_en_user'])
            df_study['date'] = (df_study['tarikh_en_user'] - min_date).dt.days + 1
            df_study['bookname'] = df_study['book__name'].map(books).fillna(0).astype(int)
            
            df_study['startpage'] = df_study.apply(lambda x: x['az_safhe'] / book_pages.get(x['book__name'], 1), axis=1)
            df_study['endpage'] = df_study.apply(lambda x: x['ta_safhe'] / book_pages.get(x['book__name'], 1), axis=1)
            
            # نرمال‌سازی زمان
            df_study['starttime'] = df_study['az_saat'].apply(lambda x: x.hour/24 + x.minute/(24*60))
            df_study['duringtime'] = df_study.apply(lambda x: (x['ta_saat'].hour - x['az_saat'].hour)/24 + (x['ta_saat'].minute - x['az_saat'].minute)/(24*60), axis=1)
            
            # 🔥🔥 تغییر مهم: مطالعه همیشه ۷ است 🔥🔥
            df_study['actiontype'] = 7 
            
            df_study['oxijen'] = df_study['mizan_rezaiat']
            df_study['estres'] = df_study['mizan_esters']
            
            # نوع مطالعه
            noe_motaleh_objects = noe_motaleh.objects.all()
            name_to_number = {obj.name: index + 1 for index, obj in enumerate(noe_motaleh_objects)}
            df_study['typestudy'] = df_study['noeMotale__name'].map(name_to_number).fillna(0).astype(int)
            
            df_study = df_study[['date', 'actiontype', 'bookname', 'startpage', 'endpage', 'starttime', 'duringtime', 'typestudy', 'oxijen', 'estres']].fillna(0)
        else:
            df_study = pd.DataFrame(columns=columns)

        # === بخش ۲: داده‌های فعالیت (Activity) ===
        if has_activity_data:
            activity_data = list(Faliat_model.objects.filter(user=user).values(
                'tarikh_en_user', 'type_faliat_id', 'az_saat', 'ta_saat', 'mizan_rezaiat', 'mizan_esters'
            ))
            df_activity = pd.DataFrame(activity_data)
            df_activity['tarikh_en_user'] = pd.to_datetime(df_activity['tarikh_en_user'])
            df_activity['date'] = (df_activity['tarikh_en_user'] - min_date).dt.days + 1
            
            # 🔥🔥 تغییر مهم: نگاشت شناسه‌ها 🔥🔥
            def map_activity_id(db_id):
                db_id = int(db_id)
                # اگر ۱ تا ۶ است (ثابت‌ها)، خودش بماند
                if 1 <= db_id <= 6:
                    return db_id
                # اگر کاستوم است (در دیکشنری ما وجود دارد)، تبدیل به ۹، ۱۰ و... شود
                if db_id in custom_map:
                    return custom_map[db_id]
                # در غیر این صورت (مثلاً فعالیتی که حذف شده یا ID عجیب دارد)
                return 0 # یا یک کد خطا

            df_activity['actiontype'] = df_activity['type_faliat_id'].apply(map_activity_id)
            # حذف ردیف‌هایی که ID نامعتبر داشتند (0)
            df_activity = df_activity[df_activity['actiontype'] != 0]

            df_activity['oxijen'] = df_activity['mizan_rezaiat']
            df_activity['estres'] = df_activity['mizan_esters']
            df_activity['starttime'] = df_activity['az_saat'].apply(lambda x: x.hour/24 + x.minute/(24*60))
            df_activity['duringtime'] = df_activity.apply(lambda x: (x['ta_saat'].hour - x['az_saat'].hour)/24 + (x['ta_saat'].minute - x['az_saat'].minute)/(24*60), axis=1)
            
            df_activity = df_activity[['date', 'actiontype', 'starttime', 'duringtime','oxijen', 'estres']].fillna(0)
        else:
            df_activity = pd.DataFrame(columns=['date', 'actiontype', 'starttime', 'duringtime','oxijen', 'estres'])

        # ادغام و خروجی
        dfstat = pd.concat([df_study, df_activity], ignore_index=True)
        for col in columns:
            if col not in dfstat.columns:
                dfstat[col] = 0
        
        return dfstat[columns].fillna(0)

    except Exception as e:
        logger.error(f"Error in create_dfstat: {e}")
        return empty_df
#================محابسه پاداش آزمون============================
def calculate_exam_reward(exam_type_value, score, previous_score, time_gap, max_reward, alpha1=0.1): 
    # وزن‌ها (فرض می‌کنیم exam_type_value عدد صحیح باشد)
    weights = {1: 1.0, 2: 0.9, 3: 0.75, 4: 0.5} 
    weight = weights.get(exam_type_value, 0.5)
    
    # ورودی‌ها در اینجا 0 تا 1 هستند (score/previous_score)
    score_f = float(score)
    previous_score_f = float(previous_score)
    time_gap_f = float(time_gap)
    
    # محاسبه پاداش پیشرفت و نمره
    # نمره از 0 تا 100 در نظر گرفته شود تا پیشرفت تاثیر بزرگتری داشته باشد (همان منطق قبلی شما):
    score_100 = score_f * 100
    previous_score_100 = previous_score_f * 100
    
    avsc = ((score_100 + previous_score_100) / 2)
    progress_reward = avsc * (score_100 - previous_score_100) # پاداش پیشرفت
    
    # پاداش اصلی (وزن * نمره * عامل فاصله زمانی) - نمره 0 تا 100
    main_reward = weight * score_100 * (1 + alpha1 * time_gap_f)
    
    final_reward = main_reward + progress_reward
    
    # ماکزیمم پاداش خام را حدس می‌زنیم (مثلاً 1500)
    # اگر max_reward از بیرون بیاید، بهتر است.
    normalized_reward = final_reward / max_reward
    return np.clip(normalized_reward, -1.0, 1.0) 
    
#===============================جدول امتحان=====================
# دریافت داده‌های آزمون از دیتابیس و ایجاد دیتافریم
@log_call
def create_dfexam(user, max_reward_exam=1500.0):
    logger.info("Starting create_dfexam")
    STATE_COLS = [
        'date', 'actiontype', 'bookname', 'startpage', 'endpage',
        'starttime', 'duringtime', 'typestudy', 'oxijen', 'estres',
        'dificulty', 'repitnumber', 'dutimerev', 'typenutrition',
        'typefood', 'typeentertinment'
    ]
    # ستون‌های آزمون که در State استفاده نمی‌شوند (مانند score و R_Exam)
    EXAM_REWARD_COLS = ['exam_type', 'previous_score', 'score', 'timebetweenstudy', 'R_Exam']
    FINAL_COLUMNS = STATE_COLS + EXAM_REWARD_COLS

    # --- ۲. دریافت حداقل تاریخ (MIN_DATE) --- <--- اصلاح حیاتی: این بخش اضافه شد
    try:
        min_study_date = study_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
        min_activity_date = Faliat_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
        min_exam_date = azmoon_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
        
        # پیدا کردن کوچکترین تاریخ (با فیلتر کردن مقادیر None)
        valid_dates = list(filter(None, [min_study_date, min_activity_date, min_exam_date]))
        if not valid_dates:
            min_date_val = None
        else:
            min_date_val = min(valid_dates)
            
    except Exception as e:
        logger.error(f"Error calculating min_date: {e}")
        min_date_val = None

    # --- ۳. واکشی نگاشت‌ها (Mapping) --- (کد شما در مورد نگاشت کتاب‌ها و نوع آزمون)
    # ... (کدهای موجود برای واکشی books و exam_types)
    try:
        # نگاشت کتاب‌ها 
        book_names = list(books_model.objects.filter(user=user).values_list('name', flat=True))
    except Exception:
        book_names = list(books_model.objects.values_list('name', flat=True))
    books = {name: idx+1 for idx, name in enumerate(book_names)}

    # نگاشت نوع آزمون
    exam_type_qs = type_azmoon_model.objects.order_by('id').values_list('name', flat=True)
    exam_types = {name: idx + 1 for idx, name in enumerate(exam_type_qs)}
    
    # --- ۴. خواندن داده‌های آزمون از دیتابیس ---
    azmoon_data = list(azmoon_model.objects.filter(user=user).values(
        'tarikh_en_user', 'book__name', 'az_safhe', 'ta_safhe',
        'az_saat', 'ta_saat', 'type_azmoon__name', 'javab_darsad'
    ))
    df_exam = pd.DataFrame(azmoon_data)

    # --- ۵. بررسی شرایط خالی بودن داده ---
    if df_exam.empty or min_date_val is None: # اگر min_date هم نامعتبر باشد
        logger.info("No exam data or valid min_date found. Returning empty DataFrame.")
        return pd.DataFrame(columns=FINAL_COLUMNS)

    # --- ۶. پردازش داده‌ها و ایجاد ستون‌های ویژگی ---
    
    # تبدیل تاریخ‌ها با استفاده از min_date_val
    min_date = pd.to_datetime(min_date_val)
    df_exam['date'] = (pd.to_datetime(df_exam['tarikh_en_user']) - min_date).dt.days + 1
    
    # ... (کدهای موجود برای map کردن bookname)
    df_exam['bookname'] = df_exam['book__name'].map(books)
    
    # ... (کدهای موجود برای محاسبه startpage و endpage)
    try:
        book_pages = {b.name: b.tedad_safahat for b in books_model.objects.filter(user=user)}
    except Exception:
        book_pages = {b.name: b.tedad_safahat for b in books_model.objects.all()}
    df_exam['startpage'] = df_exam.apply(
        lambda x: x['az_safhe'] / book_pages.get(x['book__name'], 1), axis=1
    )
    df_exam['endpage'] = df_exam.apply(
        lambda x: x['ta_safhe'] / book_pages.get(x['book__name'], 1), axis=1
    )
    
    # ... (کدهای موجود برای محاسبه starttime و duringtime)
    df_exam['starttime'] = df_exam['az_saat'].apply(
        lambda x: x.hour / 24 + x.minute / (24 * 60) if pd.notnull(x) else 0
    )
    df_exam['duringtime'] = df_exam.apply(
        lambda x: (x['ta_saat'].hour - x['az_saat'].hour) / 24 +
                 (x['ta_saat'].minute - x['az_saat'].minute) / (24 * 60) if pd.notnull(x['az_saat']) and pd.notnull(x['ta_saat']) else 0, axis=1
    )
    
    # ... (کدهای موجود برای محاسبه exam_type، score، و متغیرهای موقت/پیش‌فرض)
    df_exam['exam_type'] = df_exam['type_azmoon__name'].map(exam_types)
    df_exam['score'] = df_exam['javab_darsad'] / 100.0
    #df_exam['previous_score'] = 0
    #df_exam['timebetweenstudy'] = 0
    # ---- محاسبه نمره قبلی و فاصله زمانی بر اساس کتاب و صفحه ----
    # اول بر اساس کتاب، صفحه و تاریخ مرتب می‌کنیم که ترتیب زمانی درست باشد
    df_exam = df_exam.sort_values(['bookname', 'startpage', 'date'])

    # نمره قبلی برای همان کتاب و همان محدوده صفحه
    df_exam['previous_score'] = df_exam.groupby(
        ['bookname', 'startpage']
    )['score'].shift(1).fillna(0.0)

    # فاصله زمانی (به روز) تا امتحان قبلی همان کتاب و صفحه
    df_exam['timebetweenstudy'] = df_exam.groupby(
        ['bookname', 'startpage']
    )['date'].diff().fillna(0.0)

    df_exam['actiontype'] = 8 # کد فعالیت انحصاری برای آزمون
    df_exam['actiontype'] = df_exam['actiontype'].astype(int)
    # 🔥🔥 اصلاح حیاتی: اضافه کردن ستون‌های مفقود (مانند difliculty و...) 🔥🔥
    # ستون‌های اصلی dfstat:

    # اضافه کردن ستون‌های مفقود State به df_exam (با مقدار 0)
    for col in STATE_COLS:
        if col not in df_exam.columns:
            df_exam[col] = 0
    
    df_exam['typestudy'] = 0
    df_exam['oxijen'] = df_exam['oxijen'].fillna(0)
    df_exam['estres'] = df_exam['estres'].fillna(0)

    # --- ۷. محاسبه ستون پاداش (R_Exam) ---
    df_exam['R_Exam'] = df_exam.apply(
        lambda row: calculate_exam_reward(
            row['exam_type'], 
            row['score'], 
            row['previous_score'], 
            row['timebetweenstudy'],
            max_reward=max_reward_exam
        ),
        axis=1
    )
    
    # این خط اکنون امن است
    df_exam = df_exam[FINAL_COLUMNS].fillna(0)

    return df_exam
    
#==================================================================

def calculate_future_metrics(df, idx, max_reward_exam, min_date):
    """
    نسخه اصلاح شده برای معماری جدید:
    مطالعه = 7
    آزمون = 8
    """
    R_study_future = 0.0  
    R_exam_future = 0.0 
    try:
        if df is None or idx is None:
            return 0.0, 0.0
        
        # ... (بخش‌های ابتدایی تابع بدون تغییر) ...

        # ردیف جاری
        try:
            row = df.loc[idx]
        except Exception:
            return 0.0, 0.0

        cur_action = int(row.get('actiontype', -1))
        
        # 🔥 اصلاح ۱: آزمون الان ۸ است، نه ۲ 🔥
        if cur_action == 8:  
            R_exam_future = float(row.get('R_Exam_Score', 0.0) or 0.0)
            return float(R_study_future), float(R_exam_future)

        # جستجوی آزمون بعدی
        try:
            subsequent_idx = df.index[df.index > idx]
        except Exception:
            subsequent_idx = [i for i in df.index if i > idx]

        next_exam_idx = None
        for j in subsequent_idx:
            try:
                # 🔥 اصلاح ۲: جستجو برای آزمون (کد ۸) 🔥
                if int(df.at[j, 'actiontype']) == 8: 
                    next_exam_idx = j
                    break
            except Exception:
                continue

        # ... (بخش دریافت نمره آزمون بعدی بدون تغییر) ...

        # محاسبه پاداش مطالعه در آینده
        if next_exam_idx is not None:
            span_idx = [j for j in df.index if j > idx and j < next_exam_idx]
        else:
            span_idx = [j for j in df.index if j > idx]

        study_future_score = 0.0
        for j in span_idx:
            try:
                # 🔥 اصلاح ۳: جمع کردن پاداش برای مطالعه (کد ۷) 🔥
                if int(df.at[j, 'actiontype']) == 7: 
                    if 'R_Study' in df.columns:
                        study_future_score += float(df.at[j].get('R_Study', 0.0) or 0.0)
                    else:
                        study_future_score += 0.3 #بررسی شود
            except Exception:
                continue

        R_study_future = study_future_score
        if next_exam_idx is not None:  
            next_score = float(df.at[next_exam_idx, 'R_Exam_Score'] or 0.0)  
            R_exam_future = next_score  
        else:  
            R_exam_future = 0.0

        return float(R_study_future), float(R_exam_future)

    except Exception as e:
        return 0.0, 0.0
#===========================================================
@log_call
def prepare_rl_data(df_stat, df_exam, user, min_date, rl_id_to_db_id_map=None, max_reward_exam=1500.0, max_reward_final_scale=1.0):
    # لیست ستون‌های ویژگی حالت
    STATE_FEATURES_COLS = [
        'date', 'actiontype', 'bookname', 'startpage', 'endpage',
        'starttime', 'duringtime', 'typestudy', 'oxijen', 'estres',
        'dificulty', 'repitnumber', 'dutimerev', 'typenutrition',
        'typefood', 'typeentertinment'
    ]
    
    MAX_CHANGE_RANGE = 60 # (80 - 20)
    
    # 1. آماده‌سازی و ادغام دیتافریم‌ها
    #df_stat['R_Exam_Score'] = 0.0
    df_stat = df_stat.rename(columns={'mizan_rezaiat': 'oxijen', 'mizan_esters': 'estres'}, errors='ignore')
    
    for col in ['oxijen', 'estres']: 
        df_exam[col] = 0.0 
        
    df_final = pd.concat([df_stat, df_exam], ignore_index=True)
    df_final = df_final.sort_values(by=['date', 'starttime']).reset_index(drop=True)
    # اطمینان از اینکه RL-ID عدد صحیح است (مطالعه=1، آزمون=2، فعالیت‌ها=3..n)
    df_final['actiontype'] = df_final['actiontype'].astype(int)
        # ---- اتصال R_Exam (که در create_dfexam ساخته شده) به R_Exam_Score ----
    # اگر ستون R_Exam از create_dfexam آمده باشد، آن را به R_Exam_Score نگاشت می‌کنیم.
    # برای ردیف‌های df_stat که آزمون نیستند، مقدارش 0 می‌شود.
    if 'R_Exam' in df_final.columns:
        df_final['R_Exam_Score'] = df_final['R_Exam'].fillna(0.0)
    else:
        # اگر بنا به هر دلیل R_Exam هنوز ساخته نشده باشد، صفر بگذار
        df_final['R_Exam_Score'] = 0.0
    
    # 🔥 اصلاح ۱: حذف NaNهای احتمالی قبل از محاسبات 🔥
    df_final = df_final.fillna(0)
    # ----------------------------------------------------
    # 🔥 پایان: درج کدهای تبدیل ID 🔥
    # ----------------------------------------------------
    # اضافه کن:
    unique_actions = sorted(df_final['actiontype'].unique())
    logger.debug(f"prepare_rl_data: unique RL action types in dataset after mapping: {unique_actions}")

    # اگر تو انتظار داری اکشن‌های 1..8 و 9..n باشند، حداقل 1 و 2 باید باشند
    if 1 not in unique_actions or 2 not in unique_actions:
        logger.warning("prepare_rl_data: action types 1 or 2 missing in dataset after mapping — check DB -> RL mapping.")

    # 2. محاسبه پاداش‌های آنی (R_Stress, R_Satisfy)
    df_final['Prev_Estres'] = df_final['estres'].shift(1).fillna(0)
    df_final['Prev_Oxijen'] = df_final['oxijen'].shift(1).fillna(0)
    
    df_final['R_Stress_Change'] = (df_final['Prev_Estres'] - df_final['estres']) / MAX_CHANGE_RANGE
    df_final['R_Satisfy_Change'] = (df_final['oxijen'] - df_final['Prev_Oxijen']) / MAX_CHANGE_RANGE
    
    # 3. محاسبه پاداش‌های تأخیری (Look-Ahead)
    # این بخش فقط یک بار باید اجرا شود
    results = [
        calculate_future_metrics(df_final, i, max_reward_exam, min_date) 
        for i in df_final.index
    ]
    
    # پر کردن لیست‌ها و حذف Noneها
    df_final['R_Study_Future'] = [r[0] if r[0] is not None else 0.0 for r in results]
    df_final['R_Exam_Future'] = [r[1] if r[1] is not None else 0.0 for r in results]

    # 4. محاسبه تاریخ واقعی (برای وزن‌دهی پویا)
    if isinstance(min_date, str):
        base_date = pd.to_datetime(min_date)
    else:
        base_date = pd.to_datetime(min_date) if min_date else datetime.now()

    # تبدیل ستون 'date' به تاریخ واقعی
    df_final['real_date'] = base_date + pd.to_timedelta(df_final['date'], unit='D')
    # 🔥 محل جدید: افزودن ستون اکشن قبلی برای محاسبه جریمه تکرار 🔥
    # ----------------------------------------------------
    # .shift(1) اکشن سطر قبلی را می‌آورد. fillna(-1) برای سطر اول که اکشن قبلی ندارد.
    df_final['previous_action'] = df_final['actiontype'].shift(1).fillna(-1).astype(int)
    # 5. تعریف توابع وزن‌دهی پویا
    def get_dynamic_weights(row_date):
        """
        وزن‌دهی پویا بر اساس فاصله تا کنکور.

        λ_study(t) و λ_exam(t) با نزدیک‌شدن به کنکور افزایش می‌یابند.
        λ_comfort(t) صفر نمی‌شود، چون خواب/تغذیه/ورزش برای حفظ عملکرد ضروری‌اند،
        اما وزن آن نسبت به مطالعه و آزمون کمتر می‌شود.
        """

        konkur_date = pd.Timestamp(year=row_date.year, month=7, day=1)
        if row_date > konkur_date:
            konkur_date = pd.Timestamp(year=row_date.year + 1, month=7, day=1)

        days_left = max(0, int((konkur_date - row_date).days))

        # شروع تدریجی urgency از حدود یک سال قبل.
        # توان 1.20 باعث می‌شود رشد در ابتدا آرام و نزدیک کنکور قوی‌تر شود.
        urgency_factor = np.clip(1.0 - (days_left / 365.0), 0.0, 1.0)
        urgency_factor = float(urgency_factor ** 1.20)

        w_study = 1.40 + (4.60 * urgency_factor)
        w_exam = 1.00 + (3.50 * urgency_factor)
        w_comfort = 0.25 * (1.00 - 0.70 * urgency_factor)

        return w_study, w_exam, w_comfort, urgency_factor

    # ---------------------------------------------
    # 🔹 جریمهٔ تکرار براساس میانگین روزانه برای هر اکشن
    # ---------------------------------------------
    # فرض: df_final فقط برای یک کاربر است. اگر چند کاربر داری، باید user_id را هم در groupby اضافه کنی.
    
    # یک کلید روز بسازیم (اینجا همان ستون date کافی است؛ چون نسبی است)
    df_final['day_key'] = df_final['date']

    # برای هر روز و هر اکشن، شمارش تجمعی (تا این ردیف) بسازیم
    df_final['count_so_far_today'] = (
        df_final.groupby(['day_key', 'actiontype']).cumcount() + 1
    )

    # تعداد نهایی هر اکشن در هر روز
    daily_counts = (
        df_final
        .groupby(['day_key', 'actiontype'])
        .size()
        .reset_index(name='count_per_day')
    )

    # میانگین روزانه برای هر اکشن
    mean_daily_counts = (
        daily_counts
        .groupby('actiontype')['count_per_day']
        .mean()
        .to_dict()
    )

    # نگاشت میانگین به هر ردیف
    df_final['mean_daily_action_count'] = df_final['actiontype'].map(
        lambda a: float(mean_daily_counts.get(a, 1.0))  # اگر دیتای کافی نیست، حداقل ۱
    )

    # 6. تابع محاسبه پاداش نهایی (اصلاح شده با R_Base)
    def final_reward_calculation_dynamic(row):
        W_STUDY_DYN, W_EXAM_DYN, W_COMFORT_DYN, URGENCY = get_dynamic_weights(row['real_date'])
        
        # 🔑 استفاده از ستون جدید previous_action
        current_action = int(row['actiontype'])
        previous_action = int(row['previous_action'])
        logger.debug(f"Reward calc for action (RL_ID)={current_action}, date={row.get('date')}, R_Study_Future={row.get('R_Study_Future')}")
        
        # 🔥 بخش حیاتی: پاداش‌های پایه (Base Rewards) 🔥
        R_Base = 0.0
        
        if current_action == 1: R_Base = 0.2  # خواب (Sleep)
        if current_action == 2: R_Base = 0.3  # تغذیه (Nutrition)
        if current_action == 3: R_Base = 0.3  # ورزش (Sport)
        if current_action == 4: R_Base = 0.1  # سرگرمی (Entertainment)
        
        if current_action == 7: R_Base = 0.85 + (0.75 * URGENCY)  # مطالعه: پاداش پایه با نزدیکی کنکور افزایش می‌یابد
        if current_action == 8: R_Base = 0.05 * URGENCY  # آزمون: پاداش اصلی از نمره می‌آید؛ نزدیک کنکور کمی اهمیت پایه دارد
        # پاداش‌های متغیر
        R_Exam_Immediate = row['R_Exam_Score'] if current_action == 8 else 0.0
        R_Future = (W_EXAM_DYN * row['R_Exam_Future']) + (W_STUDY_DYN * row['R_Study_Future'])
        R_State_Change = (W_COMFORT_DYN * row['R_Satisfy_Change']) + (W_COMFORT_DYN * row['R_Stress_Change'])
        
        # جریمه‌ها
        R_Penalty = 0.0

        # جریمه کارهای غیردرسی تکراری (نزدیک کنکور) - (جریمه قبلی شما)
        if current_action not in [7, 8]: 
            if row['R_Study_Future'] < 0.1:
                penalty_multiplier = 2.0 if W_STUDY_DYN > 3.0 else 1.0
                R_Penalty += (-0.8 * penalty_multiplier)

        if current_action in [5, 6]:
            R_Penalty += -0.2  # جریمه جزئی برای کلاس فوق‌برنامه/سایر
        # ------------------------------------------------------------------
        # 🔥🔥🔥 جریمهٔ جدید تکرار براساس میانگین روزانه 🔥🔥🔥
        # ------------------------------------------------------------------
        # فعالیت مطالعه (۷) از این جریمه معاف است
        if current_action != 7:
            try:
                count_today = float(row.get('count_so_far_today', 1.0) or 1.0)
                mean_daily = float(row.get('mean_daily_action_count', 1.0) or 1.0)

                # تعداد تکرار اضافی نسبت به میانگین تاریخی
                excess_repetitions = max(0.0, count_today - mean_daily)

                if excess_repetitions > 0:
                    # هر تکرار اضافه، یک جریمهٔ خطی
                    REPETITION_PENALTY_PER_EXTRA = -0.15  # قابل تنظیم
                    penalty_repetition = REPETITION_PENALTY_PER_EXTRA * excess_repetitions
                    R_Penalty += penalty_repetition

                    logger.debug(
                        f"REWARD DEBUG: repetition penalty={penalty_repetition:.3f} "
                        f"for action={current_action}, count_today={count_today}, "
                        f"mean_daily={mean_daily:.2f}"
                    )
            except Exception as e:
                # اگر هر مشکلی بود، جریمه تکرار را نادیده بگیر
                logger.warning(f"Repetition penalty computation failed for idx={row.name}: {e}")
       

        # محاسبه نهایی
        raw_reward = R_Base + R_Exam_Immediate + R_Future + R_State_Change + R_Penalty
        
        if np.isnan(raw_reward):
            return 0.0
            
        return np.clip(raw_reward / max_reward_final_scale, -1.0, 1.0)

    # 7. اعمال تابع روی کل دیتافریم
    df_final['R_Final'] = df_final.apply(final_reward_calculation_dynamic, axis=1)

    # 8. خروجی نهایی
    ALL_COLS_FOR_ENV = STATE_FEATURES_COLS + [
        'R_Final', 'R_Exam_Future', 'R_Study_Future', 
        'R_Stress_Change', 'R_Satisfy_Change', 'R_Exam_Score',
    ]
    
    reward_analysis = df_final.groupby('actiontype')[['R_Final', 'R_Exam_Future', 'R_Study_Future', 'estres']].mean()
        
    return df_final[[col for col in ALL_COLS_FOR_ENV if col in df_final.columns]]
#===================================================================
# کش دیسکی برای داده‌های بزرگ
cache = dc.Cache('./tmp_cache')

@lru_cache(maxsize=1000)
def get_cached_dfstat(user_id):
    """کش در حافظه برای dfstat"""
    user = User.objects.get(id=user_id)
    return create_dfstat(user)

@lru_cache(maxsize=1000)  
def get_cached_dfexam(user_id):
    """کش در حافظه برای dfexam"""
    user = User.objects.get(id=user_id)
    return create_dfexam(user)
# تابع جدید برای تبدیل نوع فعالیت به عدد برای شبکه عصبی
  # این خط را حتماً در بالای فایل ایمپورت کنید

#=================انتخاب فضای فعالیت ==================
# این تابع را در فایل shoan.py جایگزین نسخه فعلی کنید.
@log_call
def get_user_action_space(user):
    """
    خروجی:
    1-6: فعالیت‌های ثابت (خواب، تغذیه، ...)
    7: مطالعه
    8: آزمون
    9+: فعالیت‌های اختصاصی کاربر
    """
    idx_to_act = {}
    act_to_idx = {}
    rl_id_map = {} # RL_ID -> DB_ID
    current_index = 0
    
    # 1. فعالیت‌های ثابت (1 تا 6)
    fixed_actions = {
        1: "خواب",
        2: "تغذیه",
        3: "ورزش", 
        4: " سرگرمی", 
        5: "کلاس آموزشی یا فوق برنامه", 
        6: "سایر"
    }
    
    for rl_id in range(1, 7):
        # نگاشت مستقیم: RL ID = DB ID
        rl_id_map[rl_id] = rl_id
        idx_to_act[current_index] = rl_id
        act_to_idx[rl_id] = current_index
        current_index += 1

    # 2. مطالعه (RL ID = 7)
    # این فعالیت در جدول Faliat نیست، پس مقدار map را یک شناسه خاص می‌گذاریم
    rl_id_map[7] = '__STUDY__'
    idx_to_act[current_index] = 7
    act_to_idx[7] = current_index
    current_index += 1

    # 3. آزمون (RL ID = 8)
    rl_id_map[8] = '__EXAM__'
    idx_to_act[current_index] = 8
    act_to_idx[8] = current_index
    current_index += 1

    # 4. فعالیت‌های کاربر (شروع از RL ID = 9)
    # فعالیت‌هایی که ID دیتابیس آنها > 6 است
    user_custom_acts = type_Faliat_model.objects.filter(user=user).exclude(id__lte=6).order_by('id')
    
    next_rl_id = 9
    for act in user_custom_acts:
        db_id = int(act.id)
        rl_id_map[next_rl_id] = db_id
        
        idx_to_act[current_index] = next_rl_id
        act_to_idx[next_rl_id] = current_index
        current_index += 1
        next_rl_id += 1

    action_dim = current_index
    
    logger.debug(f"New Action Space: {rl_id_map}")
    return action_dim, idx_to_act, act_to_idx, rl_id_map

#=====================================================================================
@log_call
def randomstate1(dfstat):
        '''تابعی برای خواندن تصادفی یک ردیف ردیف از دیتاست استیت'''
        logger.info("Starting randomstate1")
        try:
            data =dfstat 
            # انتخاب چهار سطر تصادفی
            num_rows_to_sample = 1
            random_rows = data.sample(n=num_rows_to_sample)
            random_rows=random_rows.fillna(0)
            random_rows = random_rows.to_numpy(dtype=np.float32)
            # گرد کردن به دو رقم اعشار
            #random_rows = np.round(random_rows, 2)  
            random_rows = random_rows[:batch_size]
            random_rows = tf.reshape(random_rows, (1, -1)) # 1 shode batch_size
            logger.info("end of random state1 ")
            return random_rows
        except Exception as e:
            logger.info(f"Error in runalgoritm: {str(e)}")
            raise

#================================================================== 

@log_call
def get_previous_scoresac(date1, book_name, start_page):
        dfexam=create_dfexam(user=User.objects.get(id=CURRENT_USER_ID))
        #تابعی برای پیدا کردن نمره قبلی
        try:
            logger.debug("start get_privios_score")
            dfp = dfexam 
            dfp = dfp.fillna(0)
            try:
                dfp['date'] = dfp['date'].astype(int)
            except ValueError:
                dfp['date'] = pd.to_numeric(dfp['date'], errors='coerce').fillna(0).astype(int)
            # تبدیل نوع داده‌ها به عدد صحیح در صورت نیاز
            dfp['bookname'] = dfp['bookname'].astype(str)
            dfp['startpage'] = dfp['startpage'].astype(str)
            dfp['date'] = dfp['date'].astype(int)

            # تبدیل date1 به عدد در صورت لزوم
            if isinstance(date1, tf.Tensor):
                date1 = date1.numpy()  # تبدیل EagerTensor به عدد

        # فیلتر کردن داده‌ها بر اساس کتاب، صفحات و تاریخ
            filtered_data = dfp[
                (dfp['bookname'] == book_name) &
                (dfp['startpage'] == start_page) &
                (dfp['date'] < date1)  # فقط تاریخ‌های قبلی
                ]
    
        # اگر داده‌ای وجود داشته باشد، نمره قبلی را برمی‌گرداند
            if not filtered_data.empty:
                previous_score = filtered_data['score'].values[-1]  # آخرین نمره
                last_study_date = filtered_data['date'].values[-1]  # تاریخ آخرین مطالعه
                days_difference = date1 - last_study_date  # محاسبه فاصله روزها
            else:
                previous_score=0 # در صورتی که نمره‌ای پیدا نشود
                days_difference=0

            logger.info("end get previous_score")
            return previous_score, days_difference  # نمره قبلی را برمی‌گرداند
        except Exception as e:
            logger.exception((f"Error: Failed to create dfstat. Reason: {str(e)}"))
            return 0.0, 0.0
#====================================================================================
from collections import deque
import random
@log_methods
class ReplayBuffer:
    def __init__(self, dfstat, dfexam, buffer_size=5000, batch_size=32):
        self.dfstat = dfstat
        self.dfexam = dfexam
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)  # استفاده از deque برای کارایی بهتر
        self.position = 0
        
    def add(self, state, action, reward, next_state):
        logger.debug(f"ReplayBuffer.add: storing action index: {action}")
        """افزودن یک نمونه به بافر"""
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.position] = (state, action, reward, next_state)
            self.position = (self.position + 1) % self.buffer_size

    def add_batch(self, states, actions, rewards, next_states):
        """افزودن دسته‌ای از نمونه‌ها به بافر"""
        for s, a, r, ns in zip(states, actions, rewards, next_states):
            self.add(s, a, r, ns)

    def sample(self):
        """نمونه‌برداری از بافر - نسخه تصحیح شده"""
        batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        states, actions, rewards, next_states = zip(*batch)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # تغییر به int32
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        return states, actions, rewards, next_states
    def __len__(self):
        return len(self.buffer)
#============================================================================
# کلاس اکتر
class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, name="Actor"):
        super().__init__(name=name)

        # محاسبه n_features (کد شما عالی بود، فقط کمی ساده‌تر شد)
        if isinstance(state_dim, (tuple, list, np.ndarray)):
            self.n_features = int(state_dim[-1])
        else:
            self.n_features = int(state_dim)

        self.action_dim = int(action_dim)

        # لایه‌ها
        self.fc1 = layers.Dense(units=128, kernel_initializer='he_normal', name="fc1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, name="bn1")
        self.fc2 = layers.Dense(units=128, kernel_initializer='he_normal', name="fc2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, name="bn2")
        self.logits_layer = layers.Dense(self.action_dim, kernel_initializer='glorot_normal', name="logits")

        # پیاده‌سازی داخلی
        def _impl(state, training):
            # ترافلو خودش در اینجا state را به صورت (Batch, Features) تحویل می‌دهد
            x = self.fc1(state)
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)

            x = self.fc2(x)
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)

            logits = self.logits_layer(x)
            probs = tf.nn.softmax(logits, axis=-1)
            # استفاده از epsilon برای جلوگیری از log(0)
            log_probs = tf.math.log(probs + 1e-10)
            return probs, log_probs, logits

        self._call_impl = _impl

        # تعریف امضا به صورت پویا بر اساس n_features محاسبه شده
        #input_sig = [tf.TensorSpec(shape=[None, self.n_features], dtype=tf.float32)]
        input_sig = [tf.TensorSpec(shape=[None, self.n_features], dtype=tf.float32)]
        
        self._call_train = tf.function(lambda s: self._call_impl(s, True),
                                       input_signature=input_sig)
        self._call_infer = tf.function(lambda s: self._call_impl(s, False),
                                       input_signature=input_sig)

    def call(self, state, training=False):
        # تبدیل ورودی به تنسور و اطمینان از 2 بعدی بودن (Batch, Features)
        if not tf.is_tensor(state):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
        
        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)

        if training:
            return self._call_train(state)
        return self._call_infer(state)

#============================================================================
# کلاس کریتیک
class Critic(tf.keras.Model):
    def __init__(self, state_dim, num_columns, num_action_types, name="Critic"):
        super().__init__(name=name)
        
        # --- اصلاح بخش مقداردهی (بسیار مهم) ---
        # ما باید دقیقاً بدانیم ورودی چند ستون دارد (16 ویژگی ثابت + تعداد فعالیت‌ها)
        # بهترین راه استفاده از مقدار عددی مستقیم است که از runalgoritm می‌آید
        
        if isinstance(num_columns, int):
            self.num_columns = num_columns
        elif isinstance(state_dim, (tuple, list)):
            self.num_columns = state_dim[-1]
        else:
            self.num_columns = int(num_columns)

        self.num_action_types = int(num_action_types)
        
        # --- لایه‌ها ---
        # Q1
        self.q1_fc1 = layers.Dense(128, kernel_initializer='he_normal')
        self.q1_bn1 = layers.BatchNormalization(momentum=0.9)
        self.q1_fc2 = layers.Dense(128, kernel_initializer='he_normal')
        self.q1_bn2 = layers.BatchNormalization(momentum=0.9)
        self.q1_output = layers.Dense(self.num_action_types, name="q1_output")

        # Q2
        self.q2_fc1 = layers.Dense(128, kernel_initializer='he_normal')
        self.q2_bn1 = layers.BatchNormalization(momentum=0.9)
        self.q2_fc2 = layers.Dense(128, kernel_initializer='he_normal')
        self.q2_bn2 = layers.BatchNormalization(momentum=0.9)
        self.q2_output = layers.Dense(self.num_action_types, name="q2_output")

        # --- تعریف امضای ورودی (فقط یکبار و با دقت) ---
        # استفاده از self.num_columns تضمین می‌کند که اگر کاربر فعالیت اضافه کرد، امضا تغییر کند
        input_sig = [tf.TensorSpec(shape=[None, self.num_columns], dtype=tf.float32)]

        self._call_train = tf.function(
            lambda state: self._call_impl(state, True),
            input_signature=input_sig
        )
        self._call_infer = tf.function(
            lambda state: self._call_impl(state, False),
            input_signature=input_sig
        )

    def _call_impl(self, state, training):
        # فرض بر این است که state یک tf.Tensor با dtype float32 و شکل [batch, num_columns] است
        # Q1
        x1 = self.q1_fc1(state)
        x1 = self.q1_bn1(x1, training=training)
        x1 = tf.nn.relu(x1)
        x1 = self.q1_fc2(x1)
        x1 = self.q1_bn2(x1, training=training)
        x1 = tf.nn.relu(x1)
        q1_values = self.q1_output(x1)

        # Q2
        x2 = self.q2_fc1(state)
        x2 = self.q2_bn1(x2, training=training)
        x2 = tf.nn.relu(x2)
        x2 = self.q2_fc2(x2)
        x2 = self.q2_bn2(x2, training=training)
        x2 = tf.nn.relu(x2)
        q2_values = self.q2_output(x2)

        return q1_values, q2_values

    def call(self, state, training=False):
        """
        متد عمومی؛ برای فراخوانی عمومی (خارج از tf.function) آن را صدا بزن.
        اما در بخش‌های داخل @tf.function (مثل train_step) بهتر است مستقیم
        از self._call_train یا self._call_infer استفاده شود (موارد پایین).
        """
        # normalize input to batch shape
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        if state.shape.ndims == 1:
            state = tf.expand_dims(state, axis=0)

        # اگر training به صورت Python bool فرستاده شده، مستقیماً تابع کامپایل‌شدهٔ مربوطه را فراخوانی کن
        if bool(training):
            return self._call_train(state)
        else:
            return self._call_infer(state)

#================================================================
# کلاس محیط 
@log_methods
class StudyEnvironment:
    def __init__(self, df_final_data, df_stat_features, df_exam_features, min_date, target_action_dim=None): # <--- پارامتر اضافه شد تارگت اکشن دیم بلا استفاده است
        if CURRENT_USER_ID is None:
            raise ValueError("CURRENT_USER_ID باید قبل از ایجاد Actor تنظیم شود!")
            
        self.user = User.objects.get(id=CURRENT_USER_ID)
        self.dfstat = df_stat_features
        self.dfexam = df_exam_features 
        self.column_names = column_names # مطمئن شوید column_names ایمپورت شده باشد
        self.current_step = 0
        self.current_time = 0.0      
        self.max_steps = 50          
        self.daily_schedule = []     
        self.data = df_final_data.to_dict('records') 
        
        if not self.data:
            raise ValueError("self.data خالی است. داده‌های کاربر را چک کنید.")
            
        self.current_index = 0
        self.current_state = None  

        # 1. اول داده‌ها را آماده می‌کنیم
        self.df_final = prepare_rl_data(
            self.dfstat, self.dfexam, self.user, min_date,
            rl_id_to_db_id_map=None  
        )

        # 2. حالا اکشن‌اسپیس را می‌سازیم (فقط یک بار صدا می‌زنیم)
        # این تابع خودکار self.n_actions را مقداردهی می‌کند
        self._init_action_space()
        self.state_dim = 16 + self.n_actions * 2
        
        # 3. حالا که n_actions مشخص شد، آرایه شمارنده را می‌سازیم
        # (کدهای تکراری قبلی حذف شدند)
        self.counts_today = np.zeros(self.n_actions, dtype=np.float32)

        # 4. تنظیم وضعیت پایه و پاداش‌ها
        self.base_state16 = None
        
        utility_by_actiontype = (
            self.df_final
            .groupby('actiontype')['R_Final']
            .mean()
            .to_dict()
        )

        self.action_utility_map = {
            self.actiontype_to_index[int(a)]: float(r)
            for a, r in utility_by_actiontype.items()
            if int(a) in self.actiontype_to_index
        }
        
        daily_counts = (
            self.df_final
            .groupby(['date', 'actiontype'])
            .size()
            .reset_index(name='count_per_day')
        )

        mean_daily_counts_by_actiontype = (
            daily_counts
            .groupby('actiontype')['count_per_day']
            .mean()
            .to_dict()
        )

        self.action_limits_map = {
            self.actiontype_to_index[int(a)]: float(limit)
            for a, limit in mean_daily_counts_by_actiontype.items()
            if int(a) in self.actiontype_to_index
        }
        # نگاشت action_index → mean R_Final
        self.action_rl_reward_map = (
            self.df_final
            .groupby('actiontype')['R_Final']
            .mean()
            .rename(lambda a: self.actiontype_to_index[a])
            .to_dict()
        )
    def _init_action_space(self):
        actiontypes = sorted(
            set(int(x) for x in self.df_final['actiontype'].dropna().unique())
        )

        if not actiontypes:
            raise ValueError("no data found in action type")

        self.index_to_actiontype = {i: a for i, a in enumerate(actiontypes)}
        self.actiontype_to_index = {a: i for i, a in self.index_to_actiontype.items()}
        self.n_actions = len(actiontypes)
        
    def _row_to_state16(self, row: dict):
        """
        یک ردیف activity را به بردار 16تایی float32 تبدیل می‌کند.
        اگر کلیدی نبود، 0 می‌گذارد تا کد نشکند.
        """
        def g(key, default=0.0):
            v = row.get(key, default)
            return 0.0 if v is None else float(v)

        # نکته مهم: actiontype در state، بهتر است "action_index" باشد نه actiontype خام
        # چون مدل RL با index کار می‌کند و این باعث سازگاری می‌شود.
        raw_actiontype = row.get('actiontype', None)
        if raw_actiontype is None:
            action_index = 0.0
        else:
            action_index = float(self.actiontype_to_index.get(int(raw_actiontype), 0))

        state_features = [
            g('date', 0.0),                 # 1
            action_index,                   # 2  (به جای actiontype خام)
            g('bookname', 0.0),             # 3
            g('startpage', 0.0),            # 4
            g('endpage', 0.0),              # 5
            g('starttime', 0.0),            # 6
            g('duringtime', 0.0),           # 7
            g('typestudy', 0.0),            # 8
            g('oxijen', 0.0) / 100.0,       # 9
            g('estres', 0.0) / 100.0,       # 10
            g('dificulty', 0.0),            # 11
            g('repitnumber', 0.0),          # 12
            g('dutimerev', 0.0),            # 13
            g('typenutrition', 0.0),        # 14
            g('typefood', 0.0),             # 15
            g('typeentertinment', 0.0),     # 16
        ]
        return np.array(state_features, dtype=np.float32)


    def _build_state16_from_action(self, action_index: int):
        """
        state16 را بر اساس اکشن انتخابی می‌سازد (آخرین فعالیت انجام‌شده).
        برای ویژگی‌هایی که به آن اکشن مربوط نیستند، صفر می‌گذارد.
        این همان منطق 'بعضی ویژگی‌ها برای بعضی فعالیت‌ها فعال‌اند' را رعایت می‌کند.
        """
        # اینجا باید نگاشت منطقی اکشن‌ها به ویژگی‌ها را تعیین کنی.
        # چون کد کامل پروژه‌ات را ندارم، یک پیاده‌سازی اصولی/قابل‌گسترش می‌دهم:

        # یک اسکلت پایه:
        s = np.zeros((16,), dtype=np.float32)

        # 1) زمان/روز
        s[0] = float(self.current_step)    # یا date واقعی/نرمال‌شده اگر داری
        s[1] = float(action_index)         # action_index

        # starttime را از current_time بساز (0..1). اگر می‌خواهی دقیقه باشد تبدیل کن.
        s[5] = float(self.current_time)

        # یک duringtime پیش‌فرض (مثلاً یک گام)
        s[6] = 1.0 / float(self.max_steps)

        # حالا بر اساس نوع اکشن، بعضی فیلدها را باز کن:
        # فرض: اکشن مطالعه = همان اکشنی که actiontype==7 بوده (ولی اینجا index داریم)
        # پس باید تشخیصش را از actiontype اصلی بگیریم:
        actiontype = self.index_to_actiontype[action_index]

        if actiontype == 7:
            # مطالعه: typestudy / dificulty / repitnumber / dutimerev ممکن است معنی داشته باشند
            # اگر اینها را از مدل/کاربر در لحظه می‌گیری، همینجا ست کن.
            # فعلاً صفر می‌مانند مگر داده آنلاین داری.
            pass

        if actiontype == 8:
            # آزمون: شاید bookname یا سطح سختی/استرس معنی داشته باشد
            pass

        # فعالیت‌های تغذیه/سرگرمی/...:
        # اینجا هم می‌توانی براساس actiontype برخی فیلدها را فعال کنی.

        return s


    def _get_base_features(self):
        """
        وضعیت پایه 16تایی را برمی‌گرداند.
        این همان چیزی است که get_state نیاز دارد.
        """
        if self.base_state16 is None:
            # اگر reset هنوز صدا نشده یا مقداردهی نشده
            return np.zeros((16,), dtype=np.float32)
        return self.base_state16

    #=======================تابع ریست =============================
    def reset(self):
        self.current_time = 0.0
        self.current_step = 0
        self.counts_today = np.zeros(self.n_actions, dtype=np.float32)

        # state پایه در شروع روز: "هیچ فعالیتی هنوز انجام نشده"
        self.base_state16 = np.zeros((16,), dtype=np.float32)

        # state کامل = base + counts
        return self.get_state()

    #====================== پایان تابع ریست ============================
    
    def create_state_vector(self, row):
        # 🌟 هدف: بازگرداندن یک بردار با 16 ویژگی حالت 🌟
        state_features = [
            # ۱. تاریخ و زمان (۱۶ ویژگی اصلی از dfstat)
            row['date'],                   # ۱: روز از شروع (نرمالایزشده یا خام)
            row['actiontype'],             # ۲: نوع فعالیت (مطالعه 7، آزمون 8، فعالیت ۳ به بعد)
            row['bookname'],               # ۳: کد کتاب
            row['startpage'],              # ۴: صفحه شروع (نرمالایزشده)
            row['endpage'],                # ۵: صفحه پایان (نرمالایزشده)
            row['starttime'],              # ۶: زمان شروع (نرمالایزشده)
            row['duringtime'],             # ۷: طول زمان فعالیت (نرمالایزشده)
            row['typestudy'],              # ۸: نوع مطالعه (در صورت مطالعه)
            row['oxijen'] / 100.0,         # ۹: میزان رضایت (نرمالایزشده)
            row['estres'] / 100.0,         # ۱۰: میزان استرس (نرمالایزشده)
            row['dificulty'],              # ۱۱: سختی (در صورت مطالعه)
            row['repitnumber'],            # ۱۲: تعداد تکرار
            row['dutimerev'],              # ۱۳: زمان لازم برای مرور
            row['typenutrition'],          # ۱۴: نوع تغذیه
            row['typefood'],               # ۱۵: نوع غذا
            row['typeentertinment'],       # ۱۶: نوع سرگرمی 
        ]
    
        # تبدیل به آرایه NumPy و تغییر شکل (Reshape) برای سازگاری با Keras/TensorFlow
        # (۱، 16)
        return np.array([state_features], dtype=np.float32)
    #=======================تابع گام یا استپ =============================
    def step(self, action_index):
        action_index = int(action_index)

        if action_index not in self.index_to_actiontype:
            logger.error(
                f"Invalid action_index={action_index} for user={CURRENT_USER_ID}. "
                f"Valid indices={list(self.index_to_actiontype.keys())}"
            )
            return self.get_state(), 0.0, True, {}

        actiontype = self.index_to_actiontype[action_index]

        base_reward = float(self.action_rl_reward_map.get(action_index, 0.0))

        self.counts_today[action_index] += 1
        count = self.counts_today[action_index]

        repetition_penalty = 0.0
        if actiontype != 7:
            repetition_penalty = -1.5 * (count ** 1.3)

        reward = base_reward + repetition_penalty
        reward = np.clip(reward, -2.0, 2.0)

        self.last_action_index = action_index
        self.current_step += 1
        self.current_time += (1 / self.max_steps)

        done = self.current_step >= self.max_steps

        return self.get_state(), reward, done, {
            "action": actiontype,
            "count": count,
            "penalty": repetition_penalty
        }

    #==========پایان تابع گام یا استپ ======================
    def get_state(self):
        """
        ساخت state کامل محیط:
        [16 ویژگی پایه] + [n_actions شمارنده نرمال] + [n_actions فلگ تکرار]
        """

        # 1️⃣ همیشه ویژگی‌های پایه را بساز
        if self.current_index < len(self.data):
            row = self.data[self.current_index]
        else:
            # اگر از دیتا جلو زدیم، آخرین ردیف را نگه دار
            row = self.data[-1]

        base_features = self.create_state_vector(row).flatten()  # (16,)

        # 2️⃣ شمارنده‌ها
        normalized_counts = self.counts_today / self.max_steps   # (n_actions,)

        # 3️⃣ فلگ تکرار (مثال: بیشتر از 1 بار)
        repetition_flags = np.array(
            [1.0 if c > 1 else 0.0 for c in self.counts_today],
            dtype=np.float32
        )  # (n_actions,)

        # 4️⃣ اتصال همه‌چیز
        full_state = np.concatenate([
            base_features,
            normalized_counts,
            repetition_flags
        ])

        # 5️⃣ چک ایمنی (خیلی مهم)
        assert full_state.shape[0] == self.state_dim, \
            f"state_dim mismatch: got {full_state.shape[0]}, expected {self.state_dim}"

        return full_state.reshape(1, -1).astype(np.float32)


#=======================پایان کلاس محیط =============================

    # استفاده از کلاس
if __name__ == "__main__":
    # --- ابتدا پاداش را محاسبه کنید ---
    # ۱. محاسبه min_date_of_all_data (برای رفع خطای Pylance)
    try:
        min_study_date = study_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
        min_activity_date = Faliat_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
        min_exam_date = azmoon_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
        valid_dates = list(filter(None, [min_study_date, min_activity_date, min_exam_date]))
        min_date_of_all_data = min(valid_dates) if valid_dates else None
    except Exception:
        min_date_of_all_data = None
        
    # ۲. ساخت DataFrameهای اولیه (ویژگی‌های حالت و آزمون)
    # فرض بر این است که این توابع داده‌ها را از دیتابیس واکشی می‌کنند
    df_stat_features = create_dfstat(user=user) 
    df_exam_file = create_dfexam(user=user) 
    
    # --- ابتدا پاداش را محاسبه کنید ---
    FINAL_DF = prepare_rl_data(
        df_stat=df_stat_features,       # <--- اضافه شدن آرگومان df_stat
        df_exam=df_exam_file,           # <--- اضافه شدن آرگومان df_exam
        user=user, 
        min_date=min_date_of_all_data, 
        max_reward_exam=1500.0, 
        max_reward_final_scale=1.0 
    )
    min_study_date = study_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
    min_activity_date = Faliat_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
    min_exam_date = azmoon_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
    valid_dates = list(filter(None, [min_study_date, min_activity_date, min_exam_date]))
    min_date_of_all_data = min(valid_dates) if valid_dates else None

    # --- سپس محیط را با داده‌های آماده شده بسازید ---
    study_env = StudyEnvironment(
        df_final_data=FINAL_DF,         
        df_stat_features=df_stat_features,
        df_exam_features=df_exam_file,
        min_date=min_date_of_all_data
    )
    # study_env.run() دیگر نیاز نیست چون بارگذاری داده در prepare_rl_data انجام شده است
#================================================================
#کلاس ترتیب مطالعه 
@log_methods
class StudyPriority:
    """
    v27 coverage-balanced and review-aware study selector.

    The selector operates on the network/book indices stored in df_final.
    It builds a page-level shadow state once per planning day and updates it
    immediately after every proposed study block. This prevents exact
    same-day duplicate ranges and starvation of books at the end of the list.

    Ranking principles:
    1. Unread pages before extra repetition.
    2. Fair rotation among books using time since last selection/study.
    3. Spaced reviews at 1, 7, 30, and 120 days.
    4. When no review is due, choose the least-exposed/oldest block.
    """

    STUDY_ACTIONTYPE = 7
    REVIEW_INTERVALS = (1.0, 7.0, 30.0, 120.0)
    BLOCK_PAGES = 10

    def __init__(self, dataset_file, book_info_dict, planning_day=None):
        self.dataset = (
            dataset_file.copy()
            if dataset_file is not None
            else pd.DataFrame()
        )
        self.book_info = {
            int(k): max(1, int(v))
            for k, v in (book_info_dict or {}).items()
        }

        if planning_day is None:
            if (
                not self.dataset.empty
                and "date" in self.dataset.columns
            ):
                dates = pd.to_numeric(
                    self.dataset["date"],
                    errors="coerce",
                ).dropna()
                planning_day = (
                    float(dates.max()) + 1.0
                    if not dates.empty
                    else 0.0
                )
            else:
                planning_day = 0.0

        self.planning_day = float(planning_day)
        self.planned_today = set()
        self.planned_book_counts = {
            int(book_id): 0
            for book_id in self.book_info
        }
        self.page_state = {}
        self._build_page_state()

    def _to_real_page(self, book_id, page_value):
        try:
            bid = int(book_id)
            total_pages = self.book_info.get(bid, 100)
            page_value = float(page_value)

            if not np.isfinite(page_value) or page_value <= 0:
                return 1

            if 0 < page_value <= 1:
                page = int(round(page_value * total_pages))
            else:
                page = int(round(page_value))

            return int(np.clip(page, 1, total_pages))
        except Exception:
            return 1

    def _empty_book_state(self, total_pages):
        return {
            "date_sets": [set() for _ in range(total_pages)],
            "last_day": np.full(total_pages, np.nan, dtype=float),
        }

    def _add_observation(
        self,
        book_id,
        start_page,
        end_page,
        day_value,
        real_pages=False,
    ):
        try:
            bid = int(book_id)
        except Exception:
            return

        if bid not in self.book_info:
            return

        total_pages = self.book_info[bid]

        if real_pages:
            start_page = int(np.clip(
                int(round(float(start_page))),
                1,
                total_pages,
            ))
            end_page = int(np.clip(
                int(round(float(end_page))),
                1,
                total_pages,
            ))
        else:
            start_page = self._to_real_page(bid, start_page)
            end_page = self._to_real_page(bid, end_page)

        if end_page < start_page:
            start_page, end_page = end_page, start_page

        try:
            day_value = float(day_value)
        except Exception:
            day_value = self.planning_day

        if not np.isfinite(day_value):
            day_value = self.planning_day

        state = self.page_state.setdefault(
            bid,
            self._empty_book_state(total_pages),
        )

        left = max(1, start_page) - 1
        right = min(total_pages, end_page)

        for page_index in range(left, right):
            state["date_sets"][page_index].add(day_value)
            previous = state["last_day"][page_index]
            if np.isnan(previous) or day_value > previous:
                state["last_day"][page_index] = day_value

    def _build_page_state(self):
        for bid, total_pages in self.book_info.items():
            self.page_state[bid] = self._empty_book_state(total_pages)

        if self.dataset.empty:
            return

        required = {"actiontype", "bookname", "startpage", "endpage", "date"}
        if not required.issubset(self.dataset.columns):
            logger.warning(
                "v27 StudyPriority: df_final is missing columns: %s",
                sorted(required - set(self.dataset.columns)),
            )
            return

        studies = self.dataset[
            pd.to_numeric(
                self.dataset["actiontype"],
                errors="coerce",
            ).eq(self.STUDY_ACTIONTYPE)
        ]

        for row in studies.itertuples(index=False):
            try:
                self._add_observation(
                    getattr(row, "bookname"),
                    getattr(row, "startpage"),
                    getattr(row, "endpage"),
                    getattr(row, "date"),
                )
            except Exception:
                continue

    def _book_last_day(self, book_id):
        last_values = self.page_state[book_id]["last_day"]
        finite = last_values[np.isfinite(last_values)]
        if finite.size == 0:
            return -1.0e9
        return float(np.max(finite))

    def _candidate_blocks_for_book(self, book_id):
        total_pages = self.book_info[book_id]
        state = self.page_state[book_id]
        date_sets = state["date_sets"]
        last_day = state["last_day"]
        candidates = []

        for start_page in range(1, total_pages + 1, self.BLOCK_PAGES):
            end_page = min(
                total_pages,
                start_page + self.BLOCK_PAGES - 1,
            )
            range_key = (book_id, start_page, end_page)

            if range_key in self.planned_today:
                continue

            left = start_page - 1
            right = end_page

            exposure_counts = np.asarray(
                [len(date_sets[i]) for i in range(left, right)],
                dtype=float,
            )
            block_last = last_day[left:right]
            finite_last = block_last[np.isfinite(block_last)]

            minimum_exposure = int(np.min(exposure_counts))
            mean_exposure = float(np.mean(exposure_counts))
            unread_pages = int(np.sum(exposure_counts == 0))

            if finite_last.size == 0:
                last_studied = -1.0e9
                days_since = 1.0e9
            else:
                # The most recently studied page controls whether the whole
                # block is ready for another spaced review.
                last_studied = float(np.max(finite_last))
                days_since = max(
                    0.0,
                    self.planning_day - last_studied,
                )

            if minimum_exposure <= 0:
                stage = 0
                overdue = 1.0e9
            else:
                interval_index = min(
                    minimum_exposure - 1,
                    len(self.REVIEW_INTERVALS) - 1,
                )
                required_interval = self.REVIEW_INTERVALS[interval_index]
                overdue = days_since - required_interval
                stage = 1 if overdue >= 0 else 2

            candidates.append(
                {
                    "book_id": int(book_id),
                    "start_page": int(start_page),
                    "end_page": int(end_page),
                    "stage": int(stage),
                    "unread_pages": int(unread_pages),
                    "minimum_exposure": int(minimum_exposure),
                    "mean_exposure": float(mean_exposure),
                    "last_studied": float(last_studied),
                    "days_since": float(days_since),
                    "overdue": float(overdue),
                }
            )

        return candidates

    def _best_block_for_book(self, book_id):
        candidates = self._candidate_blocks_for_book(book_id)
        if not candidates:
            return None

        def block_key(candidate):
            if candidate["stage"] == 0:
                return (
                    0,
                    -candidate["unread_pages"],
                    candidate["mean_exposure"],
                    candidate["start_page"],
                )

            if candidate["stage"] == 1:
                return (
                    1,
                    -candidate["overdue"],
                    candidate["minimum_exposure"],
                    candidate["mean_exposure"],
                    candidate["start_page"],
                )

            return (
                2,
                candidate["minimum_exposure"],
                candidate["mean_exposure"],
                candidate["last_studied"],
                candidate["start_page"],
            )

        return min(candidates, key=block_key)

    def suggest_study(self):
        candidates = []

        for book_id in sorted(self.book_info):
            candidate = self._best_block_for_book(book_id)
            if candidate is None:
                continue

            book_last_day = self._book_last_day(book_id)
            book_starvation_days = max(
                0.0,
                self.planning_day - book_last_day,
            )

            candidate["book_starvation_days"] = book_starvation_days
            candidate["planned_book_count_today"] = int(
                self.planned_book_counts.get(book_id, 0)
            )
            candidates.append(candidate)

        def candidate_key(candidate):
            # Unread pages dominate. Inside each stage, rotate books that
            # have been ignored for longer before returning to a recent book.
            if candidate["stage"] == 0:
                return (
                    0,
                    candidate["planned_book_count_today"],
                    -candidate["book_starvation_days"],
                    -candidate["unread_pages"],
                    candidate["mean_exposure"],
                    candidate["book_id"],
                )

            if candidate["stage"] == 1:
                return (
                    1,
                    candidate["planned_book_count_today"],
                    -candidate["overdue"],
                    -candidate["book_starvation_days"],
                    candidate["minimum_exposure"],
                    candidate["book_id"],
                )

            return (
                2,
                candidate["planned_book_count_today"],
                candidate["minimum_exposure"],
                -candidate["book_starvation_days"],
                candidate["mean_exposure"],
                candidate["book_id"],
            )

        candidates.sort(key=candidate_key)

        return [
            (
                candidate["book_id"],
                (
                    candidate["start_page"],
                    candidate["end_page"],
                ),
            )
            for candidate in candidates
        ]

    def suggest_next(self):
        suggestions = self.suggest_study()
        return suggestions[0] if suggestions else None

    def register_planned(self, book_id, start_page, end_page):
        bid = int(book_id)
        total_pages = self.book_info[bid]
        start_page = int(np.clip(
            int(round(float(start_page))),
            1,
            total_pages,
        ))
        end_page = int(np.clip(
            int(round(float(end_page))),
            1,
            total_pages,
        ))

        if end_page < start_page:
            start_page, end_page = end_page, start_page

        key = (bid, start_page, end_page)
        if key in self.planned_today:
            raise ValueError(
                "v27 duplicate study range in the same planning day: "
                f"book={bid}, pages={start_page}-{end_page}"
            )

        self.planned_today.add(key)
        self.planned_book_counts[bid] = (
            self.planned_book_counts.get(bid, 0) + 1
        )
        self._add_observation(
            bid,
            start_page,
            end_page,
            self.planning_day,
            real_pages=True,
        )




def _v2721_quantize_study_duration(
    requested_hours,
    remaining_daily_cap_hours,
    block_cap_hours,
    remaining_clock_hours,
):
    """
    Convert a study duration to a cap-safe whole-minute representation.

    v27.2 accumulated floating-point hours internally but serialized each block
    with decimal_to_time(), which rounded every block independently to the
    nearest minute. Across several blocks, those per-block roundings could make
    the serialized total exceed the daily cap even when the internal total was
    exactly within the cap. v27.2.1 uses one whole-minute value for scheduling,
    accumulation, invariant checking, persistence, and simulation.

    The function always rounds DOWN at the cap boundary. It therefore never
    turns an allowed total of 10.000 hours into 10.033 hours after serialization.
    """
    values = {
        "requested_hours": requested_hours,
        "remaining_daily_cap_hours": remaining_daily_cap_hours,
        "block_cap_hours": block_cap_hours,
        "remaining_clock_hours": remaining_clock_hours,
    }

    normalized = {}
    for name, value in values.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite: {value!r}")
        normalized[name] = max(0.0, numeric)

    epsilon_minutes = 1.0e-7

    requested_minutes = max(
        30,
        int(math.floor(
            normalized["requested_hours"] * 60.0 + epsilon_minutes
        )),
    )
    daily_cap_minutes = int(math.floor(
        normalized["remaining_daily_cap_hours"] * 60.0 + epsilon_minutes
    ))
    block_cap_minutes = int(math.floor(
        normalized["block_cap_hours"] * 60.0 + epsilon_minutes
    ))
    clock_minutes = int(math.floor(
        normalized["remaining_clock_hours"] * 60.0 + epsilon_minutes
    ))

    allowed_minutes = min(
        requested_minutes,
        daily_cap_minutes,
        block_cap_minutes,
        clock_minutes,
    )

    if allowed_minutes < 30:
        raise RuntimeError(
            "v27.2.1 study duration quantization left less than the "
            f"30-minute minimum block: requested={requested_minutes}m, "
            f"daily_cap={daily_cap_minutes}m, "
            f"block_cap={block_cap_minutes}m, "
            f"clock={clock_minutes}m"
        )

    duration_hours = allowed_minutes / 60.0
    duration_day_fraction = duration_hours / 24.0
    duration_text = f"{allowed_minutes // 60}:{allowed_minutes % 60:02d}"

    return {
        "minutes": allowed_minutes,
        "hours": duration_hours,
        "day_fraction": duration_day_fraction,
        "text": duration_text,
    }

#===============================================================
@log_methods
class SAC:
    def __init__(self, state_dim, action_dim, max_action, num_columns, study_env, action_maps, sac_index_to_db_id, rl_id_to_db_id_map=None):
        if study_env is None:
            raise ValueError("study_env cannot be None.")

        self.env = study_env
        self.dfstat = self.env.dfstat
        self.dfexam = self.env.dfexam
        # 🔥 اصلاح: اگر None بود، تبدیل به دیکشنری خالی شود
        self.rl_id_to_db_id_map = rl_id_to_db_id_map if rl_id_to_db_id_map is not None else {}       
        # این خط هم برای اطمینان ایمن شود
        self.db_id_to_rl_id_map = {db_id: rl_id for rl_id, db_id in self.rl_id_to_db_id_map.items()}        
        # ۱. اصلاح دسترسی به نقشه‌ها (بدون خطا)
        self.index_to_actiontype = action_maps['index_to_actiontype']
        self.actiontype_to_index = action_maps['actiontype_to_index']
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_columns = num_columns
        self.study_env = study_env
        self.sac_index_to_db_id = sac_index_to_db_id

        # ۲. استخراج دقیق n_features برای لایه‌های شبکه
        if isinstance(state_dim, (tuple, list)):
            n_features = int(state_dim[-1])
        else:
            n_features = int(state_dim)

        # ۳. ساخت مدل‌های Actor و Critic (یکبار و صحیح)
        # استفاده از n_features و num_columns تضمین می‌کند ابعاد 25 درست اعمال شود
        self.actor = Actor(n_features, self.action_dim)
        self.critic = Critic(n_features, num_columns, self.action_dim)
        self.target_critic = Critic(n_features, num_columns, self.action_dim)
        
        # مقداردهی اولیه وزن‌های Target Critic
        self.target_critic.set_weights(self.critic.get_weights())

        # ۴. تست اولیه (Build) مدل‌ها برای پایداری گراف ترافلو
        dummy_input = tf.zeros((1, n_features), dtype=tf.float32)
        _ = self.actor(dummy_input, training=False)
        _ = self.critic(dummy_input, training=False)

        # بقیه تنظیمات (ReplayBuffer, Optimizers, Entropy)
        self.replay_buffer = ReplayBuffer(self.env.dfstat, self.env.dfexam)
        self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.critic.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        
        self.gamma = 0.99
        self.tau = 0.01
        self.alpha = 0.2  # پارامتر دما برای آنتروپی در DSAC
        self.book_queryset = books_model.objects.filter(
            user=User.objects.get(id=CURRENT_USER_ID)
        ).order_by('id').values_list('id','name', 'tedad_safahat')

        # ✅ اصلاح ۱: ساخت دیکشنری self.books (نام -> آی‌دی دیتابیس)
        # این دیکشنری پایه است و باید قبل از بقیه ساخته شود
        self.books = {
            name: db_id 
            for db_id, name, _ in self.book_queryset
        }

        # ✅ اصلاح ۲: نگاشت نام به ایندکس شبکه (مثال: 'ریاضی' -> 1)
        # استفاده از enumerate برای تولید اعداد 1 تا N
        self.books_name_to_idx = {
            name: idx + 1
            for idx, (db_id, name, _) in enumerate(self.book_queryset)
        }

        # ✅ اصلاح ۳: نگاشت معکوس ایندکس شبکه به نام (مثال: 1 -> 'ریاضی')
        # بهترین کار این است که دقیقاً برعکس دیکشنری بالا ساخته شود تا هماهنگ باشند
        self.books_idx_to_name = {
            v: k 
            for k, v in self.books_name_to_idx.items()
        }
        
        # ✅ اصلاح ۴: نگاشت آی‌دی دیتابیس به نام (مثال: 10543 -> 'ریاضی')
        # برعکس self.books
        self.books_db_id_to_name = {
            v: k 
            for k, v in self.books.items()
        }
        
        # ✅ اصلاح ۵: ایندکس شبکه به ID دیتابیس (مثال: 1 -> 10543)
        self.books_idx_to_db_id = {
            self.books_name_to_idx[name]: db_id
            for db_id, name, _ in self.book_queryset
        }
        
        # ✅ اصلاح ۶: نگاشت اطلاعات (ID دیتابیس -> تعداد صفحات)
        self.book_info_dict = {
            idx + 1: tedad
            for idx, (db_id, name, tedad) in enumerate(self.book_queryset)
        }


    #------------------تابع تبدیل عدد فعالیت به نام فعالیت (اصلاح‌شده)
    def get_activity_name(self, rl_id_input, user):
        """
        تابع تبدیل کد RL به نام فعالیت (با دسترسی به SAC maps)
            نگاشت: 1-6 ثابت، 7 مطالعه، 8 آزمون، 9+ کاربر
        """
        from study.models import type_Faliat_model # نیاز است این مدل اینجا هم import شود
        if rl_id_input == "__STUDY__":
            return "مطالعه"
        if rl_id_input == "__EXAM__":
            return "آزمون"   
        try:
            # 🔥 اصلاح ۱: تبدیل مقاوم به عدد صحیح 
            # از float برای مدیریت مقادیر NumPy float و سایر انواع عددی استفاده می‌کنیم.
            rl_id = int(float(rl_id_input)) 
        except (ValueError, TypeError):
            # اگر نتوانستیم تبدیل کنیم، فوراً نامعتبر است.
            return "فعالیت نامعتبر" 

        # 1. فعالیت‌های ثابت (1 تا 6)
        fixed_map = {
            1: "خواب",
            2: "تغذیه",
            3: "ورزش",
            4: " سرگرمی", 
            5: "کلاس درس یا فوق برنامه",
            6: "سایر"
        }
    
        if 1 <= rl_id <= 6:
            return fixed_map.get(rl_id, f"فعالیت {rl_id}")

        # 2. فعالیت‌های اختصاصی (7 و 8)
        if rl_id == 7: return "مطالعه"
        if rl_id == 8: return "آزمون"

        # 3. فعالیت‌های کاربر (9 به بالا)
        if rl_id >= 9:
            # 🔥 اصلاح ۲: استفاده از نقشه ذخیره شده در SAC (self)
            db_id = self.rl_id_to_db_id_map.get(rl_id)
                
            if db_id is None:
                # 🔥 تغییر جدید: اگر پیدا نشد، فرض کنیم rl_id خودش db_id خام هست
                potential_db_id = rl_id  # rl_id رو به عنوان db_id در نظر بگیر
                rl_id_from_map = self.db_id_to_rl_id_map.get(potential_db_id)  # چک map معکوس
                if rl_id_from_map is not None:
                    db_id = potential_db_id  # حالا db_id درست (مثل 23)
                    logger.debug(f"REWARD DEBUG: Detected raw DB ID {rl_id}, mapped to RL ID {rl_id_from_map} using reverse map")
                else:
                    # اگر هنوز None، یعنی واقعاً پیدا نشد
                    return f"فعالیت شخصی ({rl_id})"

            try:
                # کوئری زدن با DB ID واقعی
                obj = type_Faliat_model.objects.get(id=db_id, user=user)
                return obj.name
            except type_Faliat_model.DoesNotExist:
                return f"فعالیت شخصی ({rl_id}) - حذف شده"
            except Exception as e:
                logger.error(f"Error fetching custom activity name for RL ID {rl_id} (DB ID {db_id}): {e}")
                return f"فعالیت شخصی ({rl_id})"
            
        return "نامشخص" # اگر هیچ‌کدام از شرایط بالا برقرار نشد
        #===========پایان تابع معکوس عدد به نام فعالیت ===============
    #@tf.function(jit_compile=True)
    @tf.function(reduce_retracing=True)
    def update_target_critic(self):
        # (این کد مانند قبل و درست است)
        for target_param, param in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target_param.assign(target_param * (1 - self.tau) + param * self.tau)

    # برای توابع ساده‌تر، autograph را غیرفعال کنید
    #@tf.function(jit_compile=True)
    @tf.function(reduce_retracing=True)
    def select_action(self, state):
        """تابع اصلاح شده - بدون @tf.function"""
        if not isinstance(state, tf.Tensor):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
        
        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)
        elif len(state.shape) > 2:
            state = tf.reshape(state, (1, -1))

        # استفاده از call معمولی به جای _call_infer
        probs, log_probs, _ = self.actor(state, training=False)
        
        # نمونه‌گیری و تبدیل به عدد
        action_index_tensor = tf.random.categorical(log_probs, 1)
        
        # تبدیل ایمن به عدد - بدون .numpy() در محیط symbolic
        #action_index = int(action_index_tensor[0][0])
        
        return action_index_tensor
    #---------------------------------------------------
    
    @tf.function(input_signature=UPDATE_SIGNATURE_SAC, jit_compile=True)
    def update(self, states, actions, rewards, next_states): 
        """
        تابع update: بهینه شده برای XLA.
        ورودی‌ها (states, actions,...) باید از قبل در حلقه train نمونه‌برداری و به عنوان آرگومان ارسال شده باشند.
        """
        # 🌟 1. تبدیل به تنسور (اگرچه ورودی input_signature انتظار تنسور می‌دهد، این کار ضرری ندارد)
        # ⚠️ اگر در حلقه train از قبل تنسور نساخته‌اید، این خطوط ضروری هستند.
        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_t = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32) 
        rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
        rewards_t = tf.reshape(rewards_t, [-1, 1])

        # ---------- آموزش Critic در یک بلوک ----------
        with tf.GradientTape() as tape:
            # محاسبات با تنسورهای جدید (*_t)
            next_probs, next_log_probs, _ = self.actor(next_states_t, training=False)
            next_q1, next_q2 = self.target_critic(next_states_t, training=False)
            next_q = tf.minimum(next_q1, next_q2) # اصلاح منطقی SAC
            
            # محاسبه V
            next_v = tf.reduce_sum(
                next_probs * (next_q - self.alpha * next_log_probs),
                axis=1,
                keepdims=True
            )
            target_q = rewards_t + (self.gamma * next_v) # استفاده از rewards_t

            q1, q2 = self.critic._call_train(states_t)
            
            # محاسبه زیان
            batch_indices = tf.range(tf.shape(actions_t), dtype=tf.int32)
            action_indices = tf.stack([batch_indices, actions_t], axis=1)

            q1_taken = tf.gather_nd(q1, action_indices)
            q2_taken = tf.gather_nd(q2, action_indices)
            q1_taken = tf.reshape(q1_taken, [-1, 1])
            q2_taken = tf.reshape(q2_taken, [-1, 1])

            critic_loss = tf.reduce_mean((q1_taken - target_q)**2 + (q2_taken - target_q)**2)
            
    
        # بهینه‌سازی یکجا
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # ---------- آموزش Actor در یک بلوک ----------
        with tf.GradientTape() as tape:
            probs, log_probs, _ = self.actor._call_train(states)  # یک بار فراخوانی
            q1, q2 = self.critic._call_infer(states)              # یک بار فراخوانی
            q = tf.minimum(q1, q2)

            actor_objective = tf.reduce_sum(
                probs * (q - self.alpha * log_probs),
                axis=1,
                keepdims=True
            )
            actor_loss = -tf.reduce_mean(actor_objective)

        # بهینه‌سازی یکجا
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        #========دیباگ ===============
        # اگر q موجود است از همان استفاده کنیم
        try:
            q_vals = q.numpy()  # q باید tensor شکل (batch, action_dim) یا (batch,) باشد
            # اگر q_vals بعد دوم وجود ندارد بسته به ساختار، آن را تطبیق می‌دهیم
            if q_vals.ndim == 1:
                # اگر q shape == (batch,) احتمالا مشکل داریم؛ اما حداقل mean overall را چاپ می‌کنیم
                print("DEBUG Critic q-values mean (overall):", q_vals.mean())
            else:
                print("DEBUG Critic q-values mean per action:", q_vals.mean(axis=0))
        except Exception as e:
            # fallback: اگر q موجود نبود، تلاش کن q1,q2 را چاپ کنی
            try:
                q1_vals = q1.numpy()
                q2_vals = q2.numpy()
                qmin = np.minimum(q1_vals, q2_vals)
                if qmin.ndim == 1:
                    print("DEBUG Critic q-values mean (overall):", qmin.mean())
                else:
                    print("DEBUG Critic q-values mean per action:", qmin.mean(axis=0))
            except Exception as ex:
                print("DEBUG: unable to print critic q-values:", ex)
        #============پایان دیباگ ================    
        # به‌روزرسانی هدف
        self.update_target_critic()
                

    #=====================تابع آموزش یا ترین =========================
    def train(self, num_episodes, exam_type, score, previous_score, time_gap):
        """تابع بهینه‌شده train -"""
        #🌟 2. تبدیل num_episodes به int پایتونی (برای حلقه ساده)
        num_episodes_int = int(num_episodes)
        # پارامترهای بهینه‌سازی
        UPDATE_FREQUENCY = 10  # کاهش فرکانس به‌روزرسانی
        BATCH_ACCUMULATION = 32  # جمع‌آوری چند نمونه قبل از به‌روزرسانی
            
        batch_counter = 0
        #batch_counter = tf.constant(0, dtype=tf.int32) # شمارنده تنسوری
            
        for episode in range(num_episodes_int): 
            state = self.env.reset()
            done = False
            step_count = 0
            max_steps = 20
                
            while not done and step_count < max_steps:
                # انتخاب action
                action_index_tensor = self.select_action(state) # نام متغیر را تغییر می‌دهیم

                # 🔥 تبدیل حیاتی: تبدیل تنسور به عدد صحیح پایتون 🔥
                # این کار را باید با .numpy() انجام دهیم زیرا train() خارج از @tf.function است.
                action_0_indexed = int(action_index_tensor.numpy().flatten()[0])
                if action_0_indexed >= self.env.n_actions:
                    logger.warning(
                        f"Masked invalid action_index={action_0_indexed} "
                        f"for user={CURRENT_USER_ID}, n_actions={self.env.n_actions}"
                    )
                    continue
                next_state, reward, done, info = self.env.step(action_0_indexed)                                      
                # ذخیره در بافر
                self.replay_buffer.add(state, action_0_indexed, reward, next_state)
                    
                state = next_state
                step_count += 1
                batch_counter += 1
                from collections import Counter
                try:
                    print("DEBUG --- Replay buffer action counts:", Counter(self.replay_buffer.action_indices))
                except Exception as e:
                    print("Replay buffer count check error:", e)

                # به‌روزرسانی مشروط و با فرکانس کمتر
                if (len(self.replay_buffer) >= self.replay_buffer.batch_size and 
                    batch_counter >= BATCH_ACCUMULATION and 
                    step_count % UPDATE_FREQUENCY == 0):
                    # 🌟 2. فراخوانی update() با داده‌های نمونه‌برداری شده
                    self.update(states, actions, rewards, next_states)
                    batch_counter = 0  # reset counter

                    
                if done:
                    break
        #====================نگاشت اکشن به فعالیت ========================
    
    def map_action_to_activity(self, action):
        # نگاشت اکشن عددی به نوع فعالیت
        #activity_type = int(action[0].numpy()[0][0])  # استخراج مقدار از Tensor و تبدیل به عدد صحیح
        activity_type = int(action)
        start_time = action[1].numpy()[0][0]  # استخراج مقدار زمان شروع از Tensor
        duration = action[2].numpy()[0][0]  # استخراج مقدار مدت زمان از Tensor
        
        return {
            "activity_type": activity_type,
            "start_time": start_time,
            "duration": duration
        }
    #=====================تابع تبدیل عدد به ساعت و دقیقه =========================
    def decimal_to_time(self, decimal):
        #تابعی که عدد را به ساعت و دقیقه تبدین می کند
        try:
            # حالت‌های نامعتبر: None یا NaN
            if decimal is None or (isinstance(decimal, float) and math.isnan(decimal)):
                return "0:00"
            
            decimal_float = float(decimal)
            hours = int(decimal_float)
            minutes = int(round((decimal_float - hours) * 60))  # اضافه کردن round برای دقیقه‌های دقیق
            return f"{hours}:{minutes:02d}"
        
        except (ValueError, TypeError):
            return "0:00"
#=======================
    def suggest_study_type(self,book_name, start_page, end_page):
        #تابعی برای یافتن نوع مطالعه     
        #این تابع دوباره باید با مشورت مشاوره دوباره بازبینی و تغییر کند
        activities_df= getattr(self, 'dfstat', pd.DataFrame()) #create_dfstat(user=User.objects.get(id=CURRENT_USER_ID)) 
        exams_df= getattr(self, 'dfexam', pd.DataFrame()) #create_dfexam(user=User.objects.get(id=CURRENT_USER_ID)) 
        # بررسی آیا کتاب و بازه صفحات قبلاً مطالعه شده‌اند
        previous_studies = activities_df[
            (activities_df['actiontype'] == 7) &
            (activities_df['bookname'] == int(book_name)) &
            (activities_df['startpage'] == int(start_page)) &
            (activities_df['endpage'] == int(end_page))
        ]
    
        if previous_studies.empty:
            # اگر اولین بار است که کتاب مطالعه می‌شود، نوع مطالعه را ۲ (مطالعه با خلاصه‌نویسی) پیشنهاد دهید
            return 2
        else:
            # بررسی نوع مطالعه‌ی قبلی
            last_study_type = previous_studies['typestudy'].iloc[-1]
        
            if last_study_type == 1:
                # اگر نوع مطالعه‌ی قبلی ۱ بود، نمره‌ی آزمون را بررسی کنید
                exam_result = exams_df[
                    (exams_df['bookname'] == book_name) &
                    (exams_df['startpage'] == start_page) &
                    (exams_df['endpage'] == end_page)
                ]
            
                if not exam_result.empty:#باید اصلاح شود 
                    #keywords = ['ریاضی', 'فیزیک', 'شیمی', 'حسابان', 'هندسه']
                    score = exam_result['score'].iloc[-1]
                
                    if score >= 0.8:
                        # اگر نمره بالای ۰.۸ بود، نوع مطالعه را ۱ پیشنهاد دهید
                        return 1
                    elif 0.5 <= score < 0.8:
                        # اگر نمره بین ۰.۵ تا ۰.۸ بود، برای کتاب‌های ریاضی، فیزیک و شیمی نوع مطالعه را ۳ پیشنهاد دهید
                        #if book_name in [1, 2, 3]:  # فرض کنید ۱: ریاضی، ۲: فیزیک، ۳: شیمی
                        #    return 3
                        #if isinstance(book_name, str):
                        #    for kw in keywords:
                        #        if kw in book_name:
                        #            return 3
                        #else:
                            # برای بقیه کتاب‌ها نوع مطالعه را ۲ پیشنهاد دهید
                        return 2
                    else:
                        # اگر نمره کمتر از ۰.۵ بود، نوع مطالعه را ۷ پیشنهاد دهید
                        return 7
                else:
                    # اگر نمره‌ی آزمون موجود نبود، نوع مطالعه را ۲ پیشنهاد دهید
                    return 2
            elif last_study_type in [2, 6]:
                # اگر نوع مطالعه‌ی قبلی ۲ یا ۶ بود، این بار هم نوع مطالعه را ۶ پیشنهاد دهید
                return 6
            else:
                # برای سایر موارد، نوع مطالعه را ۲ پیشنهاد دهید
                return 1
#======================ادامه مطالعه در صورتی که کتابی برای مطالعه وجود نداشت ========================
    def get_continued_study(self, user_id, df_stat):
        """
        ادامه آخرین مطالعه قبلی را برمی‌گرداند:
        (book_name, start_page, end_page)
        """
        if df_stat is None or df_stat.empty:
            return None, 0, 0

        last_study = df_stat[df_stat['actiontype'] == 7]
        if last_study.empty:
            return None, 0, 0

        last_row = last_study.iloc[-1]

        book_id = int(last_row['bookname'])
        last_end = int(last_row['endpage'])
        last_start = int(last_row['startpage'])

        # اندازه مطالعه قبلی
        study_size = last_end - last_start
        if study_size <= 0:
            study_size = 10  # fallback امن

        new_start = last_end + 1
        new_end = new_start + study_size

        return book_id, new_start, new_end

#=====================تابع تعیین نوع تغذیه و نوع غذا =========================
    def determine_food(self,current_time, daily_schedule, studydataset_path):
        #این تابع برای تعیین نوع تغذیه و نوع غذا است
        #مهمترین صبحانه است  به همین خاطر تمام صبحانه های قبلی را بررسی می کنیم هر صبحانه ای
        #  که منجر به آن شده باشد که بیشتر درس را بخواند آن را انتخاب می کند
        # خواندن دیتاست
        #df = create_dfstat(user=User.objects.get(id=CURRENT_USER_ID))
        df = getattr(self, 'dfstat', None)
        #if df is None:
        #    df = create_dfstat(user=User.objects.get(id=CURRENT_USER_ID))
        # جایگزینی مقادیر NaN با صفر
        df.fillna(0, inplace=True)

        # تعریف متغیرهای لازم
        typenutrition = 0
        food_type = 0
    
        # تعیین حالت فعلی (state) از daily_schedule
        selected_meals = {
            1: False,  # صبحانه
            2: False,  # نهار
            3: False   # شام
        }

        # بررسی daily_schedule برای تنظیم حالت فعلی
        for entry in daily_schedule:
            if 'typenutrition' in entry and entry['typenutrition'] in selected_meals:
                selected_meals[entry['typenutrition']] = True

        # تعیین نوع تغذیه بر اساس ساعت
        if 0<current_time < 0.48:  # قبل از 11:30 (0.48)
            if not selected_meals[1]:  # اگر هیچ صبحانه‌ای ثبت نشده باشد
                typenutrition = 1  # نوع تغذیه: صبحانه
            
                # تعیین بهترین نوع غذا برای صبحانه از دیتاست
                best_food_per_day = {}
                dates = df['date'].unique()
                for date in dates:
                    daily_data = df[df['date'] == date]
                    max_study_hours = 0
                    best_food_type = 1  # پیش‌فرض: نوع 1
                    for food in range(1, 5):  # انواع صبحانه
                        subset = daily_data[(daily_data['typenutrition'] == 1) & (daily_data['typefood'] == food)]
                        total_study_hours = subset['duringtime'].sum()
                        if total_study_hours > max_study_hours:
                            max_study_hours = total_study_hours
                            best_food_type = food
                    best_food_per_day[date] = best_food_type
            
                # انتخاب نوع غذا بر اساس دیتاست
                current_date = df['date'].max()  # آخرین تاریخ در دیتاست
                food_type = best_food_per_day.get(current_date, 1)  # پیش‌فرض: نوع 1
        
            else:  # اگر قبلاً صبحانه انتخاب شده باشد
                typenutrition = 4  # نوع تغذیه: میان‌وعده
                food_type = 5 if daily_schedule[-1].get('typefood', 5) != 5 else 6  # نوع غذا: 5 یا 6
    
        elif 0.48 <= current_time < 0.75:  # بین 11:30 تا 18:00 (0.75)
            if not selected_meals[2]:  # اگر هیچ نهاری ثبت نشده باشد
                typenutrition = 2  # نوع تغذیه: نهار
            
                # پیدا کردن آخرین نوع غذا برای نهار از دیتاست
                last_lunch_food = df[df['typenutrition'] == 2]['typefood'].iloc[-1] if (df['typenutrition'] == 2).any() else None
                if last_lunch_food is not None:
                    food_type = 1 if last_lunch_food != 1 else 2
                else:
                    food_type = 1  # نوع غذا پیش‌فرض
        
            else:  # اگر قبلاً نهار انتخاب شده باشد
                typenutrition = 4  # نوع تغذیه: میان‌وعده
                last_snack_food = daily_schedule[-1].get('typefood', 5)
                if last_snack_food is not None:
                    food_type = 5 if last_snack_food != 5 else 6
                else:
                    food_type = 5  # نوع غذا پیش‌فرض
    
        else:  # بعد از 18:00 (0.75)
            if not selected_meals[3]:  # اگر هیچ شامی ثبت نشده باشد
                typenutrition = 3  # نوع تغذیه: شام
            
                # پیدا کردن آخرین نوع غذا برای نهار از دیتاست
                last_lunch_food = df[df['typenutrition'] == 2]['typefood'].iloc[-1] if (df['typenutrition'] == 2).any() else None
                if last_lunch_food is not None:
                    food_type = 3 if last_lunch_food != 3 else 4
                else:
                    food_type = 3  # نوع غذا پیش‌فرض
        
            else:  # اگر قبلاً شام انتخاب شده باشد
                typenutrition = 4  # نوع تغذیه: میان‌وعده
                last_snack_food = daily_schedule[-1].get('typefood', 5)
                if last_snack_food is not None:
                    food_type = 5 if last_snack_food != 5 else 6
                else:
                    food_type = 5  # نوع غذا پیش‌فرض
    
        return typenutrition, food_type
    #===================== تابع تعیین کتاب و صفحه شروع و پایان آزمون =========================
    @log_call
    def suggest_exam(self, df_stat, df_exam):
        """
        پیشنهاد آزمون بر اساس مطالعه‌های مرتبط و نزدیک.

        منطق علمی:
        - آزمون باید ابزار retrieval practice و formative assessment باشد.
        - اگر فاصله آخرین مطالعه مرتبط خیلی زیاد باشد، آزمون مستقیم باعث نمره پایین
          و داده کم‌کیفیت می‌شود؛ در این حالت بهتر است ابتدا مرور/مطالعه پیشنهاد شود.
        - بنابراین آزمون‌های 0 تا 3 روز بعد از مطالعه اولویت بالا دارند،
          4 تا 6 روز اولویت متوسط دارند، و بازه‌های خیلی قدیمی فقط fallback هستند.
        """

        assert isinstance(next(iter(self.book_info_dict.keys())), int)

        if df_stat is None or df_stat.empty:
            return []

        if df_exam is None:
            df_exam = pd.DataFrame()

        studies = df_stat[df_stat['actiontype'] == 7].copy()
        if studies.empty:
            return []

        studies['date'] = pd.to_numeric(studies['date'], errors='coerce').fillna(0)

        current_day = float(studies['date'].max())

        def to_real_page(book_id, p):
            try:
                total = self.book_info_dict.get(int(book_id), 100)
                p = float(p)
                if 0 < p <= 1:
                    return max(1, int(round(p * total)))
                return max(1, int(round(p)))
            except Exception:
                return 1

        # آماده‌سازی آزمون‌های قبلی برای جلوگیری از پیشنهاد آزمون تکراری پس از آخرین مطالعه
        exam_lookup = []
        if df_exam is not None and not df_exam.empty:
            ex = df_exam.copy()
            ex['date'] = pd.to_numeric(ex.get('date', 0), errors='coerce').fillna(0)
            for _, erow in ex.iterrows():
                try:
                    eb = int(erow.get('bookname', 0))
                    es = to_real_page(eb, erow.get('startpage', 1))
                    ee = to_real_page(eb, erow.get('endpage', es))
                    ed = float(erow.get('date', 0))
                    exam_lookup.append((eb, es, ee, ed))
                except Exception:
                    continue

        # آخرین مطالعه هر بازه صفحه
        candidates_raw = []

        for _, row in studies.iterrows():
            try:
                book_id = int(row.get('bookname', 0))
                if book_id not in self.book_info_dict:
                    continue

                s = to_real_page(book_id, row.get('startpage', 1))
                e = to_real_page(book_id, row.get('endpage', s))
                d = float(row.get('date', 0))

                if e < s:
                    e = s

                candidates_raw.append((book_id, s, e, d))
            except Exception:
                continue

        if not candidates_raw:
            return []

        # برای هر کتاب/بازه، فقط آخرین مطالعه را نگه می‌داریم.
        latest = {}
        for book_id, s, e, d in candidates_raw:
            key = (book_id, s, e)
            if key not in latest or d > latest[key]:
                latest[key] = d

        exam_suggestions = []
        stale_fallback = []

        for (book_id, s, e), last_study_day in latest.items():
            days_passed = current_day - last_study_day

            # آیا بعد از آخرین مطالعه، آزمونی برای همین بازه گرفته شده؟
            tested_after_last_study = False
            for eb, es, ee, ed in exam_lookup:
                overlap = not (ee < s or es > e)
                if eb == book_id and overlap and ed >= last_study_day:
                    tested_after_last_study = True
                    break

            if tested_after_last_study:
                continue

            # اولویت علمی
            if 0 <= days_passed <= 3:
                tier = 0
                priority_score = 100 - days_passed
                exam_suggestions.append((tier, -priority_score, book_id, (s, e)))

            elif 3 < days_passed <= 6:
                tier = 1
                priority_score = 60 - days_passed
                exam_suggestions.append((tier, -priority_score, book_id, (s, e)))

            elif 6 < days_passed <= 10:
                # فقط fallback، نه اولویت اصلی
                tier = 2
                priority_score = 20 - days_passed
                stale_fallback.append((tier, -priority_score, book_id, (s, e)))

        # اگر candidate تازه داریم، از staleها استفاده نمی‌کنیم.
        final_candidates = exam_suggestions if exam_suggestions else stale_fallback

        final_candidates.sort(key=lambda x: (x[0], x[1]))

        # خروجی فقط book_id و pages
        return [(book_id, pages) for _, _, book_id, pages in final_candidates[:30]]


    #===================== پایان تابع آزمون نام کتاب صفحه شروع و پایان =========================
    
    #=====================تابع تولید برنامه روزانه =========================
    def generate_daily_schedule(self):
        daily_schedule = []
        # اگر در شبیه‌سازی تاریخ هدف تنظیم شده باشد، همان را استفاده می‌کنیم
        # در غیر این صورت کد مثل قبل با تاریخ امروز کار می‌کند
        target_date = SIMULATION_TARGET_DATE or datetime.today().date()

        if isinstance(target_date, datetime):
            target_date = target_date.date()

        date = target_date.strftime('%Y-%m-%d')
        date2 = target_date
        weekday = date2.weekday()
        #daily_schedule = []
        #date=datetime.today().strftime('%Y-%m-%d')
        #date2 = datetime.now().date() # برای تعطیلی پنج شنبه و جمعه
        #weekday = date2.weekday()  # 0: شنبه, 6: جمعه
        # بارگذاری داده‌ها از دیتافریم کاربر مربوطه
        dfstat = self.env.dfstat
        def fix_bookname_type(df):
            """
            تبدیل یکدست bookname به integer در هر دو محیط
            """
            if 'bookname' in df.columns:
                # ابتدا تبدیل به numeric با مدیریت خطا
                df['bookname'] = pd.to_numeric(df['bookname'], errors='coerce')
                # حذف مقادیر NaN
                df = df[df['bookname'].notna()]
                # تبدیل به integer
                df['bookname'] = df['bookname'].astype(int)
            return df
        dfstat = fix_bookname_type(dfstat)                   
        df = dfstat.fillna(0) 
        book_list = list(self.book_queryset)
        self.exam_suggestions = self.suggest_exam(self.env.dfstat, self.env.dfexam)
        self.exam_index = 0
        # پیدا کردن تاریخ امروز (آخرین تاریخ ثبت‌شده در دیتافریم)
        current_date = df['date'].max()
        # پیدا کردن تاریخ روز قبل
        previous_date = current_date - 1
        #========تابع محاسبه زمان شروع فعالیت ها (نسخه اصلاح شده کامل)=========
        def calculate_best_start_time(dfstat):
            """
            این تابع با بررسی ۱۴ روز گذشته‌ی دانش‌آموز
            و با استفاده از سه فاکتور مدت مطالعه، استرس و رضایت،
            بهترین ساعت شروع فعالیت روزانه را محاسبه می‌کند.
            خروجی: مقدار نرمال‌شده ساعت شروع در متغیر current_time
            """

            # -----------------------------
            # ۱. آماده‌سازی داده‌ها
            # -----------------------------

            if dfstat.empty:

                return 8.0 / 24.0  # مقدار پیش‌فرض ساعت ۸ صبح (نرمال شده)

            # تبدیل ستون تاریخ به عدد صحیح (اگر نیست)
            dfstat['date'] = pd.to_numeric(dfstat['date'], errors='coerce')

            # فقط ۱۴ روز اخیر را نگه داریم
            max_day = dfstat['date'].max()
            df_14 = dfstat[dfstat['date'] > max_day - 14].copy()

            if df_14.empty:

                return 8.0 / 24.0

            # -----------------------------
            # ۲. محاسبه مجموع مطالعه در هر روز
            # -----------------------------

            # فقط ردیف‌هایی که نوع فعالیت = مطالعه هستند (actiontype == 1)
            df_study = df_14[df_14['actiontype'] == 7].copy()

            # محاسبه مجموع مدت مطالعه در هر روز
            daily_study = df_study.groupby('date')['duringtime'].sum().reset_index()
            daily_study.rename(columns={'duringtime': 'total_study_duration'}, inplace=True)

            # -----------------------------
            # ۳. محاسبه میانگین استرس و رضایت (فقط مقادیر غیر صفر)
            # -----------------------------

            def mean_ignore_zeros(series):
                """میانگین فقط از مقادیری که صفر نیستند"""
                nonzero = series[series != 0]
                return nonzero.mean() if len(nonzero) > 0 else np.nan

            daily_emotions = (
                df_study.groupby('date')
                .agg({
                    'estres': mean_ignore_zeros,
                    'oxijen': mean_ignore_zeros
                })
                .reset_index()
                .rename(columns={'estres': 'avg_stress', 'oxijen': 'avg_satisfaction'})
            )

            # -----------------------------
            # ۴. ساخت اولیه جدول روزانه (ترکیب مطالعه و احساسات)
            # ⚠️ خطای شما اینجا بود: این خط احتمالاً پاک شده بود
            # -----------------------------
            df_days = pd.merge(daily_study, daily_emotions, on='date', how='outer')

            # -----------------------------
            # ۵. اصلاح شده: محاسبه ساعت شروع فعالیت (بدون خواب)
            # -----------------------------
            # تبدیل اکشن‌تایپ به عدد برای جلوگیری از خطای مقایسه (رشته با عدد)
            df_14['actiontype'] = pd.to_numeric(df_14['actiontype'], errors='coerce')

            # تعریف مرز شروع صبح: ساعت ۰۴:۳۰ صبح
            # (4.5 تقسیم بر 24 می‌شود 0.1875)
            EARLY_MORNING_THRESHOLD = 0.1875 

            # فیلتر کردن داده‌ها:
            # ۱. فعالیت خواب (۳) نباشد
            # ۲. زمان شروع فعالیت باید بعد از ۴:۳۰ صبح باشد (تا فعالیت‌های نیمه شب حذف شوند)
            non_sleep_activities = df_14[
                (df_14['actiontype'] != 1) & 
                (df_14['starttime'] > EARLY_MORNING_THRESHOLD)
            ].copy()
            
            if non_sleep_activities.empty:
                # اگر هیچ فعالیتی بعد از ۵ صبح پیدا نشد، پیش‌فرض ساعت ۷:۳۰ را برگردان
                # (0.3125 * 24 = 7.5 ساعت)
                first_start = pd.DataFrame({'date': df_days['date'], 'first_starttime': 0.3125}) 
            else:
                # محاسبه اولین فعالیت روز
                first_start = (
                    non_sleep_activities.groupby('date')['starttime']
                    .min()
                    .reset_index()
                    .rename(columns={'starttime': 'first_starttime'})
                )
            
            # -----------------------------
            # ۶. ادغام زمان شروع با جدول اصلی
            # -----------------------------
            # حالا df_days وجود دارد و خطا نمی‌دهد
            df_days = pd.merge(df_days, first_start, on='date', how='left')

            # حذف روزهایی که ساعت شروع یا مدت مطالعه ندارند
            df_days = df_days.dropna(subset=['total_study_duration', 'first_starttime'])

            if df_days.empty:
                return 8.0 / 24.0

            # -----------------------------
            # ۷. نرمال‌سازی مقادیر برای ترکیب منصفانه
            # -----------------------------

            def normalize(series):
                """نرمال‌سازی به بازه [0,1]"""
                if series.max() == series.min():
                    return pd.Series([0.5]*len(series), index=series.index)
                return (series - series.min()) / (series.max() - series.min())

            df_days['norm_study'] = normalize(df_days['total_study_duration'])
            df_days['norm_stress'] = normalize(df_days['avg_stress'])
            df_days['norm_satisfaction'] = normalize(df_days['avg_satisfaction'])

            # -----------------------------
            # ۸. محاسبه امتیاز روز با وزن‌های مشخص‌شده
            # -----------------------------

            w1, w2, w3 = 0.4, 0.3, 0.3

            df_days['DayScore'] = (
                w1 * df_days['norm_study'] +
                w2 * (1 - df_days['norm_stress'].fillna(0.5)) +  # استرس کمتر = امتیاز بالاتر
                w3 * df_days['norm_satisfaction'].fillna(0.5)
            )

            # -----------------------------
            # ۹. انتخاب روزهای برتر و میانگین ساعت شروع آن‌ها
            # -----------------------------

            top_days = df_days.nlargest(5, 'DayScore')  # ۵ روز با امتیاز بالاتر
            current_time = top_days['first_starttime'].mean()

            # اگر مقدار غیرمنطقی بود (مثلاً NaN)
            if pd.isna(current_time):
                current_time = 8.0 / 24.0  # پیش‌فرض ۸ صبح

            return current_time

        
        try:
            state_1d = self.env.reset()
            # تبدیل به numpy اگر تنسور است
            if tf.is_tensor(state_1d):
                state_1d = state_1d.numpy()
            elif hasattr(state_1d, 'numpy'):
                state_1d = state_1d.numpy()
            
            # اطمینان از شکل صحیح
            state_1d = np.array(state_1d, dtype=np.float32).flatten()
            state = tf.convert_to_tensor([state_1d], dtype=tf.float32)
            
        except Exception as e:
            logger.error(f"Error initializing state: {e}")
            # حالت fallback
            state = tf.random.normal((1, self.state_dim[1]), dtype=tf.float32)
        
        #============پایان تابع زمان شروع فعالیت ها=========
        # =================================================================
        # 🌟 بخش جدید: تنظیم پویای آلفا (Alpha) بر اساس نزدیکی به کنکور 🌟
        # =================================================================
        
        # ۱. تعیین تاریخ کنکور (۱۰ تیر ماه سال جاری یا سال بعد)
        # فرض می‌کنیم سیستم میلادی است. ۱۰ تیر تقریبا برابر ۱ جولای است.
        today = target_date
        current_year = today.year
        #today = datetime.now().date()
        #current_year = today.year
        konkur_date = datetime(current_year, 7, 1).date() # 1 July = 10 Tir
        
        # اگر امروز بعد از ۱۰ تیر است، تاریخ کنکور می‌شود سال بعد
        if today > konkur_date:
             konkur_date = datetime(current_year + 1, 7, 1).date()
             
        # ۲. محاسبه تعداد روزهای باقی‌مانده
        days_until_exam = (konkur_date - today).days
        if days_until_exam < 0: days_until_exam = 0 # محض اطمینان

        # ۳. فرمول پویا برای آلفا (Linear Decay)
        # اگر بیش از ۲۰۰ روز مانده: آلفا = ۰.۲۵ (اکتشاف بالا - بگذار روش‌های مختلف را تست کند)
        # اگر ۱ روز مانده: آلفا = ۰.۰۰۱ (تمرکز مطلق - فقط بهترین کار را بکن)
        
        max_days = 200
        max_alpha = 0.25
        min_alpha = 0.005 # نزدیک صفر
        
        if days_until_exam > max_days:
            dynamic_alpha = max_alpha
        else:
            # فرمول خطی کاهش آلفا
            ratio = days_until_exam / max_days
            dynamic_alpha = min_alpha + (max_alpha - min_alpha) * ratio

        # ۴. اعمال آلفا به ایجنت
        self.alpha = dynamic_alpha
        logger.info(f"Days to Konkur: {days_until_exam}, Dynamic Alpha set to: {self.alpha}")
        
        # =================================================================
        current_time = calculate_best_start_time(dfstat)
        previous_action = 0
        selected_actions = set()

        # -----------------------------------------------------------------
        # v27 scheduler safety and feasibility layer
        # -----------------------------------------------------------------
        data = dfstat
        average_times = data.groupby("actiontype")["duringtime"].median()

        final_dates = pd.to_numeric(
            self.env.df_final.get("date", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        planning_day = (
            float(final_dates.max()) + 1.0
            if not final_dates.empty
            else 0.0
        )

        study_priority = StudyPriority(
            self.env.df_final,
            self.book_info_dict,
            planning_day=planning_day,
        )

        history_studies = dfstat[
            pd.to_numeric(
                dfstat.get("actiontype", pd.Series(index=dfstat.index, dtype=float)),
                errors="coerce",
            ).eq(7)
        ].copy()

        if history_studies.empty:
            daily_study_history_hours = pd.Series(dtype=float)
            trailing_zero_study_days = 999
        else:
            history_studies["_v27_date"] = pd.to_numeric(
                history_studies["date"],
                errors="coerce",
            )
            history_studies["_v27_hours"] = (
                pd.to_numeric(
                    history_studies["duringtime"],
                    errors="coerce",
                ).fillna(0.0) * 24.0
            )
            daily_study_history_hours = (
                history_studies
                .dropna(subset=["_v27_date"])
                .groupby("_v27_date")["_v27_hours"]
                .sum()
            )

            studied_days = {
                int(round(float(day_value)))
                for day_value in daily_study_history_hours.index
            }
            trailing_zero_study_days = 0
            probe_day = int(round(planning_day - 1.0))
            while (
                probe_day not in studied_days
                and trailing_zero_study_days < 365
            ):
                trailing_zero_study_days += 1
                probe_day -= 1

        # v27.2: use calendar dates from the authoritative study table to
        # determine the true trailing zero-study gap. The previous v27 logic
        # relied on normalized RL day indices, which can diverge from calendar
        # dates after warm-start and caused the day-three study floor to be
        # missed for some perfect-adherence users.
        try:
            historical_study_dates = set(
                study_model.objects.filter(
                    user_id=CURRENT_USER_ID,
                    tarikh_en_user__lt=target_date,
                ).values_list("tarikh_en_user", flat=True)
            )

            trailing_zero_study_days = 0
            probe_calendar_day = target_date - timedelta(days=1)

            while (
                probe_calendar_day not in historical_study_dates
                and trailing_zero_study_days < 365
            ):
                trailing_zero_study_days += 1
                probe_calendar_day -= timedelta(days=1)

        except Exception as exc:
            logger.warning(
                "v27.2 calendar zero-gap lookup failed for user=%s: %s; "
                "falling back to normalized-history estimate=%s",
                CURRENT_USER_ID,
                exc,
                trailing_zero_study_days,
            )

        # A 15-hour day is allowed only in the final 30 days. The rolling
        # weekly budget and recovery rule prevent 15-hour days from becoming
        # a normal daily pattern for several consecutive weeks.
        if days_until_exam <= 30:
            study_block_cap_hours = 3.0
            hard_daily_study_cap_hours = 15.0
            rolling_weekly_average_cap_hours = 11.5
            minimum_daily_study_hours = 3.0
            maximum_study_blocks = 7
        elif days_until_exam <= 90:
            study_block_cap_hours = 2.5
            hard_daily_study_cap_hours = 13.0
            rolling_weekly_average_cap_hours = 10.5
            minimum_daily_study_hours = 2.0
            maximum_study_blocks = 7
        elif days_until_exam <= 180:
            study_block_cap_hours = 2.0
            hard_daily_study_cap_hours = 11.5
            rolling_weekly_average_cap_hours = 9.5
            minimum_daily_study_hours = 1.0
            maximum_study_blocks = 6
        else:
            study_block_cap_hours = 2.0
            hard_daily_study_cap_hours = 10.0
            rolling_weekly_average_cap_hours = 9.0
            # v27.2 design lock: every simulated learner receives study
            # on every planning day. Recovery changes the upper cap, never
            # removes the daily study floor.
            minimum_daily_study_hours = 1.0
            maximum_study_blocks = 6

        if daily_study_history_hours.empty:
            previous_six_day_hours = 0.0
            yesterday_hours = 0.0
            two_days_ago_hours = 0.0
        else:
            previous_six_day_hours = 0.0
            for offset in range(1, 7):
                previous_six_day_hours += float(
                    daily_study_history_hours.get(
                        planning_day - float(offset),
                        0.0,
                    )
                )
            yesterday_hours = float(
                daily_study_history_hours.get(planning_day - 1.0, 0.0)
            )
            two_days_ago_hours = float(
                daily_study_history_hours.get(planning_day - 2.0, 0.0)
            )

        remaining_weekly_budget = (
            7.0 * rolling_weekly_average_cap_hours
            - previous_six_day_hours
        )

        daily_study_cap_hours = min(
            hard_daily_study_cap_hours,
            max(
                minimum_daily_study_hours,
                remaining_weekly_budget,
            ),
        )

        # Recovery after repeated severe load. This keeps an occasional
        # 15-hour final-phase day possible but prevents a long 15-hour streak.
        if yesterday_hours > 14.0:
            daily_study_cap_hours = min(
                daily_study_cap_hours,
                9.0,
            )

        if yesterday_hours > 12.0 and two_days_ago_hours > 12.0:
            daily_study_cap_hours = min(
                daily_study_cap_hours,
                8.0,
            )

        daily_study_cap_hours = max(
            minimum_daily_study_hours,
            daily_study_cap_hours,
        )

        # v27.2: reserve enough wall-clock time for the mandatory daily
        # study floor before any stochastic action is materialised.
        if minimum_daily_study_hours > 0.0:
            latest_floor_start = max(
                0.0,
                1.0 - (minimum_daily_study_hours / 24.0),
            )
            if current_time > latest_floor_start:
                current_time = latest_floor_start
                self.env.current_time = current_time

        daily_study_planned_hours = 0.0
        study_block_count = 0
        safety_projection_count = 0

        def v27_pick_nonstudy_action(probabilities, fallback=3):
            sorted_indices = np.argsort(probabilities)[::-1]

            for candidate_index in sorted_indices:
                candidate_index = int(candidate_index)
                candidate_rl = self.index_to_actiontype[candidate_index]

                try:
                    is_active = bool(
                        float(
                            self.active_action_mask[
                                candidate_index
                            ].numpy()
                        ) > 0.0
                    )
                except Exception:
                    is_active = True

                if (
                    is_active
                    and candidate_rl not in (7, 8)
                    and candidate_rl != previous_action
                ):
                    return candidate_rl

            if fallback in self.actiontype_to_index:
                return fallback

            return 1

        logger.info(
            "v27 study limits | user=%s | days_to_exam=%s | "
            "block_cap=%.2fh | daily_cap=%.2fh | minimum=%.2fh | "
            "past6=%.2fh | trailing_zero_days=%s",
            CURRENT_USER_ID,
            days_until_exam,
            study_block_cap_hours,
            daily_study_cap_hours,
            minimum_daily_study_hours,
            previous_six_day_hours,
            trailing_zero_study_days,
        )

        planning_attempts = 0
        max_planning_attempts = 250
        while current_time < 1:  # برنامه‌ریزی تا پایان روز، با محافظ حلقه
            planning_attempts += 1
            if planning_attempts > max_planning_attempts:
                logger.warning(
                    f"Planning guard reached for user={CURRENT_USER_ID}; "
                    f"current_time={current_time:.4f}, attempts={planning_attempts}"
                )
                break
            action_index_tensor = self.select_action(state)
            # 🔥 تغییر حیاتی: تبدیل صحیح تنسور به عدد صحیح پایتون و استفاده از آن 🔥
            if tf.is_tensor(action_index_tensor):
                # گرفتن مقدار عددی تنسور و تبدیل آن به عدد صحیح پایتون
                # .numpy() را فقط در محیط Eager (خارج از @tf.function) فراخوانی کنید.
                action_index_python = int(action_index_tensor.numpy().flatten()[0])
            else:
                # در صورت عدم موفقیت، فرض بر این است که عدد صحیح پایتون است
                action_index_python = int(action_index_tensor)
            
            # --- robust state shaping before actor call ---
            s = np.asarray(state)
            if s.ndim == 1:
                s = s[np.newaxis, :]
            # اگر state سه بعدی مثل (1,1,17) است -> به (1,17) ری‌شِیپ کن
            elif s.ndim == 3 and s.shape[0] == 1:
                s = s.reshape(1, s.shape[-1])
            # اگر state دو بعدی (1,17) باشد کاری نمی‌کنیم
            else:
                # سعی می‌کنیم هر چندبعدی را به (batch, features) تبدیل کنیم
                try:
                    s = s.reshape(-1, s.shape[-1])
                except Exception:
                    # آخرین راه: فِلَت کن و reshape به (1, features) با فرض features == state_dim
                    s = s.flatten()
                    s = s[np.newaxis, :]

            # v26: use the same dynamically masked policy distribution for
            # both the primary draw and any anti-repetition fallback.
            probs = self.masked_action_probabilities(s.astype(np.float32))

            # دیباگ ایمن
            # Debug output removed for performance
            pass

            best_action_index = self.index_to_actiontype[action_index_python]
            # best_action_index الان یک RL_ID است (مثلاً 1,2,..., یا 9 برای فعالیت کاربر)
            rl_id = best_action_index
            
            # -----------------------------
            # ۱) جلوگیری از تکرار پشت سر هم (به‌جز مطالعه و خواب)
            # -----------------------------
            ACTIONS_ALLOWED_TO_REPEAT = [1, 7,8]   # 1: مطالعه, 3: خواب
            EXCLUDED_RL_IDS = []  # v26: optional fixed actions are controlled by the recency mask

            if rl_id == previous_action and rl_id not in ACTIONS_ALLOWED_TO_REPEAT:
                # از روی probs یک اکشن جایگزین با احتمال بالا پیدا کن
                sorted_indices = np.argsort(probs)[::-1]
                for alt_idx in sorted_indices:
                    alt_rl = self.index_to_actiontype[int(alt_idx)]  # هنوز RL_ID
                    is_active = bool(float(self.active_action_mask[int(alt_idx)].numpy()) > 0.0)
                    if is_active and alt_rl != previous_action and alt_rl not in EXCLUDED_RL_IDS:
                        rl_id = alt_rl
                        break

            # v27.2: convert an unavailable/exhausted exam action to study
            # BEFORE applying study caps and floors. In v27 this conversion
            # happened later, after the safety projection, allowing converted
            # exam actions to bypass maximum study-block and daily-hour caps.
            if (
                rl_id == 8
                and (
                    not self.exam_suggestions
                    or self.exam_index >= len(self.exam_suggestions)
                )
            ):
                rl_id = 7
                safety_projection_count += 1

            # -------------------------------------------------------------
            # v27 safety projection
            # -------------------------------------------------------------
            # A small phase-dependent study floor prevents multi-week gaps.
            # This is a feasibility constraint, not a claim that study load
            # should rise monotonically every day.
            must_meet_study_floor = (
                daily_study_planned_hours + 1.0e-9
                < minimum_daily_study_hours
                and study_block_count < maximum_study_blocks
                and 7 in self.actiontype_to_index
            )

            if must_meet_study_floor and rl_id != 7:
                rl_id = 7
                safety_projection_count += 1

            remaining_study_capacity = (
                daily_study_cap_hours
                - daily_study_planned_hours
            )

            if (
                rl_id == 7
                and (
                    remaining_study_capacity < 0.50
                    or study_block_count >= maximum_study_blocks
                )
            ):
                rl_id = v27_pick_nonstudy_action(probs)
                safety_projection_count += 1

            previous_action = rl_id

            # Keep the action passed to env.step consistent with any
            # anti-repetition or safety projection applied above.
            action_index_python = int(
                self.actiontype_to_index.get(
                    rl_id,
                    action_index_python,
                )
            )

            # v27: optional class/other actions are admitted only when active in the recency mask.
            if rl_id in EXCLUDED_RL_IDS:
                continue

            best_action_index = rl_id  # از این به بعد، کل منطق تابع روی rl_id اصلاح شده عمل کند
            # 2) نگاشت RL ID به DB ID برای ذخیره در دیتابیس
            # ----------------------------------------------------
            # 🔥 کد اصلاح شده و امن:
            # چک می‌کنیم دیکشنری خالی نباشد و کلید در آن وجود داشته باشد
            if self.rl_id_to_db_id_map and rl_id in self.rl_id_to_db_id_map:
                db_id_for_storage = self.rl_id_to_db_id_map[rl_id]
            else:
                db_id_for_storage = rl_id

            data = dfstat  
            # تعیین نوع فعالیت
            action_type = best_action_index
            # فیلتر کردن داده‌ها بر اساس نوع فعالیت
            #filtered_data = data[data['actiontype'] == action_type]
            # محاسبه میانگین زمان سپری شده مشکل مدت زمان اینجاست احتمالأ بررسی شود
            # مدت زمان انجام هر فعالیت باید میانگین وقتی که برای آن فعالیت ثبت شده در نظر گرفته شود نه همه فعالیت ها با هم
            mean_during_time = average_times.get(best_action_index, 0)

            # شرط زیر بخاطر آن است که بعد از یازده شب تنها فعالیت خواب باید انجام دهد
            if current_time > 0.959:
                best_action_index = 1
                rl_id = 1
                action_index_python = int(
                    self.actiontype_to_index.get(
                        rl_id,
                        action_index_python,
                    )
                )

            if rl_id in EXCLUDED_RL_IDS:
                continue
            
            #اگر فعالیت خواب قبل از ده شب انتخاب شود فقط نیم ساعت باید طول بکشد و بعد از ده هشت ساعت
            #باید این کد تغییر کند اگر ما بین ساعت یازده شب تا مثلأ شش صبح خواب را انتخاب کند هشت ساعت باشد در غیر این صورت نیم ساعت
            if rl_id ==1 and 0<current_time < 0.9166:
                mean_during_time=0.0208
            elif rl_id ==1 and current_time>0.9166:
                mean_during_time=0.3333
            
            #determine_food(self,current_time, daily_schedule, studydataset_path)typenutrition, food_type
            if rl_id == 2 : # برای زمانی که نوع فعالیت تغذیه باشد 
                studydataset_path=dfstat 
                mean_during_time = average_times.get(2, 0)
                typenutrition, food_type=self.determine_food(current_time, daily_schedule, studydataset_path)
                # (اختیاری) اگر می‌خواهید مطمئن شوید که عدد NaN برنمی‌گردد:
                if pd.isna(mean_during_time):
                    mean_during_time = 0  # یا یک زمان پیش‌فرض
            else:
                typenutrition=0
                food_type=0

            if rl_id == 4:# سرگرمی                
                mean_during_time = average_times.get(4, 0)
               #به مرور زمان مدت سرگرمی را پنج درصد کمی میکند تا پنج دقیه
                if mean_during_time > 0.0035:
                        mean_during_time=mean_during_time-(mean_during_time/20)

            if rl_id == 3: # ورزش
                #activity_5_history = data[data['actiontype'] == 5]
                #mean_during_time = activity_5_history['duringtime'].mean()
                mean_during_time = average_times.get(3, 0)

            # تعریف مقادیر پیش‌فرض برای جلوگیری از خطای UnboundLocalError
            # شمارنده آزمون (باید خارج از loop اصلی تعریف شود)
            #book_name = start_page = end_page = studytype = 0
            # ---------------------------------------------------------
            # ۱. متغیرهای اولیه را با مقدار پیش‌فرض تعریف می‌کنیم
            # ---------------------------------------------------------
            book_id_selected = None     # اینجا منظور DB_ID است اگر نیاز شد
            book_idx_selected = 0       # 🔥 متغیر جدید: عدد کوچک شبکه عصبی (1 تا N)
            book_name_str = None
            start_page = 0
            end_page = 0
            studytype = None

            # ساخت دیکشنری تبدیل ID به نام (یکبار انجام شود بهتر است)
            # فرض بر این است که self.book_queryset قبلاً مقداردهی شده است
            id_to_book_name = {pk: name for pk, name, _ in self.book_queryset}

            # ---------------------------------------------------------
            # ۲. بررسی فعالیت آزمون (ID = 8)
            # ---------------------------------------------------------
            # v27.2: unavailable/exhausted exam actions were already
            # converted before the safety projection. Reaching this point
            # with an unavailable exam would indicate a control-flow bug.
            if (
                rl_id == 8
                and (
                    not self.exam_suggestions
                    or self.exam_index >= len(self.exam_suggestions)
                )
            ):
                raise RuntimeError(
                    "v27.2 invariant violation: exhausted exam action "
                    "reached the exam scheduling branch."
                )

            # ---------------------------------------------------------
            # ۲. بررسی فعالیت آزمون (ID = 8)
            # ---------------------------------------------------------
            if rl_id == 8:
                if self.exam_suggestions:
                    idx = self.exam_index
                    exam_book_obj, (start_page, end_page) = self.exam_suggestions[idx]
                    
                    # مقدار پیش‌فرض
                    book_name_str = None
                    book_id_selected = None

                    if isinstance(exam_book_obj, int):
                        input_val = exam_book_obj
                        
                        # جستجوی اول: آیا این عدد، ایندکس شبکه است؟ (مثلاً 0 تا 10)
                        # معمولاً اعداد شبکه کوچک هستند
                        if input_val in self.books_idx_to_name:
                            book_name_str = self.books_idx_to_name[input_val]
                            # پیدا کردن ID دیتابیس از روی نام
                            book_id_selected = self.books.get(book_name_str)
                            
                        # جستجوی دوم: اگر پیدا نشد، آیا این عدد ID دیتابیس است؟ (مثلاً 42، 52...)
                        elif input_val in self.books_db_id_to_name:
                            book_name_str = self.books_db_id_to_name[input_val]
                            book_id_selected = input_val
                        
                        else:
                            # اگر کلا پیدا نشد (حالت خطا)
                            print(f"❌ هشدار: کد کتاب {input_val} شناسایی نشد.")
                            book_name_str = "آزمون عمومی"
                            book_id_selected = 0
                            
                    else:
                        # اگر ورودی رشته بود (نام کتاب)
                        book_name_str = str(exam_book_obj)
                        book_id_selected = self.books.get(book_name_str, 0)

                    # اطمینان نهایی برای جلوگیری از خطای NoneType
                    if book_name_str is None:
                        book_name_str = "نامشخص"
                    
                    if book_id_selected is None: 
                        book_id_selected = 0

                    self.exam_index += 1
                    
                else:
                    # اگر پیشنهادی وجود نداشت
                    book_id_selected = 0
                    book_name_str = "آزمون عمومی"
                    start_page = 1
                    end_page = 20

            # ---------------------------------------------------------
            # ۱. بررسی فعالیت مطالعه (ID = 7)
            # ---------------------------------------------------------
            elif rl_id == 7:
                try:
                    selected = study_priority.suggest_next()

                    if selected is None:
                        available_books = sorted(self.book_info_dict)
                        if not available_books:
                            raise ValueError(
                                "v27 scheduler found no registered books."
                            )

                        raw_val = int(available_books[0])
                        start_page = 1
                        end_page = min(
                            10,
                            int(self.book_info_dict[raw_val]),
                        )
                    else:
                        raw_val, (start_page, end_page) = selected
                        raw_val = int(raw_val)

                    # StudyPriority returns the network/book index. Resolve
                    # it deterministically instead of ambiguously treating
                    # small database IDs as network indices.
                    if raw_val in self.books_idx_to_name:
                        book_name_str = self.books_idx_to_name[raw_val]
                        book_id_selected = self.books_idx_to_db_id.get(
                            raw_val,
                            self.books.get(book_name_str, 0),
                        )
                    elif raw_val in self.books_db_id_to_name:
                        book_name_str = self.books_db_id_to_name[raw_val]
                        book_id_selected = raw_val
                    else:
                        raise ValueError(
                            "v27 invalid book reference: "
                            f"{raw_val}"
                        )

                    try:
                        studytype = self.suggest_study_type(
                            book_id_selected,
                            start_page,
                            end_page,
                        )
                    except Exception as exc:
                        logger.warning(
                            "v27 suggest_study_type failed: %s; "
                            "using simple study.",
                            exc,
                        )
                        studytype = 1

                    mean_during_time = average_times.get(7, 0.0416)

                    # A weekend block may be slightly longer, but the common
                    # v27 block cap below is always enforced afterward.
                    if mean_during_time < 0.0834:
                        if weekday in [4, 5]:
                            mean_during_time *= 1.50
                        else:
                            mean_during_time *= 1.10

                    study_priority.register_planned(
                        raw_val,
                        start_page,
                        end_page,
                    )

                except Exception as exc:
                    # v27.2: a mandatory daily study block may never be
                    # silently projected away. Fail the user's transaction
                    # so the command returns a visible non-zero result.
                    if must_meet_study_floor:
                        raise RuntimeError(
                            "v27.2 mandatory daily study scheduling failed "
                            f"for user={CURRENT_USER_ID}, date={date}: {exc}"
                        ) from exc

                    # Outside the mandatory floor, an invalid optional study
                    # proposal is projected to a non-study action.
                    logger.warning(
                        "v27.2 optional study scheduling projected to "
                        "non-study for user=%s: %s",
                        CURRENT_USER_ID,
                        exc,
                    )
                    rl_id = v27_pick_nonstudy_action(probs)
                    best_action_index = rl_id
                    previous_action = rl_id
                    action_index_python = int(
                        self.actiontype_to_index.get(
                            rl_id,
                            action_index_python,
                        )
                    )
                    book_id_selected = 0
                    book_name_str = None
                    start_page = None
                    end_page = None
                    studytype = 0
                    mean_during_time = average_times.get(
                        rl_id,
                        0.0208,
                    )
            # ---------------------------------------------------------
            # ۴. تنظیم زمان‌ها و لاجیک‌های عمومی (مشترک همه فعالیت‌ها)
            # ---------------------------------------------------------
            if mean_during_time < 0:
                mean_during_time = -mean_during_time

            if np.isnan(mean_during_time) or mean_during_time == 0:
                # استفاده از دیکشنری مقادیر پیش‌فرض برای تمام فعالیت‌ها
                defaults = {
                    7: 0.0416, # مطالعه: ۱ ساعت
                    8: 0.0104, # آزمون: ۱۵ دقیقه
                    1: 0.3333, # خواب
                    2: 0.0312, # غذا
                    3: 0.0416, # ورزش
                    4: 0.0416, # سرگرمی
                }
                mean_during_time = defaults.get(rl_id, 0.0208)

            # محدودیت‌های زمانی (مینیمم و ماکزیمم)
            if mean_during_time < 0.0034: # کمتر از ۵ دقیقه
                mean_during_time = 0.0034

            if rl_id not in [1, 7, 8] and mean_during_time > 0.0833:
                mean_during_time = 0.0833

            if rl_id == 7:
                remaining_study_capacity = max(
                    0.0,
                    daily_study_cap_hours
                    - daily_study_planned_hours,
                )
                remaining_clock_hours = max(
                    0.0,
                    (1.0 - current_time) * 24.0,
                )

                # v27.2.1a atomic daily-floor hotfix:
                # The environment may report done after the first study action.
                # Therefore a mandatory study action must satisfy the entire
                # remaining phase floor in this single materialised block,
                # rather than relying on a later loop iteration.
                required_floor_hours = max(
                    0.0,
                    minimum_daily_study_hours
                    - daily_study_planned_hours,
                )
                atomic_floor_request_hours = (
                    required_floor_hours
                    if must_meet_study_floor
                    else 0.0
                )
                proposed_block_hours = max(
                    0.50,
                    float(mean_during_time) * 24.0,
                    atomic_floor_request_hours,
                )
                proposed_block_hours = min(
                    proposed_block_hours,
                    study_block_cap_hours,
                    remaining_study_capacity,
                    remaining_clock_hours,
                )

                if proposed_block_hours <= 0.0:
                    # The action should already have been projected away
                    # above; keep this defensive guard for numerical edges.
                    proposed_block_hours = min(
                        0.25,
                        remaining_clock_hours,
                    )

                mean_during_time = (
                    proposed_block_hours / 24.0
                )

            # نگاشت ID فعالیت به دیتابیس
            if hasattr(self, "rl_id_to_db_id_map") and rl_id in self.rl_id_to_db_id_map:
                db_id_for_storage = self.rl_id_to_db_id_map[rl_id]
            else:
                db_id_for_storage = rl_id

            # ---------------------------------------------------------
            # ۵. بررسی نهایی برای آزمون (جلوگیری از ثبت آزمون ناقص)
            # ---------------------------------------------------------
            if rl_id == 8:
                if not book_name_str and not book_id_selected: # اگر نه اسم هست نه ID
                    logger.warning(f"Ignoring exam for user {CURRENT_USER_ID}: no valid exam suggestion found.")
                    # در حلقه اصلی باید continue باشد، اینجا فرض می‌کنیم کد داخل حلقه است
                    # continue 

            # ---------------------------------------------------------
            # ۶. آماده‌سازی آبجکت کتاب برای ذخیره در دیتابیس
            # ---------------------------------------------------------
            
            # v27.2.1 long-horizon duration quantization:
            # use exactly the same whole-minute duration for internal
            # accumulation, invariant checking, persistence, and simulation.
            if rl_id == 7:
                v2721_duration = _v2721_quantize_study_duration(
                    requested_hours=float(mean_during_time) * 24.0,
                    remaining_daily_cap_hours=max(
                        0.0,
                        daily_study_cap_hours - daily_study_planned_hours,
                    ),
                    block_cap_hours=study_block_cap_hours,
                    remaining_clock_hours=max(
                        0.0,
                        (1.0 - current_time) * 24.0,
                    ),
                )
                mean_during_time = v2721_duration["day_fraction"]
                duration_text_for_storage = v2721_duration["text"]
            else:
                v2721_duration = None
                duration_text_for_storage = self.decimal_to_time(
                    mean_during_time * 24
                )

            schedule_entry = {
                "user": User.objects.get(id=CURRENT_USER_ID),
                "date": date,
                "actiontype": db_id_for_storage,
                "starttime": self.decimal_to_time(current_time * 24),
                "duringtime": duration_text_for_storage,
                # ذخیره نام کتاب (string) بدون ID
                # PostgreSQL CharField را می‌خواهد، نه عددی
                "bookname": book_name_str if book_name_str else None,
                "startpage": int(start_page) if start_page is not None else None,
                "endpage": int(end_page) if end_page is not None else None,
                "typestudy": studytype,
                "typenutrition": typenutrition,
                "typefood": food_type,
                # Internal-only marker used by v27.2 pre-save invariants.
                # save_schedule_to_django intentionally ignores this field.
                "_v272_kind": (
                    "study" if rl_id == 7
                    else "exam" if rl_id == 8
                    else "activity"
                ),
            }
            assert (
                schedule_entry.get("bookname") is None 
                or isinstance(schedule_entry.get("bookname"), str)
            ), f"Invalid bookname: {schedule_entry.get('bookname')}"


            if rl_id == 7:
                # v27.2.1: accumulate the already-serialized whole-minute
                # duration, not the pre-serialization floating-point proposal.
                scheduled_block_hours = float(v2721_duration["hours"])
                daily_study_planned_hours += scheduled_block_hours
                study_block_count += 1

            daily_schedule.append(schedule_entry)
            current_time += mean_during_time
            # آپدیت زمان محیط
            self.env.current_time = current_time
            # صدا زدن step محیط (حتی اگر فقط برای terminated باشه)
            # صدا زدن step محیط (حتی اگر فقط برای done باشه)
            # اگر خطا داد، حداقل break بزن تا حلقه بی‌نهایت نشه
            if current_time >= 1.0:
                break

            # --- مهم: محیط رو هم از تغییر زمان مطلع کن ---
            try:
                if action_index_python >= self.env.n_actions:
                    logger.warning(
                        f"Masked invalid action_index={action_index_python} "
                        f"for user={CURRENT_USER_ID}, n_actions={self.env.n_actions}"
                    )
                    continue
                # به محیط بگو که یه اکشن انجام شده و زمان جلو رفته
                next_state_from_env, reward_env, done, info = self.env.step(action_index_python)
                
                # state رو از محیط بگیر (بدون [ ] اضافی، خود next_state_from_env باید (1, n_features) باشد)
                state = tf.convert_to_tensor(next_state_from_env, dtype=tf.float32)
                
                # اگر روز تموم شد، حلقه رو متوقف کن
                if done or current_time >= 1.0:
                    #print(f"end of running    : {current_time * 24:.1f} time")
                    break
                    
            except Exception as e:
                logger.error(f"خطا در فراخوانی env.step(): {e}")
                # اگر محیط خطا داد، ادامه بده
                break
            # state را برای تصمیم‌گیری در حلقه بعدی آپدیت می‌کنیم
            # ما باید زمان جدید (current_time) را در state بازنویسی کنیم
            
            try:
                # ۱) تبدیل state به numpy برای آپدیت راحت‌تر
                if tf.is_tensor(state):
                    state_np = state.numpy()
                else:
                    state_np = np.array(state, dtype=np.float32)

                # ۲) اطمینان از این‌که state دو بعدی است: (1, n_features)
                if state_np.ndim == 1:
                    state_np = state_np[np.newaxis, :]
                elif state_np.ndim == 3 and state_np.shape[0] == 1:
                    # اگر سه‌بعدی مثل (1,1,16) بود → به (1,16) تبدیلش کن
                    state_np = state_np.reshape(1, state_np.shape[-1])

                # ۳) اندیس 5 همان starttime در state است (date=0, actiontype=1, bookname=2, startpage=3, endpage=4, starttime=5)
                state_np[0, 5] = current_time

                # ۴) برگرداندن به تنسور TF با شکل (1, n_features)
                state = tf.convert_to_tensor(state_np, dtype=tf.float32)

            except Exception as e:
                logger.error(f"خطا در به‌روزرسانی state: {e}")                

        # v27.2 final scheduler invariants. These checks run before any
        # DailySchedule row is written. A planning error therefore fails
        # atomically instead of silently contaminating the database.
        def _v27_duration_hours(value):
            if value is None:
                return 0.0
            if isinstance(value, (int, float, np.integer, np.floating)):
                return max(0.0, float(value))
            text = str(value).strip()
            if not text:
                return 0.0
            if ":" in text:
                parts = text.split(":")
                hours = float(parts[0] or 0)
                minutes = float(parts[1] or 0) if len(parts) > 1 else 0.0
                seconds = float(parts[2] or 0) if len(parts) > 2 else 0.0
                return max(0.0, hours + minutes / 60.0 + seconds / 3600.0)
            return max(0.0, float(text))

        def _v272_is_study_entry(entry):
            if entry.get("_v272_kind") == "study":
                return True

            action_value = entry.get("actiontype")
            if action_value in ("__STUDY__", 7, "7", "مطالعه"):
                return True

            # Defensive fallback for legacy materialisation paths.
            return (
                entry.get("bookname") is not None
                and entry.get("startpage") is not None
                and entry.get("endpage") is not None
                and action_value not in ("__EXAM__", 8, "8", "آزمون")
            )

        v27_study_entries = [
            entry
            for entry in daily_schedule
            if _v272_is_study_entry(entry)
        ]
        v27_study_keys = [
            (
                str(entry.get("bookname") or ""),
                int(entry.get("startpage") or 0),
                int(entry.get("endpage") or 0),
            )
            for entry in v27_study_entries
        ]
        v27_duplicate_count = (
            len(v27_study_keys) - len(set(v27_study_keys))
        )
        v27_study_hours = [
            _v27_duration_hours(entry.get("duringtime"))
            for entry in v27_study_entries
        ]
        v27_total_study_hours = float(sum(v27_study_hours))
        v27_over_block_count = sum(
            hours > study_block_cap_hours + 1.0e-6
            for hours in v27_study_hours
        )

        invariant_errors = []
        if not v27_study_entries:
            invariant_errors.append("missing_daily_study=1")
        if v27_duplicate_count:
            invariant_errors.append(
                f"duplicate_study_ranges={v27_duplicate_count}"
            )
        if len(v27_study_entries) > maximum_study_blocks:
            invariant_errors.append(
                "study_blocks="
                f"{len(v27_study_entries)}>{maximum_study_blocks}"
            )
        if v27_over_block_count:
            invariant_errors.append(
                f"blocks_above_cap={v27_over_block_count}"
            )
        if v27_total_study_hours > daily_study_cap_hours + 1.0e-6:
            invariant_errors.append(
                "daily_study_hours="
                f"{v27_total_study_hours:.3f}>"
                f"{daily_study_cap_hours:.3f}"
            )
        if (
            minimum_daily_study_hours > 0.0
            and v27_total_study_hours + 1.0e-6
            < minimum_daily_study_hours
        ):
            invariant_errors.append(
                "study_floor="
                f"{v27_total_study_hours:.3f}<"
                f"{minimum_daily_study_hours:.3f}"
            )

        if invariant_errors:
            raise RuntimeError(
                "v27.2 scheduler invariant failure for user="
                f"{CURRENT_USER_ID}, date={date}: "
                + "; ".join(invariant_errors)
            )

        logger.info(
            "v27.2 scheduler invariants passed | user=%s | date=%s | "
            "study_blocks=%s | study_hours=%.3f | projections=%s",
            CURRENT_USER_ID,
            date,
            len(v27_study_entries),
            v27_total_study_hours,
            safety_projection_count,
        )

        # ذخیره برنامه روزانه در فایل 
        self.save_schedule_to_django(daily_schedule)
        
        return daily_schedule
    
    #----تابع جدید ذخیره برنامه روزانه
    def save_schedule_to_django(self, daily_schedule):
        from django.utils import timezone

        def time_string_to_float(time_str):
            #تبدیل رشته وردی به زمان 
            """
            تبدیل رشته زمانی (مثل '7:59' یا '0:45') به عدد اعشاری (به ساعت).
            مثال: '7:59' -> 7.9833, '2:30' -> 2.5, '0:45' -> 0.75
            """
            try:
                if isinstance(time_str, (int, float)):
                    return float(time_str)
                time_str = time_str.strip()
                if ':' in time_str:
                    hours, minutes = map(int, time_str.split(':'))
                    return hours + (minutes / 60.0)
                else:
                    minutes = int(time_str)
                    return minutes / 60.0
            except (ValueError, AttributeError) as e:
                logger.error(f"error in time '{time_str}': {e}", exc_info=True)
                return 0.0
        
        
        #============تابع تبدیل نوع مطالعه =========================
        def reverse_typestudy(typestudy_index):
            '''
            تبدیل عدد نوع مطالعه به نام نوع مطالعه متنی برای ذخیره در دیتابیس یا نمایش.

            پارامتر:
                typestudy_index (int | float | str): عدد نوع مطالعه از خروجی الگوریتم

            خروجی:
                str: نام نوع مطالعه متنی (مثلاً "مطالعه ساده"، "حل تست"، ...)
            '''
            fixed_typestudy_map = {
                1: "مطالعه ساده",
                2: "مطالعه با خلاصه نویسی",
                3: "مطالعه با حل مساله",
                4: "حل مساله",
                5: "حل تست",
                6: "مطالعه خلاصه مطالب",
                7: "بررسی آزمون های داده شده"
            }

            try:
                typestudy_index = int(float(typestudy_index))  # تبدیل ایمن از str یا float به int
                logger.info(f"در حال تبدیل عدد نوع مطالعه به نام: ورودی={typestudy_index}")
                study_type_name = fixed_typestudy_map.get(typestudy_index)
            except (ValueError, TypeError) as e:
                logger.warning(f"نوع مطالعه نامعتبر: {typestudy_index}، خطا: {e}")
                return "0"

            if study_type_name is None:
                logger.warning(f"عدد نوع مطالعه نامعتبر دریافت شد: {typestudy_index}")
                logger.debug(f"نگاشت نوع مطالعه: {fixed_typestudy_map}")
                return "0"

            logger.info(f"نام نوع مطالعه مربوط به عدد {typestudy_index}: {study_type_name}")
            return study_type_name
        # ====================پایان تابع تبدیل نوع مطالعه ====================      
        def parse_time_to_hour_minute(starttime_value):
            try:
                # بخش عددی (float/int): بهبود با استفاده از timedelta برای دقت بیشتر
                if isinstance(starttime_value, (float, int)):
                    total_hours = float(starttime_value)
                    hours = int(total_hours)
                    fractional_hour = total_hours - hours
                    minutes = int(round(fractional_hour * 60))  # rounding حفظ شود

                    # چک overflow دقیقه
                    if minutes >= 60:
                        hours += 1
                        minutes = 0  # یا minutes -= 60

                    # چک overflow ساعت
                    if hours >= 24:
                        hours = 23
                        minutes = 59

                    return time(hours, minutes)

                # بخش رشته‌ای: بهبود با validate پیشین
                elif isinstance(starttime_value, str):
                    starttime_value = starttime_value.strip()

                    # split برای چک دستی قبل از parse
                    parts = starttime_value.split(':')
                    if len(parts) < 2 or len(parts) > 3:
                        raise ValueError(f"فرمت نامعتبر زمان: {starttime_value}")

                    hour_str = parts[0]
                    minute_str = parts[1]
                    second_str = parts[2] if len(parts) == 3 else '00'  # اگر ثانیه نبود، ۰۰

                    # تبدیل به int و چک معتبر بودن
                    try:
                        hour = int(hour_str)
                        minute = int(minute_str)
                        second = int(second_str)
                    except ValueError:
                        raise ValueError(f"اعداد نامعتبر در زمان: {starttime_value}")

                    # چک دستی برای دقیقه و ثانیه
                    if minute >= 60:
                        hour += 1  # افزایش ساعت
                        minute = 0  # reset دقیقه
                        logger.warning(f"دقیقه نامعتبر ({minute_str}) در {starttime_value} تنظیم شد به {hour}:{minute:02d}")

                    if second >= 60:
                        minute += 1
                        second = 0
                        if minute >= 60:  # chain overflow
                            hour += 1
                            minute = 0

                    if hour >= 24:
                        hour = 23
                        minute = 59
                        second = 0

                    # حالا رشته معتبر بساز و parse کن
                    valid_time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
                    dt = datetime.strptime(valid_time_str, '%H:%M:%S')
                    return dt.time()  # بدون second اگر لازم نیست

                else:
                    raise TypeError(f"نوع نامعتبر ورودی زمان: {type(starttime_value)}")

            except Exception as e:
                logger.error(f"خطا در تبدیل زمان '{starttime_value}': {e}")
                return None #time(0, 0)   پیش‌فرض
                    
        user_obj = self.env.user 
        for entry in daily_schedule:            
            # 🌟 خطای حاد: اگر DailySchedule.objects.create به user=entry["user"] نیاز دارد،
            # باید مطمئن شوید که entry["user"] در خروجی الگوریتم وجود دارد. 
            
            try:
                # 1. تبدیل date
                date_value = entry['date']
                if isinstance(date_value, str):
                    date_value = datetime.strptime(date_value, '%Y-%m-%d').date()

                # 2. تبدیل starttime
                starttime_value = entry['starttime']
                starttime_value = parse_time_to_hour_minute(starttime_value)
                if isinstance(starttime_value, str):
                    starttime_value = datetime.strptime(starttime_value, '%H:%M').time()

                # 3. تبدیل duringtime
                duringtime_value = entry['duringtime']
                if isinstance(duringtime_value, str):
                    duringtime_value = time_string_to_float(duringtime_value)
                else:
                    duringtime_value = float(duringtime_value)

                # بدون تبدیل bookname - برای PostgreSQL باید string باشد
                bookname_value = entry.get('bookname')
                if bookname_value is not None and bookname_value != '':
                    bookname_value = str(bookname_value).strip()
                else:
                    bookname_value = None
                
                # 6. استخراج rl_id از actiontype برای تعیین نوع فعالیت
                # این مرحله باید قبل از تبدیل typestudy انجام شود
                actiontype_index = entry.get('actiontype')
                rl_id = None
                if actiontype_index == '__STUDY__':
                    rl_id = 7
                elif actiontype_index == '__EXAM__':
                    rl_id = 8
                else:
                    try:
                        rl_id = int(float(actiontype_index))
                    except (ValueError, TypeError):
                        rl_id = None
                
                # 5. تبدیل typestudy - فقط برای فعالیت مطالعه و آزمون
                typestudy_raw = entry.get('typestudy')
                typestudy_value = None  # مقدار پیش‌فرض: خالی برای سایر فعالیت‌ها
                
                if rl_id == 7:  # فعالیت مطالعه
                    # برای مطالعه، اگر typestudy عدد بود، نام آن را بگیر
                    if typestudy_raw is not None:
                        try:
                            # تلاش برای تبدیل به عدد
                            typestudy_index = int(float(typestudy_raw))
                            # استفاده از تابع reverse_typestudy برای تبدیل به نام
                            typestudy_value = reverse_typestudy(typestudy_index)
                            # اگر مقدار غیرمعتبر بود در "0" برگشت، تبدیل به None
                            if typestudy_value == "0":
                                typestudy_value = "مطالعه ساده"  # پیش‌فرض معقول
                        except (ValueError, TypeError):
                            # اگر عدد نبود، شاید از قبل نام است
                            typestudy_value = str(typestudy_raw).strip()
                    else:
                        typestudy_value = "مطالعه ساده"  # مقدار پیش‌فرض برای مطالعه: نام نه عدد
                
                elif rl_id == 8:  # فعالیت آزمون
                    typestudy_value = "آزمون"
                else:
                    # برای سایر فعالیت‌ها، typestudy خالی می‌ماند
                    typestudy_value = None

                # 7. تبدیل actiontype (فعالیت)
                actiontype_index = entry.get('actiontype')
                if actiontype_index == '__STUDY__':
                    actiontype_value = "مطالعه"
                elif actiontype_index == '__EXAM__':
                    actiontype_value = "آزمون"
                else:
                    try:
                        rl_id_check = int(float(actiontype_index))
                        if rl_id_check >= 9:
                            db_id_for_storage = rl_id_to_db_id_map[rl_id_check]
                        elif rl_id_check in (7, 8):
                            db_id_for_storage = rl_id_check  # استفاده از خود کد RL برای مطالعه/آزمون
                        else:
                            db_id_for_storage = rl_id_check  # برای ۱-۶ نیز خودشان
                        #actiontype_value = reverse_activities(rl_id, user_obj)  # بدون مقدار پیش‌فرض!
                    except Exception as e:
                        logger.warning(f"Invalid actiontype_index: {actiontype_index}, error: {e}")
                        actiontype_value = "فعالیتs نامعتبر"
                        
                # 8. تبدیل typenutrition
                raw_nutrition_value = entry.get('typenutrition')
                nutrition_map = {1: "صبحانه", 2: "نهار", 3: "شام", 4: "میان وعده"}
                nutrition_key = None
                try:
                    nutrition_key = int(float(raw_nutrition_value))
                except (ValueError, TypeError):
                    pass
                typenutrition_value = nutrition_map.get(nutrition_key, "0")
                # دیکشنری تبدیل نوع غذا
                food_type_map = {
                    1: "غذاهای پروتئینی",
                    2: "غذاهای شیرین",
                    3: "لبنیات",
                    4: "غذاهای کامل",
                    5: "میوه",
                    6: "سایر"
                }

                # فرض بر این که row یا entry دادهٔ جاری است:
                raw_typefood_value = entry.get('typefood')
                try:
                    typefood_key = int(float(raw_typefood_value))
                except (ValueError, TypeError):
                    typefood_key = None

                typefood_value = food_type_map.get(typefood_key, "0")

                action_code = entry['actiontype']
        
                # 🔥 اصلاح ۳: فراخوانی متد جدید SAC 🔥
                action_name = self.get_activity_name(action_code, user_obj)
                
                # تبدیل startpage و endpage به عدد صحیح
                startpage_value = None
                endpage_value = None
                try:
                    raw_startpage = entry.get('startpage')
                    if raw_startpage is not None:
                        startpage_value = int(float(raw_startpage)) if isinstance(raw_startpage, (str, float)) else int(raw_startpage)
                except (ValueError, TypeError):
                    startpage_value = None
                    
                try:
                    raw_endpage = entry.get('endpage')
                    if raw_endpage is not None:
                        endpage_value = int(float(raw_endpage)) if isinstance(raw_endpage, (str, float)) else int(raw_endpage)
                except (ValueError, TypeError):
                    endpage_value = None
                
                # 🌟🌟🌟 ذخیره‌سازی نهایی در مدل DailySchedule 🌟🌟🌟
                DailySchedule.objects.create(
                    user=user_obj, # استفاده از آبجکت user که از محیط گرفته شده
                    date=date_value,
                    actiontype=action_name,
                    starttime=starttime_value,
                    duringtime=duringtime_value,
                    bookname=bookname_value, 
                    startpage=startpage_value,
                    endpage=endpage_value,
                    typestudy=typestudy_value,
                    typenutrition=typenutrition_value,
                    typefood=typefood_value, #entry.get('typefood'),
                )
                logger.info("DailySchedule entry successfully created.")
                
            except Exception as e:
                # 🌟 تغییر: به جای continue و raise، فقط خطا را لاگ می‌کنیم تا سایر رکوردها اجرا شوند.
                logger.error(f"Error in saving DailySchedule entry for user {user_obj.id}: {str(e)}", exc_info=True)


        
    # استفاده از کلاس SAC
if __name__ == "__main__":
    activity_file = dfstat 
    exam_file =create_dfexam(user=User.objects.get(id=CURRENT_USER_ID)) 
    # خواندن داده‌های فایل آزمون
    exam_data = create_dfexam(user=User.objects.get(id=CURRENT_USER_ID)) 
    exam_data = exam_data.fillna(0)  
    df = activity_file #create_dfstat(user=User.objects.get(id=CURRENT_USER_ID))#dfstat 
    # 0. ابتدا باید داده‌های خام را بسازیم (چون در این بلوک موجود نیستند)
    # اگر این کد را برای تست در if __name__ اجرا می‌کنید، باید این خطوط را اضافه کنید:
    df_test = create_dfstat(user=User.objects.get(id=CURRENT_USER_ID))
    exam_file_test = create_dfexam(user=User.objects.get(id=CURRENT_USER_ID))
    # 🔥 خطوط محاسبه min_date_of_all_data باید اینجا باشند 🔥
    try:
        min_study_date = study_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
        min_activity_date = Faliat_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
        min_exam_date = azmoon_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
        
        valid_dates = list(filter(None, [min_study_date, min_activity_date, min_exam_date]))
        if not valid_dates:
            min_date_of_all_data = None
        else:
            min_date_of_all_data = min(valid_dates)
            
    except Exception as e:
        # در صورت خطا در پایگاه داده، یک تاریخ پیش‌فرض برای تست تنظیم کنید
        logger.error(f"Error calculating min_date in test block: {e}", exc_info=True)
        min_date_of_all_data = None
    # --- ابتدا پاداش را محاسبه کنید ---
    FINAL_DF = prepare_rl_data(
        df_stat=df_test,             
        df_exam=exam_file_test,     
        user=User.objects.get(id=CURRENT_USER_ID), 
        min_date=min_date_of_all_data, 
        max_reward_exam=1500.0, 
        max_reward_final_scale=1.0 
    )
    
    min_study_date = study_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
    min_activity_date = Faliat_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
    min_exam_date = azmoon_model.objects.filter(user=user).aggregate(min_date=Min('tarikh_en_user'))['min_date']
    valid_dates = list(filter(None, [min_study_date, min_activity_date, min_exam_date]))
    min_date_of_all_data = min(valid_dates) if valid_dates else None

    # --- سپس محیط را با داده‌های آماده شده بسازید ---
    study_env = StudyEnvironment(
        df_final_data=FINAL_DF,
        df_stat_features=df_test,
        df_exam_features=exam_file_test,
        min_date=min_date_of_all_data
    )
    # استخراج تعداد سطرها
    num_rows = df.shape[0]
    # تبدیل داده‌ها به لیست یا استفاده مستقیم
    exam_types = exam_data['exam_type'].tolist()
    scores = exam_data['score'].tolist()
    date1 = exam_data['date']
    book_name = exam_data['bookname']
    start_page = exam_data['startpage']

    previous_scores, time_gaps =get_previous_scoresac(date1, book_name, start_page)
    # خواندن تعداد رکوردهای فایل
    data = dfstat
    num_episodes = 500 #len(data)//2    بررسی شود تعداد رکوردها برابر با تعداد اپیزودها

    num_actions = df['actiontype'].nunique() # تعداد عمل موجود در ستون نوع عمل دیتاست می خواند
    #  استفاده از کلاس SAC
    num_columns = df.shape[1]  # تعداد ستون‌ها را بدست می‌آوریم
    state_dim = (num_columns,) # تعداد ستون های دیتاست استیت ویژگی حالت
    action_dim =num_actions # (num_actions,)  # از دیتاست می خواند عمل ممکن
    max_action = 1  # حداکثر مقدار عمل

    sac_agent = SAC(state_dim, action_dim, max_action, study_env)
    # فقط زمانی که امتحان داده شده و نوع امتحان مخالف صفر است حلقه آموزش مدل اجرا شود
    filtered_exam_data = exam_data[exam_data['exam_type'] != 0]
    # حلقه برای آموزش مدلi in range(len(exam_types)):
    for _, row in filtered_exam_data.iterrows():
        # استفاده از مقدار iام از exam_types و scores
        current_exam_type = row['exam_type']

        current_score = row['score']
        current_date = row['date']
        current_book_name = row['bookname']
        current_start_page = row['startpage']
        # دریافت نمرات قبلی و زمان گپ مربوط به همان رکورد
        current_previous_score, current_time_gap = get_previous_scoresac(current_date, current_book_name, current_start_page)
        # آموزش مدل با استفاده از مقادیر جاری
        sac_agent.train(num_episodes, current_exam_type, current_score, current_previous_score, current_time_gap)

    # ذخیره برنامه روزانه
    daily_schedule = sac_agent.generate_daily_schedule()  # اصلاح شده

    
    # اجرای محیط
    study_env.run()


# ================= DMA-DSAC v27.2 audited runtime patch ================
from .dma_dsac_v27_2_runtime import apply_patch as _apply_dma_dsac_v27_2_patch
_apply_dma_dsac_v27_2_patch(globals(), mode="full")
