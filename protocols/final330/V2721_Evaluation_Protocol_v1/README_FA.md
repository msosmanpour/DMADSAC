# V27.2.1 Evaluation Protocol v1

این بسته Scheduler را تغییر نمی‌دهد. هدف آن قفل‌کردن ارزیابی استاندارد و جلوگیری از آلودگی Trace آزمون‌های طبیعی است.

## فایل‌ها

- `standardized_exam_v1.py`
- `validate_natural_exam_trace_v1.py`
- `EVALUATION_PROTOCOL_LOCK_CANDIDATE.json`
- `PACKAGE_SHA256SUMS.txt`

## پیامد اصلی

`equal_book_expected_recall_score` در دامنه ثابت `anchor`، با وزن مساوی برای هر ۹ کتاب.

ویژگی‌ها:

- ۱۰ سؤال قطعی برای هر کتاب
- صفحات سؤال یکسان در Full و Static
- پیش‌آزمون یکسان
- بدون رضایت
- بدون استفاده از آزمون‌های قبلی
- محتوای مطالعه‌نشده برابر صفر
- بدون Noise در پیامد اصلی
- نمره چندگزینه‌ای تحقق‌یافته فقط به‌عنوان حساسیت، با عدد تصادفی مشترک جفت‌شده

## پروتکل‌ها

- `gate30`: PRE و D30
- `gate60`: PRE، D30 و D60
- `final330`: PRE، D110، D220 و D330

## Trace طبیعی

قبل از هر Run تازه:

```powershell
python validate_natural_exam_trace_v1.py --mode preflight --trace <path>
```

پس از Run:

```powershell
python validate_natural_exam_trace_v1.py --mode postrun --trace <path> --db <db> --start-date 2025-07-26 --days 30 --from-user 1 --to-user 100
```

Clone باید تازه باشد و `paper_exports_final\exam_scoring_trace.csv` از Run قبلی نداشته باشد.
