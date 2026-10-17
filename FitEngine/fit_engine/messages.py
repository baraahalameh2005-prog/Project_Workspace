"""Bilingual (Arabic / English) message templates.

Every message the engine emits is a dict:
    {"code": <stable machine code>, "message_ar": ..., "message_en": ...}
Other systems should branch on "code", never on the text.
"""

STATUS_LABELS = {
    "good": ("مناسب", "good fit"),
    "slightly_tight": ("ضيق قليلًا", "slightly tight"),
    "too_tight": ("ضيق", "too tight"),
    "slightly_loose": ("واسع قليلًا", "slightly loose"),
    "too_loose": ("واسع", "too loose"),
    "slightly_short": ("قصير قليلًا", "slightly short"),
    "too_short": ("قصير", "too short"),
    "slightly_long": ("طويل قليلًا", "slightly long"),
    "too_long": ("طويل", "too long"),
    "missing_body": ("قياس الجسم غير متوفر", "body measurement missing"),
    "missing_garment": ("قياس القطعة غير متوفر لهذا المقاس", "garment measurement missing for this size"),
}

OVERALL_LABELS = {
    "good": ("مناسب", "good fit"),
    "tight": ("ضيق", "tight"),
    "loose": ("واسع", "loose"),
    "mixed": ("ضيق في مناطق وواسع في أخرى", "tight in some areas, loose in others"),
    "insufficient_data": ("معلومات غير كافية", "insufficient data"),
}

_T = {
    # ---- validation errors / warnings ----
    "invalid_document": (
        "البيانات المرسلة في '{where}' يجب أن تكون كائن JSON (dict).",
        "'{where}' must be a JSON object (dict).",
    ),
    "invalid_number": (
        "القيمة '{value}' في {where} ليست رقمًا صالحًا.",
        "Value '{value}' in {where} is not a valid number.",
    ),
    "non_positive": (
        "القيمة {value} في {where} يجب أن تكون أكبر من صفر.",
        "Value {value} in {where} must be greater than zero.",
    ),
    "out_of_range": (
        "القيمة {value_cm} سم في {where} خارج النطاق المعقول ({lo}–{hi} سم).{hint_ar}",
        "Value {value_cm} cm in {where} is outside the plausible range ({lo}-{hi} cm).{hint_en}",
    ),
    "unknown_unit": (
        "الوحدة '{unit}' في {where} غير مدعومة. الوحدات المدعومة: cm, mm, m, in.",
        "Unit '{unit}' in {where} is not supported. Supported: cm, mm, m, in.",
    ),
    "missing_unit": (
        "لم تُحدَّد وحدة القياس في {where}؛ تم افتراض السنتيمتر.",
        "No unit given in {where}; centimetres assumed.",
    ),
    "unknown_measurement": (
        "القياس '{key}' في {where} غير معروف للنظام وتم تجاهله.",
        "Measurement '{key}' in {where} is unknown and was ignored.",
    ),
    "alias_used": (
        "تم تحويل اسم القياس '{alias}' إلى الاسم الموحّد '{key}' في {where}.",
        "Measurement name '{alias}' was mapped to canonical '{key}' in {where}.",
    ),
    "duplicate_measurement": (
        "القياس '{key}' مكرر في {where} (بأسماء مختلفة)؛ تم اعتماد أول قيمة.",
        "Measurement '{key}' appears twice in {where}; the first value was kept.",
    ),
    "weight_ignored": (
        "الوزن لا يُستخدم لتحديد المقاس؛ النظام يعتمد على قياسات الجسم فقط.",
        "Weight is not used to choose a size; only body measurements are used.",
    ),
    "possible_flat_measurement": (
        "{name_ar} في المقاس {size} ({garment} سم) أصغر بكثير من الجسم ({body} سم). "
        "هل القياسات نصف محيط (مسطّحة)؟ إن كانت كذلك استخدم measurement_type = flat.",
        "{name_en} of size {size} ({garment} cm) is far smaller than the body ({body} cm). "
        "Are these half (flat) measurements? If so use measurement_type = flat.",
    ),
    "unusual_proportions": (
        "نِسب القياسات غير معتادة ({detail}). تأكد من صحة القياسات.",
        "Unusual proportions ({detail}). Please double-check the measurements.",
    ),
    "no_body_measurements": (
        "لا توجد أي قياسات صالحة لجسم المستخدم.",
        "No valid body measurements were provided.",
    ),
    "missing_category": (
        "نوع قطعة الملابس (category) غير محدد. الأنواع المدعومة: {supported}.",
        "Garment category is missing. Supported: {supported}.",
    ),
    "unknown_category": (
        "نوع الملابس '{value}' غير مدعوم. الأنواع المدعومة: {supported}.",
        "Garment category '{value}' is not supported. Supported: {supported}.",
    ),
    "unknown_fit_type": (
        "القصة '{value}' غير معروفة؛ تم اعتماد القصة العادية (regular).",
        "Fit type '{value}' is unknown; 'regular' was used.",
    ),
    "fit_type_default": (
        "قصة الملابس غير محددة؛ تم افتراض القصة العادية (regular)، وقد تختلف الملاءمة الفعلية.",
        "Fit type not given; 'regular' assumed, real fit may differ.",
    ),
    "unknown_stretch": (
        "درجة مرونة القماش '{value}' غير معروفة؛ تم افتراض قماش غير مطاطي.",
        "Stretch level '{value}' is unknown; non-stretch fabric assumed.",
    ),
    "stretch_default": (
        "مرونة القماش غير محددة؛ تم افتراض قماش غير مطاطي، وقد تختلف الملاءمة الفعلية.",
        "Fabric stretch not given; non-stretch assumed, real fit may differ.",
    ),
    "unknown_measurement_type": (
        "طريقة قياس القطعة '{value}' غير معروفة. القيم المدعومة: circumference, flat.",
        "Garment measurement_type '{value}' is unknown. Supported: circumference, flat.",
    ),
    "no_sizes": (
        "لا توجد مقاسات للقطعة.",
        "The garment has no sizes.",
    ),
    "size_without_label": (
        "المقاس رقم {index} لا يحتوي على اسم (label).",
        "Size #{index} has no label.",
    ),
    "duplicate_size": (
        "اسم المقاس '{label}' مكرر.",
        "Size label '{label}' is duplicated.",
    ),
    "size_no_measurements": (
        "المقاس '{label}' لا يحتوي على أي قياسات صالحة.",
        "Size '{label}' has no valid measurements.",
    ),
    "invalid_ease_override": (
        "قيمة ease_overrides للقياس '{key}' يجب أن تكون [أدنى، أعلى] بالسنتيمتر.",
        "ease_overrides for '{key}' must be [min, max] in cm.",
    ),
    "unknown_ease_override": (
        "ease_overrides يحتوي على قياس غير معروف '{key}' وتم تجاهله.",
        "ease_overrides contains unknown measurement '{key}'; it was ignored.",
    ),
    "incomplete_size": (
        "المقاس {size} ينقصه قياس القطعة لـ: {keys_ar}، لذلك لم يُقيَّم بالكامل ولا يمكن التوصية به.",
        "Size {size} is missing garment data for: {keys_en}, so it could not be fully evaluated "
        "and cannot be recommended.",
    ),
    "incomplete_garment_data": (
        "لا يمكن إصدار توصية موثوقة: المقاسات التي قد تناسب ({sizes}) تنقصها قياسات أساسية للقطعة، "
        "والمقاسات المكتملة البيانات لا تناسب. يجب استكمال بيانات المقاسات من نظام الملابس.",
        "Cannot give a reliable recommendation: the sizes that might fit ({sizes}) are missing key "
        "garment measurements, and the fully measured sizes do not fit. Complete the size data first.",
    ),
    # ---- result reasons ----
    "reason_all_good": (
        "المقاس {size}: كل القياسات المقارنة ضمن نطاق الراحة المطلوب لهذه القصة.",
        "Size {size}: every compared measurement is within the target ease for this fit.",
    ),
    "reason_best_acceptable": (
        "المقاس {size} هو الأقرب لنطاق الراحة المطلوب من بين المقاسات القابلة للارتداء.",
        "Size {size} is the closest to the target ease among wearable sizes.",
    ),
    "reason_primary_ok": (
        "القياسات الأساسية لهذا النوع ({keys_ar}) لا تمنع ارتداء المقاس {size}.",
        "The key measurements for this garment type ({keys_en}) do not rule out size {size}.",
    ),
    "issue_width": (
        "{name_ar}: القطعة أكبر من الجسم بـ {ease} سم ({label_ar})، والمطلوب بين {lo} و{hi} سم.",
        "{name_en}: garment exceeds body by {ease} cm ({label_en}); target is {lo} to {hi} cm.",
    ),
    "issue_length": (
        "{name_ar}: الفرق {ease} سم ({label_ar})، والمطلوب بين {lo} و{hi} سم.",
        "{name_en}: difference {ease} cm ({label_en}); target is {lo} to {hi} cm.",
    ),
    "all_too_small": (
        "كل المقاسات المتوفرة أضيق من المطلوب. أقرب مقاس هو {size}، لكنه ضيق في: {keys_ar}.",
        "All available sizes are too small. Closest is {size}, but it is tight at: {keys_en}.",
    ),
    "all_too_large": (
        "كل المقاسات المتوفرة أوسع من المطلوب. أقرب مقاس هو {size}، لكنه واسع في: {keys_ar}.",
        "All available sizes are too large. Closest is {size}, but it is loose at: {keys_en}.",
    ),
    "no_single_size": (
        "لا يوجد مقاس واحد يناسب كل المناطق. {detail_ar}",
        "No single size fits every area. {detail_en}",
    ),
    "insufficient_primary": (
        "لا يمكن إصدار توصية موثوقة: القياسات الأساسية لهذا النوع ({keys_ar}) غير متوفرة للمقارنة.",
        "Cannot give a reliable recommendation: key measurements for this type ({keys_en}) are unavailable.",
    ),
    "missing_primary_partial": (
        "بعض القياسات الأساسية غير متوفرة ({keys_ar})، لذلك التوصية أقل موثوقية.",
        "Some key measurements are missing ({keys_en}), so the recommendation is less reliable.",
    ),
    "missing_secondary": (
        "قياسات ثانوية غير متوفرة ولم تدخل في المقارنة: {keys_ar}.",
        "Secondary measurements not available and not compared: {keys_en}.",
    ),
    "not_compared": (
        "{name_ar} للقطعة لا يقابله قياس جسم مباشر؛ يُعرض للمعلومة فقط.",
        "Garment {name_en} has no direct body counterpart; shown for information only.",
    ),
    "not_relevant": (
        "{name_ar} غير مهم لتحديد المقاس في هذا النوع من الملابس ولم يدخل في المقارنة.",
        "{name_en} does not drive sizing for this garment type and was not compared.",
    ),
    "alt_larger": (
        "المقاس {size} بديل لمن يفضّل قصة أوسع وأكثر راحة.",
        "Size {size} is an alternative for a roomier, more relaxed fit.",
    ),
    "alt_smaller": (
        "المقاس {size} بديل لمن يفضّل قصة أكثر إحكامًا.",
        "Size {size} is an alternative for a closer fit.",
    ),
    "alt_fixes_tight": (
        "المقاس {size} قد يكون أريح لأن المقاس الموصى به ضيق قليلًا في: {keys_ar}.",
        "Size {size} may be more comfortable because the recommended size is slightly tight at: {keys_en}.",
    ),
    "between_sizes": (
        "قياساتك تقع بين المقاسين {a} و{b}.",
        "Your measurements fall between sizes {a} and {b}.",
    ),
    "stretch_note": (
        "القماش مطاطي ({stretch})، لذلك يُسمح بأن تكون القطعة أصغر من الجسم بحدود {pct}% في المحيطات.",
        "Fabric is stretchy ({stretch}); garment circumferences may be up to {pct}% smaller than the body.",
    ),
    "ease_override_used": (
        "تم استخدام نطاق راحة مخصص من بيانات المنتج للقياس: {keys_ar}.",
        "Product-specific ease range used for: {keys_en}.",
    ),
    "disclaimer": (
        "هذه توصية تقديرية مبنية على مقارنة رقمية للقياسات ونطاقات راحة افتراضية، "
        "ولم يتم التحقق من دقتها على ملابس حقيقية بعد. لا تضمن ملاءمة مثالية.",
        "This is an estimate based on numeric comparison and default ease ranges. "
        "It has not been validated on real garments yet and does not guarantee a perfect fit.",
    ),
}


def msg(code, **kw):
    ar, en = _T[code]
    return {"code": code, "message_ar": ar.format(**kw), "message_en": en.format(**kw)}


def fmt(x):
    """Format a cm number compactly: 4.0 -> '4', 3.25 -> '3.3', signed for ease."""
    r = round(float(x), 1)
    if r == int(r):
        return str(int(r))
    return f"{r:.1f}"


def fmt_signed(x):
    r = round(float(x), 1)
    s = fmt(r)
    return s if r < 0 else ("+" + s if r > 0 else "0")
