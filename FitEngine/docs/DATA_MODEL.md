# عقد البيانات (Data Contract) — fit_engine v0.2.0

هذا المستند هو المرجع لشكل المدخلات والمخرجات عند الدمج المستقبلي مع موديول الأفاتار/قياسات الجسم (براءه) وموديول الملابس (سجى).
**لم يُنفَّذ أي دمج بعد.** الحقول المعلّمة بـ 🔶 يعتمد تحديدها النهائي على ما تنتجه موديولات الزميلتين فعلًا.

- الوحدة الأساسية داخل المحرك: **سنتيمتر**. أي وحدة مدعومة أخرى تُحوَّل تلقائيًا.
- كل أسماء الحقول بالإنجليزي، أحرف صغيرة، `snake_case`.
- الأنظمة الأخرى يجب أن تعتمد على حقول `status` و`code`، **وليس على نص الرسائل**.

## 0. ثلاثة أنواع من القياسات — لا تُخلط

| النوع | ماذا يصف | مثال | من أين يأتي |
|---|---|---|---|
| **محيط الجسم** | الشريط حول جسم الشخص نفسه | صدر الشخص = 92 سم | الأفاتار أو إدخال المستخدم |
| **محيط القطعة** | الشريط حول القطعة المصنوعة (من الداخل) | صدر التيشيرت M = 100 سم | نظام الملابس، `measurement_type: "circumference"` |
| **عرض القطعة المسطّح** | القطعة ممدودة على طاولة، القياس من طرف لطرف | عرض صدر التيشيرت M = 50 سم | نظام الملابس، `measurement_type: "flat"` |

قواعد التحويل في المحرك:
- عرض مسطّح ← محيط القطعة: **× 2** للمحيطات فقط (`chest`, `waist`, `hip`, `thigh`, `sleeve_opening`). الافتراض: القطعة أنبوب بطبقتين متساويتين؛ قد يختلف قليلًا مع الطيات أو الخياطة غير المتماثلة.
- `shoulder_width` والأطوال (`sleeve_length`, `inseam`, `length`) **عرض/طول** وليست محيطًا، فلا تُضاعف أبدًا.
- محيط الجسم ≠ محيط القطعة: لا يُقارنان بالتساوي. المحرك يقارن **الفرق** (`ease = قطعة − جسم`) بنطاق راحة مطلوب.
- إذا كانت قيمة محيط القطعة أصغر من 65% من محيط الجسم، يصدر تنبيه `possible_flat_measurement` (احتمال أن العرض المسطّح أُرسل على أنه محيط).

---

## 1. مدخلات قياسات الجسم (Body) — من موديول الأفاتار/الإدخال اليدوي

```json
{
  "user_id": "u-123",
  "source": "avatar",
  "unit": "cm",
  "measurements": {
    "height": 172, "chest": 92, "waist": 80, "hip": 96,
    "shoulder_width": 44, "arm_length": 60, "inseam": 79, "thigh": 56
  }
}
```

### 1.1 حقول المستند

| الحقل | النوع | إلزامي | الوصف والتحقق |
|---|---|---|---|
| `measurements` | object | نعم* | خريطة `اسم القياس → قيمة`. يجب أن تحتوي قياسًا صالحًا واحدًا على الأقل |
| `unit` | string | لا | `cm` `mm` `m` `in` (تُقبل أيضًا `سم` `ملم` `إنش`...). غيابه = سم مع تنبيه `missing_unit`. غير مدعوم = خطأ `unknown_unit` |
| `user_id` | أي قيمة | لا | يُعاد كما هو في النتيجة، لا يُستخدم في الحساب |
| `source` | string | لا | مثل `manual` أو `avatar`؛ يُعاد في `body_source`، لا يُستخدم في الحساب |

\* يُقبل أيضًا شكل مختصر بدون `measurements`: `{"chest": 92, "waist": 80}` (سم).

### 1.2 القياسات

كل القياسات **اختيارية فرديًا**؛ المحرك يحدد ما ينقص لكل نوع ملابس، ولا يخترع أي قيمة ناقصة.

| المفتاح | النوع | الوحدة | التعريف | النطاق المقبول بعد التحويل (سم) | يُستخدم في |
|---|---|---|---|---|---|
| `chest` | number | طول | محيط الصدر أفقيًا عند أعرض نقطة (للنساء: عند أبرز نقطة للصدر) 🔶 | 40–200 | كل الأنواع العلوية والفستان |
| `waist` | number | طول | محيط الخصر الطبيعي (أضيق نقطة فوق السرة) 🔶 | 30–200 | كل الأنواع |
| `hip` | number | طول | محيط الورك/الأرداف عند أعرض نقطة | 40–220 | كل الأنواع |
| `shoulder_width` | number | طول | عرض الكتفين من رأس الكتف إلى رأس الكتف الآخر عبر الظهر 🔶 | 20–70 | تيشيرت، قميص، جاكيت، فستان |
| `arm_length` | number | طول | من رأس الكتف إلى عظمة الرسغ والذراع مستقيمة | 30–100 | قميص، جاكيت (مقابل طول الكم) |
| `inseam` | number | طول | من أعلى الفخذ الداخلي إلى الكاحل | 30–120 | بنطلون |
| `thigh` | number | طول | محيط الفخذ عند أعرض نقطة | 25–110 | بنطلون |
| `height` | number | طول | الطول الكلي | 50–250 | فحص منطقية فقط، لا يحدد المقاس |
| `weight` | — | — | يُقبل ويُتجاهل مع تنبيه `weight_ignored` | — | لا شيء |

### 1.3 قواعد التحقق لكل قيمة

| الحالة | النتيجة |
|---|---|
| رقم، أو نص رقمي (`"92"`, `"92,5"`)، أو `{"value": 36, "unit": "in"}` | مقبول |
| `true/false`، `null`، نص غير رقمي، `NaN`، `Infinity` | خطأ `invalid_number` |
| صفر أو سالب | خطأ `non_positive` |
| خارج النطاق المعقول | خطأ `out_of_range` + تلميح ("هل القياس بالإنش؟ / بالملم؟") |
| اسم بديل (`bust`, `hips`, `shoulders`...) | يُوحَّد مع تنبيه `alias_used` (الجدول في `schema.BODY_ALIASES`) |
| اسم غير معروف | تنبيه `unknown_measurement` ويُتجاهل |
| نِسب غير منطقية (خصر > 1.6 × صدر، ورك < 0.6 × خصر، ساق > 60% من الطول) | تنبيه `unusual_proportions` (لا يُرفض) |

---

## 2. مدخلات قطعة الملابس (Garment) — من موديول الملابس

```json
{
  "garment_id": "TS-001",
  "name": "تيشيرت قطن",
  "category": "t_shirt",
  "fit_type": "regular",
  "stretch": "low",
  "measurement_type": "circumference",
  "unit": "cm",
  "ease_overrides": { "chest": [4, 12] },
  "sizes": [
    { "label": "S", "measurements": { "chest": 92,  "shoulder_width": 42, "length": 68 } },
    { "label": "M", "measurements": { "chest": 100, "shoulder_width": 44, "length": 70 } }
  ]
}
```

### 2.1 حقول المستند

| الحقل | النوع | إلزامي | القيم والتحقق | أثر الغياب |
|---|---|---|---|---|
| `category` | string | **نعم** | `t_shirt` `shirt` `jacket` `dress` `pants` `skirt` 🔶 | خطأ `missing_category` / `unknown_category` |
| `sizes` | array | **نعم** | قائمة غير فارغة، **مرتبة من الأصغر للأكبر** | خطأ `no_sizes` |
| `garment_id` | أي قيمة | لا | يُعاد في النتيجة | — |
| `name` | string | لا | يُعاد في النتيجة | — |
| `fit_type` | string | لا | `slim` `regular` `loose` `oversized` 🔶 | يُفترض `regular` + تنبيه + **الثقة تنخفض** |
| `stretch` | string | لا | `none` `low` `medium` `high` 🔶 | يُفترض `none` + تنبيه + **الثقة تنخفض** |
| `measurement_type` | string | لا | `circumference` (محيط كامل) أو `flat` (عرض مسطّح). `null` = `circumference` 🔶 | يُفترض `circumference` |
| `unit` | string | لا | `cm` `mm` `m` `in` | سم + تنبيه |
| `ease_overrides` | object | لا | `{مفتاح قياس: [أدنى، أعلى]}` بالسم، أرقام محدودة و`أدنى < أعلى`. مفتاح غير معروف = تنبيه | تُستخدم القيم الافتراضية في `rules.py` |

متى يستخدم المحرك القصة والمرونة؟ فقط إذا أُرسلت قيمة صريحة مدعومة. عند غيابها يُستخدم أساس محايد (`regular` / `none`) ويُصرَّح بذلك في `warnings` وتنخفض `confidence`.

### 2.2 كل مقاس داخل `sizes`

| الحقل | النوع | إلزامي | التحقق |
|---|---|---|---|
| `label` | string أو number | **نعم** | غير فارغ، فريد داخل القطعة (`null` و`true/false` مرفوضة) — يُعاد دائمًا كنص (`32` ← `"32"`، لكن `32.0` ← `"32.0"`؛ يُفضّل إرساله نصًا) |
| `measurements` | object | **نعم** | قياس صالح واحد على الأقل (نفس قواعد القيم في 1.3) |
| `unit` | string | لا | يتجاوز `unit` المستند لهذا المقاس فقط |

### 2.3 قياسات القطعة

| المفتاح | النوع | التعريف | يتأثر بـ `flat` | يُقارن مع (الجسم) | النطاق (سم) |
|---|---|---|---|---|---|
| `chest` | number | محيط صدر القطعة تحت الإبط | ×2 | `chest` | 30–280 |
| `waist` | number | محيط خصر القطعة / حزام البنطلون | ×2 | `waist` | 20–280 |
| `hip` | number | محيط الورك في القطعة | ×2 | `hip` | 30–300 |
| `thigh` | number | محيط فخذ البنطلون | ×2 | `thigh` | 20–150 |
| `shoulder_width` | number | من خياطة الكتف إلى الأخرى | لا | `shoulder_width` | 15–90 |
| `sleeve_length` | number | من خياطة الكتف إلى نهاية الكم | لا | `arm_length` | 5–110 |
| `inseam` | number | طول الساق الداخلي للبنطلون | لا | `inseam` | 5–130 |
| `sleeve_opening` | number | محيط نهاية الكم | ×2 | — للمعلومة فقط | 5–80 |
| `length` | number | الطول الكلي للقطعة | لا | — للمعلومة فقط | 15–200 |

### 2.4 القياسات المطلوبة حسب النوع (من `rules.CATEGORY_RULES`)

| النوع | أساسية (تحدد إمكانية الارتداء) | ثانوية (تحسّن الاختيار) |
|---|---|---|
| `t_shirt` | chest, shoulder_width | waist, hip |
| `shirt` | chest, shoulder_width | waist, sleeve_length, hip |
| `jacket` | chest, shoulder_width | sleeve_length, waist, hip |
| `dress` | chest, waist, hip | shoulder_width |
| `pants` | waist, hip | thigh, inseam |
| `skirt` | waist, hip | — |

**الحد الأدنى لإصدار توصية:** قياس أساسي واحد على الأقل متوفر في الجسم وفي القطعة. للثقة العالية: كل الأساسية + نصف الثانوية على الأقل + القصة والمرونة محددتان.

---

## 3. المخرجات (Result)

```json
{
  "engine_version": "0.2.0",
  "status": "ok",
  "garment_id": "TS-001", "garment_name": "تيشيرت قطن",
  "category": "t_shirt", "fit_type": "regular", "stretch": "low",
  "body_source": "avatar", "user_id": "u-123",
  "recommended_size": "M",
  "closest_size": "M",
  "overall_fit": "good", "overall_fit_ar": "مناسب", "overall_fit_en": "good fit",
  "confidence": "high",
  "alternative_size": {"size": "L", "direction": "larger", "code": "alt_larger",
                       "message_ar": "...", "message_en": "..."},
  "comparisons": [
    {"measurement": "chest", "body_measurement": "chest", "name_ar": "محيط الصدر", "name_en": "chest",
     "importance": "primary", "body_cm": 92.0, "garment_cm": 100.0, "ease_cm": 8.0,
     "target_ease_cm": [3.24, 16.0], "ease_source": "default", "penalty": 0.0032,
     "status": "good", "label_ar": "مناسب", "label_en": "good fit"}
  ],
  "issues":   [],
  "reasons":  [{"code": "reason_all_good", "message_ar": "...", "message_en": "..."}],
  "warnings": [],
  "not_compared": [{"measurement": "length", "garment_cm": 70.0, "code": "not_compared", "message_ar": "...", "message_en": "..."}],
  "missing_measurements": {"body": [], "garment": []},
  "all_sizes": [{"size": "S", "score": 1.96, "acceptable": false, "incomplete_primary": [],
                 "overall_fit": "tight", "statuses": {"chest": "too_tight", "shoulder_width": "good"}}],
  "disclaimer": {"code": "disclaimer", "message_ar": "...", "message_en": "..."}
}
```

### 3.1 الحقول

| الحقل | النوع | يظهر | المعنى |
|---|---|---|---|
| `status` | string | دائمًا | `ok` · `no_suitable_size` · `insufficient_data` · `invalid_input` |
| `situation` | string | مع `no_suitable_size` و حالة واحدة من `insufficient_data` | `all_sizes_too_small` · `all_sizes_too_large` · `no_single_size_fits` · `incomplete_garment_data` |
| `recommended_size` | string \| null | دائمًا | المقاس الموصى به؛ `null` إذا لم تكن هناك توصية موثوقة |
| `closest_size` | string \| null | دائمًا | أقرب مقاس (حتى لو غير مناسب) — للعرض فقط وليس توصية |
| `alternative_size` | object \| null | ما عدا `invalid_input` | `size`، `direction` (`larger`/`smaller`)، `code` (`alt_larger`/`alt_smaller`/`alt_fixes_tight`)، والرسالة |
| `overall_fit` | string | دائمًا | `good` · `tight` · `loose` · `mixed` · `insufficient_data` (+ `_ar`/`_en`) |
| `confidence` | string | دائمًا | `high` · `medium` · `low` · `none` |
| `comparisons[]` | array | ما عدا `invalid_input` | **تقييم كل منطقة** للمقاس الموصى به (أو الأقرب) — انظر 3.2 |
| `issues[]` | array | ما عدا `invalid_input` | المناطق غير المناسبة فقط، مع `measurement` و`status` ورسالة |
| `reasons[]` | array | ما عدا `invalid_input` | سبب التوصية أو سبب عدمها |
| `warnings[]` | array | دائمًا | تنبيهات (افتراضات، قياسات ناقصة، أسماء بديلة...) |
| `errors[]` | array | مع `invalid_input` فقط | أخطاء المدخلات، كل خطأ فيه `field` يحدد مكانه بالضبط |
| `missing_measurements` | object | ما عدا `invalid_input` | `body`: قياسات جسم ناقصة لهذا النوع؛ `garment`: قياسات ناقصة في بعض المقاسات |
| `all_sizes[]` | array | ما عدا `invalid_input` | ملخص كل مقاس: `score` (أقل = أفضل، **ليس نسبة دقة**)، `acceptable`، `incomplete_primary`، `statuses` |
| `not_compared[]` | array | ما عدا `invalid_input` | قياسات القطعة التي لم تدخل في القرار ولماذا |
| `disclaimer` | object | ما عدا `invalid_input` | تنبيه بأن التوصية تقديرية |
| `engine_version` | string | دائمًا | لتتبع تغيّر القواعد |

### 3.2 حالات كل منطقة (`comparisons[].status`)

| status | المعنى |
|---|---|
| `good` | الفرق (قطعة − جسم) داخل النطاق المطلوب |
| `slightly_tight` / `slightly_loose` | خارج النطاق بمسافة ≤ السماحية (3 سم محيطات، 1.5 سم كتفين) — قابل للارتداء |
| `too_tight` / `too_loose` | خارج النطاق بأكثر من السماحية — يمنع التوصية بهذا المقاس |
| `slightly_short` / `too_short` / `slightly_long` / `too_long` | للأطوال (سماحية 2 سم) — تنبيه فقط، لا يمنع |
| `missing_body` | قياس الجسم غير متوفر |
| `missing_garment` | قياس القطعة غير متوفر لهذا المقاس |

### 3.3 الاستجابة حسب الحالة

| الحالة | `recommended_size` | ماذا يعرض النظام المستدعي |
|---|---|---|
| `ok` | المقاس | المقاس + البديل + تقييم المناطق + الأسباب |
| `no_suitable_size` | `null` | "لا يوجد مقاس مناسب"، `closest_size` و`situation` و`reasons` توضح أي المناطق تمنع |
| `insufficient_data` | `null` | `reasons` توضح ما ينقص (قياسات أساسية، أو `incomplete_garment_data`) |
| `invalid_input` | `null` | `errors[]` مع `field` لكل خطأ |

### 3.4 HTTP (`python -m fit_engine serve`)
| الطلب | الاستجابة |
|---|---|
| `POST /recommend` بجسم `{"body": {...}, "garment": {...}}` | 200 للنتائج (`ok` / `no_suitable_size` / `insufficient_data`)، 422 لـ `invalid_input`، 400 لطلب بشكل خاطئ |
| `GET /schema` | الأنواع والقصات والمرونة وأسماء القياسات و`category_rules` |
| `GET /health` | `{"status": "ok", "engine_version": ...}` |
| `GET /` | واجهة المستخدم |

---

## 4. نقاط مفتوحة تعتمد على براءه وسجى 🔶

| # | السؤال | لماذا يهم | مع من |
|---|---|---|---|
| 1 | ما القياسات التي ينتجها موديول الأفاتار فعلًا، وبأي أسماء ووحدة؟ | القياس الناقص يخفض الثقة أو يمنع التوصية | براءه |
| 2 | هل قياسات الأفاتار **محيطات حقيقية** على سطح الموديل أم تقديرات من الطول/الوزن؟ | التقديرات من الوزن لا تكفي لتحديد مقاس موثوق | براءه |
| 3 | تعريف `shoulder_width` و`chest` و`waist` في الأفاتار يطابق الجدول 1.2؟ | اختلاف التعريف = خطأ منهجي في كل التوصيات | براءه |
| 4 | هل جدول المقاسات عند سجى **قياسات القطعة نفسها** أم **جدول مقاسات جسم** ("يناسب صدر 92–96")؟ | المحرك الحالي يتوقع قياسات القطعة. جدول مقاسات الجسم يحتاج طريقة مقارنة مختلفة **لم تُنفَّذ** | سجى |
| 5 | هل القياسات محيط كامل أم عرض مسطّح؟ | خطأ هنا يضاعف/ينصّف المحيطات | سجى |
| 6 | كيف تُخزَّن القصة والمرونة، وهل يمكن تحويلها للقيم المدعومة؟ | بدونها تنخفض الثقة | سجى |
| 7 | ما أنواع الملابس في الكتالوج غير الستة المدعومة (ثوب، عباية...)؟ | النوع غير المدعوم يُرفض | سجى |
| 8 | ترتيب المقاسات: هل يُرسل من الأصغر للأكبر؟ | يُستخدم لتحديد المقاس البديل المجاور | سجى |

الحل المقترح عند اختلاف الأسماء: طبقة تحويل (adapter) داخل هذا الموديول، **دون تعديل كود الزميلتين**.
