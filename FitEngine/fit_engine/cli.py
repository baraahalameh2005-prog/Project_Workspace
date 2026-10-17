"""Command-line interface.

    python -m fit_engine recommend --body examples/body_medium.json --garment examples/tshirt_regular.json
    python -m fit_engine recommend --body ... --garment ... --summary
    python -m fit_engine serve --port 8765
"""

import argparse
import json
import sys

from .engine import recommend_size
from .rules import CATEGORY_NAMES_AR


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summary_ar(res):
    """Short human-readable Arabic summary of a result."""
    lines = []
    cat = CATEGORY_NAMES_AR.get(res.get("category"), res.get("category"))
    name = res.get("garment_name") or res.get("garment_id") or ""
    lines.append(f"القطعة: {name} ({cat}) — القصة: {res.get('fit_type')} — المرونة: {res.get('stretch')}")
    st = res["status"]
    if st == "invalid_input":
        lines.append("❌ المدخلات غير صالحة:")
        lines += [f"  - {e['message_ar']}" for e in res["errors"]]
        return "\n".join(lines)
    if st == "ok":
        lines.append(f"✅ المقاس الموصى به: {res['recommended_size']}")
    elif st == "no_suitable_size":
        lines.append(f"⚠️ لا يوجد مقاس مناسب. الأقرب: {res['closest_size']}")
    else:
        lines.append("⚠️ لا يمكن إصدار توصية: معلومات غير كافية")
    lines.append(f"تقييم الملاءمة: {res['overall_fit_ar']} — الثقة: {res['confidence']}")
    if res.get("comparisons"):
        lines.append("المقارنة (القطعة − الجسم):")
        for c in res["comparisons"]:
            if c["ease_cm"] is None:
                lines.append(f"  • {c['name_ar']}: {c['label_ar']}")
            else:
                lo, hi = c["target_ease_cm"]
                lines.append(f"  • {c['name_ar']}: جسم {c['body_cm']} / قطعة {c['garment_cm']} "
                             f"→ فرق {c['ease_cm']:+.1f} (المطلوب {lo:+.1f}..{hi:+.1f}) = {c['label_ar']}")
    for r in res.get("reasons", []):
        lines.append(f"السبب: {r['message_ar']}")
    if res.get("alternative_size"):
        lines.append(f"بديل: {res['alternative_size']['message_ar']}")
    for w in res.get("warnings", []):
        lines.append(f"تنبيه: {w['message_ar']}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fit_engine", description="Clothing size recommendation")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("recommend", help="recommend a size from two JSON files")
    r.add_argument("--body", required=True, help="path to body measurements JSON")
    r.add_argument("--garment", required=True, help="path to garment JSON")
    r.add_argument("--summary", action="store_true", help="print an Arabic summary instead of JSON")

    s = sub.add_parser("serve", help="run a local JSON HTTP API (no external dependencies)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)

    args = p.parse_args(argv)
    if args.cmd == "recommend":
        try:
            body, garment = _load(args.body), _load(args.garment)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error reading input: {e}", file=sys.stderr)
            return 2
        res = recommend_size(body, garment)
        if args.summary:
            print(summary_ar(res))
        else:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["status"] != "invalid_input" else 1
    if args.cmd == "serve":
        from .server import serve
        serve(args.host, args.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
