import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import sys, time
sys.path.insert(0, f"{ROOT}")
from dress3d.schema import parse_spec
from dress3d.generator import DressGenerator
from dress3d import export as ex
from dress3d.render import turntable
for sleeve in ("none", "long"):
    spec = parse_spec({"style": {"silhouette": "a-line", "sleeve": sleeve},
                       "measurements": {"dress_length": 95, "bust": 92, "waist": 78, "hip": 100, "hem": 150},
                       "options": {"fold_count": 8, "fold_amplitude": 1.5}})
    dm = DressGenerator(spec).build()
    tex = ex.make_print_texture(dm.atlas_size_cm)
    t = time.time()
    img = turntable(dm.render_vertices, dm.render_faces, uv=ex.pattern_uv(dm), texture=tex, size=(320, 420))
    print(sleeve, "render s", round(time.time()-t, 1))
    img.save(f"{ROOT}/cache/smoke/turn_{sleeve}.png")
