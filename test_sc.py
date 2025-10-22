import time
from pathlib import Path
import io, csv, inspect
from workshops_ai.settings import get_settings
import workshops_ai.tools.search_catalogs as sc

def log(msg):
    print(time.strftime("%H:%M:%S"), "-", msg, flush=True)

log("python OK")
src_idx = inspect.getsource(sc._build_threat_index)
print("index has Value token?:", any(x in src_idx for x in ["att_value_col","Include","Included","Enabled"," Value"]))

# 0) ukaž zdroje funkcí, které jdeme používat
print("--- mapped_threats_for_asset_type source ---")
print(inspect.getsource(sc.mapped_threats_for_asset_type))
print("--- normalize_threat_code source (first 20 lines) ---")
n_src = inspect.getsource(sc.normalize_threat_code)
print("\n".join(n_src.splitlines()[:20]))

# 1) RYCHLÝ TEST bez normalizace (vynecháme normalize_threat_code -> nevoláme index)
log("raw-mapped without normalization: start")
base = Path(get_settings().paths.catalogs_dir)
p = base / "Asset type x Threat.csv"
from workshops_ai.tools.search_catalogs import _detect_encoding_and_delimiter, _first_present_column
enc, delim = _detect_encoding_and_delimiter(p)
rows = list(csv.DictReader(io.StringIO(p.read_text(encoding=enc)), delimiter=delim))
a = _first_present_column(rows, ("asset_type","Asset type","EN Name","Name"))
t = _first_present_column(rows, ("threat_code","Threat","threat","EN Name","Name"))
raw_codes = sorted({ (r.get(t,"") or "").strip()
                     for r in rows
                     if (r.get(a) or "").strip()=="Application software" and (r.get(t) or "").strip() })
log(f"raw-mapped without normalization: done -> {len(raw_codes)} (head={raw_codes[:12]})")

# 2) Otestuj mapped_threats_for_asset_type s vypnutou normalizací (identity)
log("mapped_threats_for_asset_type with identity normalize: start")
_orig_norm = sc.normalize_threat_code
try:
    sc.normalize_threat_code = lambda s: (s or "").strip()
    mapped_id = sc.mapped_threats_for_asset_type("Application software")
    log(f"mapped_threats_for_asset_type identity -> {len(mapped_id)} (head={mapped_id[:12]})")
finally:
    sc.normalize_threat_code = _orig_norm

# 3) Změř přesně, kde to visí při plné normalizaci
log("mapped_threats_for_asset_type (full) A: calling...")
try:
    mapped_full = sc.mapped_threats_for_asset_type("Application software")
    log(f"mapped_threats_for_asset_type (full) B: returned -> {len(mapped_full)}")
except Exception as e:
    log(f"mapped_threats_for_asset_type (full) ERROR: {e!r}")

# 4) Explicitně zavolej jen normalize_threat_code na jeden řetězec, ať vidíme, zda visí index
if raw_codes:
    sample = raw_codes[0]
    log(f"normalize_threat_code('{sample}') start")
    try:
        c = sc.normalize_threat_code(sample)
        log(f"normalize_threat_code('{sample}') done -> {c}")
    except Exception as e:
        log(f"normalize_threat_code('{sample}') ERROR: {e!r}")

# 5) Vynuceně postav index a hned vytiskni jeho velikost, ať víme, že doběhne
log("_build_threat_index() start")
try:
    idx = sc._build_threat_index()
    log(f"_build_threat_index() done -> {len(idx)} threats")
except Exception as e:
    log(f"_build_threat_index() ERROR: {e!r}")

print("DONE")
