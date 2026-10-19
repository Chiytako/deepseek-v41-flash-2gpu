"""TextVQA validation (lmms-lab/textvqa) の先頭 2 行グループ (200 行) だけを範囲読みで取る (~37MB)。"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import os
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
OUT = DSV41 + "/work/vl/raw/lmms-lab__textvqa/validation-00000-rg0-1.parquet"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fs = HfFileSystem()
with fs.open("datasets/lmms-lab/textvqa/data/validation-00000-of-00003.parquet", "rb", block_size=8 << 20) as f:
    pf = pq.ParquetFile(f)
    t = pf.read_row_groups([0, 1])
pq.write_table(t, OUT)
print(t.num_rows, os.path.getsize(OUT))
