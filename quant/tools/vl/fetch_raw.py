"""work/vl/raw に画像評価・校正の元データを取る (合計 ~650MB)。取得物は信用しない (データとして読むだけ)。"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import os, sys
from huggingface_hub import hf_hub_download
RAW = DSV41 + "/work/vl/raw"
SPECS = [
    ("SakanaAI/JA-VLM-Bench-In-the-Wild", "data/test-00000-of-00001.parquet"),
    ("SakanaAI/JA-VLM-Bench-In-the-Wild", "README.md"),
    ("SakanaAI/JA-VG-VQA-500", "data/test-00000-of-00001.parquet"),
    ("SakanaAI/JA-VG-VQA-500", "README.md"),
    ("turing-motors/Japanese-Heron-Bench", "LICENCE.md"),
    ("turing-motors/Japanese-Heron-Bench", "README.md"),
    ("turing-motors/Japanese-Heron-Bench", "questions_ja.jsonl"),
    ("turing-motors/Japanese-Heron-Bench", "answers_gpt4.jsonl"),
    ("turing-motors/Japanese-Heron-Bench", "context_ja.jsonl"),
    ("llm-jp/MECHA-ja", "data/test-00000-of-00002.parquet"),
    ("llm-jp/MECHA-ja", "README.md"),
    ("JMMMU/JMMMU", "Japanese_Heritage/test-00000-of-00001.parquet"),
    ("JMMMU/JMMMU", "Computer_Science/test-00000-of-00001.parquet"),
    ("JMMMU/JMMMU", "README.md"),
    ("wulipc/CC-OCR", "multi_lan_ocr/Japanese/Japanese_150.tsv"),
    ("wulipc/CC-OCR", "LICENSE"),
    ("lmms-lab/ChartQA", "data/test-00000-of-00001.parquet"),
    ("lmms-lab/DocVQA", "DocVQA/validation-00000-of-00006.parquet"),
    ("lmms-lab/MMBench", "en/dev-00000-of-00001.parquet"),
]
only = sys.argv[1:]
for rid, fn in SPECS:
    if only and not any(o in rid for o in only):
        continue
    try:
        p = hf_hub_download(rid, fn, repo_type="dataset", local_dir=os.path.join(RAW, rid.replace("/", "__")))
        print("OK", p, os.path.getsize(p), flush=True)
    except Exception as e:
        print("ERR", rid, fn, type(e).__name__, str(e)[:200], flush=True)

def heron_images():
    for i in range(1, 22):
        fn = f"images/{i:03d}.jpg"
        p = hf_hub_download("turing-motors/Japanese-Heron-Bench", fn, repo_type="dataset",
                            local_dir=os.path.join(RAW, "turing-motors__Japanese-Heron-Bench"))
        print("OK", p, flush=True)
if __name__ == "__main__" and only == ["heron-images"]:
    heron_images()
