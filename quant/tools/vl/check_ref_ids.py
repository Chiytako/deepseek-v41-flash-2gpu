"""vl_kl / vl_calib の ids・token_types が参照実装 (encoding.encode_case + image_processor.prepare_vl_inputs) と一致するか確かめる。
CPU のみ。torch と transformers が要るので dsv41-quant イメージの中で動かす:
  sudo docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v $DSV41/ref:/ref:ro \
    -v $MODELS/DeepSeek-V4.1-Flash-1017:/model:ro -v $DSV41/work/vl:/vl:ro \
    -v $DSV41/tools/vl:/tv:ro --entrypoint python dsv41-quant:151 /tv/check_ref_ids.py
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import json, os, sys, types
sys.path.insert(0, "/model/encoding"); sys.path.insert(0, "/ref")
from transformers import AutoTokenizer
from encoding import encode_case
from image_processor import prepare_vl_inputs
cfg = json.load(open("/model/inference/config.json"))
args = types.SimpleNamespace(image_token_id=129264, vision_enabled=cfg.get("vision_n_layers", 0) > 0,
                             **{k: cfg.get(k) for k in ("vision_patch_size", "vision_downsample_ratio", "vision_max_n_token", "vision_min_pixels", "vision_max_wh_ratio")})
print("args", vars(args))
tok = AutoTokenizer.from_pretrained("/model")
os.chdir("/vl")
bad = 0; n = 0
for name in ("vl_kl.jsonl", "vl_calib.jsonl"):
    for line in open(name):
        r = json.loads(line); n += 1
        prompt, recs = encode_case({"messages": r["messages"]}, r["thinking_mode"])
        ids, tt, imgs = prepare_vl_inputs(prompt, recs, tok, args)
        ok = ids == r["vl_ids"] and tt == r["token_types"] and [(i.start, i.n_vit_h, i.n_vit_w) for i in imgs] == \
             [(s["start"], s["n_vit_h"], s["n_vit_w"]) for s in r["image_spans"]]
        if not ok:
            bad += 1
            print("MISMATCH", r["id"], len(ids), len(r["vl_ids"]), ids[:4], r["vl_ids"][:4], [(i.start, i.n_vit_h, i.n_vit_w) for i in imgs][:2], [(s["start"], s["n_vit_h"], s["n_vit_w"]) for s in r["image_spans"]][:2])
print(f"checked {n}, mismatches {bad}")
