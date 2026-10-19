"""GLM reqlog -> per-request language/task rows (CPU only). usage: ja_survey.py <reqlog_dir> <out.jsonl> [nproc]"""
import glob, hashlib, multiprocessing as mp, os, sys, re
import orjson

def stats(t):
    kana = cjk = hang = ascii_al = nonsp = 0
    for ch in t:
        o = ord(ch)
        if ch.isspace():
            continue
        nonsp += 1
        if 0x3040 <= o <= 0x30ff or 0x31f0 <= o <= 0x31ff or 0xff66 <= o <= 0xff9f:
            kana += 1
        elif 0x4e00 <= o <= 0x9fff or 0x3400 <= o <= 0x4dbf:
            cjk += 1
        elif 0xac00 <= o <= 0xd7af:
            hang += 1
        elif o < 128 and ch.isalpha():
            ascii_al += 1
    return kana, cjk, hang, ascii_al, nonsp

def lang(t):
    k, c, h, a, n = stats(t)
    if n == 0:
        return "empty", 0.0
    jr = (k + c) / n
    if k >= 3 and k / max(1, k + c) >= 0.15 and jr >= 0.10:
        return "ja", jr
    if c / n >= 0.2 and k < 3:
        return "zh", c / n
    if h / n >= 0.2:
        return "ko", h / n
    return "en", jr

def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join((b.get("text") or "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""

def nimg(c):
    if isinstance(c, list):
        return sum(1 for b in c if isinstance(b, dict) and b.get("type") in ("image_url", "image"))
    return 0

def proc(path):
    rows = []
    for line in open(path, "rb"):
        if b'"event": "request.received.openai"' not in line[:400]:
            continue
        i = line.find(b"{")
        try:
            d = orjson.loads(line[i:])
        except orjson.JSONDecodeError:
            continue
        o = d.get("obj") or {}
        msgs = o.get("messages") or []
        sys_t = "\n".join(text_of(m.get("content")) for m in msgs if m.get("role") in ("system", "developer"))
        users = [text_of(m.get("content")) for m in msgs if m.get("role") == "user"]
        tools_r = [text_of(m.get("content")) for m in msgs if m.get("role") == "tool"]
        asst = [text_of(m.get("content")) for m in msgs if m.get("role") == "assistant"]
        first_u = users[0] if users else ""
        last_u = users[-1] if users else ""
        # strip common wrappers (<system-reminder>... ) for task detection
        fu_clean = re.sub(r"<system-reminder>.*?</system-reminder>", "", first_u, flags=re.S).strip()
        lu_clean = re.sub(r"<system-reminder>.*?</system-reminder>", "", last_u, flags=re.S).strip()
        ut = "\n".join(re.sub(r"<system-reminder>.*?</system-reminder>", "", u, flags=re.S) for u in users)
        ctk = o.get("chat_template_kwargs") or {}
        tools = o.get("tools") or []
        tnames = sorted({(t.get("function") or {}).get("name", "") for t in tools if isinstance(t, dict)})
        lu_lang, lu_r = lang(lu_clean)
        fu_lang, fu_r = lang(fu_clean)
        ut_lang, ut_r = lang(ut)
        sy_lang, sy_r = lang(sys_t)
        tr_lang, tr_r = lang("\n".join(tools_r)[-200000:])
        as_lang, as_r = lang("\n".join(asst)[-200000:])
        rows.append({
            "ts": d.get("timestamp"), "file": os.path.basename(path), "nmsg": len(msgs),
            "sys_h": hashlib.md5(sys_t[:400].encode()).hexdigest()[:10], "sys_len": len(sys_t), "sys_head": sys_t[:160],
            "task_h": hashlib.md5(fu_clean[:2000].encode()).hexdigest()[:12],
            "fu": fu_clean[:400], "lu": lu_clean[:300], "fu_len": len(fu_clean), "lu_len": len(lu_clean),
            "fu_lang": fu_lang, "lu_lang": lu_lang, "ut_lang": ut_lang, "sys_lang": sy_lang, "tool_lang": tr_lang, "asst_lang": as_lang,
            "ut_jr": round(ut_r, 3), "ntools": len(tools), "tnames": tnames[:40], "nimg": sum(nimg(m.get("content")) for m in msgs),
            "thinking": ctk.get("enable_thinking", ctk.get("thinking", None)),
            "effort": o.get("reasoning_effort") or ctk.get("reasoning_effort"),
            "temp": o.get("temperature"), "max_tokens": o.get("max_tokens") or o.get("max_completion_tokens"),
            "model": o.get("model"), "chars_user": sum(len(u) for u in users), "chars_tool": sum(len(t) for t in tools_r),
            "chars_asst": sum(len(a) for a in asst), "chars_sys": len(sys_t),
        })
    return rows

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.log*")))
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    with open(sys.argv[2], "wb") as f, mp.Pool(n) as pool:
        for rows in pool.imap_unordered(proc, files):
            for r in rows:
                f.write(orjson.dumps(r) + b"\n")
