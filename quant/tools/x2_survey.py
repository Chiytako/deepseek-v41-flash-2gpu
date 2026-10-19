"""X2 前調査: GLM reqlog の構造を集計する (CPU のみ)。
usage: x2_survey.py <reqlog_dir>
"""
import collections
import glob
import multiprocessing as mp
import os
import sys

import orjson


def survey(path):
    c = collections.Counter()
    maxline = 0
    pend = collections.deque()
    for line in open(path, "rb"):
        maxline = max(maxline, len(line))
        i = line.find(b"{")
        if i < 0:
            continue
        try:
            d = orjson.loads(line[i:])
        except orjson.JSONDecodeError:
            c["bad_json"] += 1
            continue
        ev = d.get("event")
        c["ev:" + str(ev)] += 1
        o = d.get("obj") or {}
        if ev == "request.received.openai":
            msgs = o.get("messages") or []
            if o.get("tools"):
                c["has_tools"] += 1
            for m in msgs:
                c["role:" + str(m.get("role"))] += 1
                if m.get("reasoning_content"):
                    c["asst_reasoning"] += 1
                if m.get("tool_calls"):
                    c["asst_tool_calls"] += 1
                cont = m.get("content")
                if isinstance(cont, list):
                    for b in cont:
                        c["block:" + str(b.get("type"))] += 1
            c["thinking:" + str((o.get("chat_template_kwargs") or {}).get("thinking"))] += 1
            pend.append(msgs)
        elif ev == "request.received":
            txt = o.get("text") or ""
            if pend:
                msgs = pend.popleft()
                last = msgs[-1].get("content") if msgs else ""
                if isinstance(last, list):
                    last = " ".join(b.get("text", "") for b in last if isinstance(b, dict))
                probe = (last or "")[:40]
                c["pair_ok" if probe in txt else "pair_ng"] += 1
            else:
                c["pair_none"] += 1
            if o.get("image_data"):
                c["image_req"] += 1
        elif ev == "request.finished":
            out = d.get("out") or {}
            mi = out.get("meta_info") or {}
            c["prompt_tokens"] += int(mi.get("prompt_tokens") or 0)
            c["cached_tokens"] += int(mi.get("cached_tokens") or 0)
            c["completion_tokens"] += int(mi.get("completion_tokens") or 0)
    c["maxline"] = maxline
    return os.path.basename(path), dict(c)


def main():
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.log*")))
    tot = collections.Counter()
    with mp.Pool(24) as pool:
        for name, c in pool.imap_unordered(survey, files):
            ml = c.pop("maxline", 0)
            tot["maxline"] = max(tot["maxline"], ml)
            tot.update(c)
    for k, v in sorted(tot.items()):
        print(k, v)


if __name__ == "__main__":
    main()
