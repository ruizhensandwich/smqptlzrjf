"""Model board: every evaluated model side by side, with strengths and weaknesses.

    python tools/model_board.py                     # print the board (also results/model_board.md)
    python tools/model_board.py --add <run_id>      # add an evaluate.py run from results/runs/

evaluate.py adds a row automatically whenever it scores all validation files
(train too with --split all) and prints the board at the end, so does train.py
after training. Rows live in results/model_board.csv (committed); one row per
weight file (same weights scored again replace their row).

Numbers are scored as denoise.py writes the files (PCM_16), per file then averaged:
  Val PESQ / ΔPESQ    validation PESQ-WB and its gain over the noisy input
  0 dB / 15 dB / hard validation PESQ at the lowest and highest SNR and on hard noise
  나빠진 파일         validation files whose PESQ fell below the noisy input
  Train PESQ          train PESQ-WB (if scored)
  일반화 차           train gain − validation gain: large = fits training speakers/noises
  추론 속도           inference time / audio time on the device used (lower is faster)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BOARD_CSV = ROOT / "results" / "model_board.csv"
BOARD_MD = ROOT / "results" / "model_board.md"
COLUMNS = ["date", "run_id", "tag", "model", "params", "weights_mb", "weights_sha", "epochs",
           "best_epoch", "train_min", "train_device", "val_pesq", "val_d_pesq", "val_snr",
           "val_pesq_0db", "val_pesq_5db", "val_pesq_10db", "val_pesq_15db", "val_d_0db",
           "val_pesq_hard", "val_worse_pct", "train_pesq", "train_d_pesq", "gen_gap",
           "best_category", "best_category_d", "worst_category", "worst_category_d",
           "category_split", "rtf", "infer_device", "note"]


# --------------------------------------------------------------------------
# metrics from per-file scores
def _paired(per_file: list[dict], method: str):
    noisy = {r["segment_id"]: r for r in per_file if r["method"] == "noisy_input"}
    return [(noisy[r["segment_id"]], r) for r in per_file if r["method"] == method]


def _mean(xs):
    xs = [x for x in xs if x == x]          # drop NaN
    return float(np.mean(xs)) if xs else float("nan")


def split_metrics(per_file: list[dict], method: str) -> dict:
    pairs = _paired(per_file, method)
    m = {"pesq": _mean([float(b["pesq"]) for _, b in pairs]),
         "d_pesq": _mean([float(b["pesq"]) - float(a["pesq"]) for a, b in pairs]),
         "snr": _mean([float(b["snr_db"]) for _, b in pairs]),
         "worse_pct": 100 * _mean([float(float(b["pesq"]) < float(a["pesq"])) for a, b in pairs])}
    for snr in ("0", "5", "10", "15"):
        sel = [(a, b) for a, b in pairs if float(b["target_snr_db"]) == float(snr)]
        m[f"pesq_{snr}db"] = _mean([float(b["pesq"]) for _, b in sel])
        m[f"d_{snr}db"] = _mean([float(b["pesq"]) - float(a["pesq"]) for a, b in sel])
    m["pesq_hard"] = _mean([float(b["pesq"]) for _, b in pairs if b["noise_difficulty"] == "hard"])
    cats: dict[str, list[float]] = {}
    for a, b in pairs:
        cats.setdefault(b["noise_category"], []).append(float(b["pesq"]) - float(a["pesq"]))
    cat_d = {c: _mean(v) for c, v in cats.items() if len(v) >= 3}
    if cat_d:
        best, worst = max(cat_d, key=cat_d.get), min(cat_d, key=cat_d.get)
        m.update(best_category=best, best_category_d=cat_d[best],
                 worst_category=worst, worst_category_d=cat_d[worst])
    return m


def make_row(per_split: dict[str, list[dict]], method: str, info: dict) -> dict:
    """per_split: {'validation': per_file rows, 'train': ... (optional)}; info: run facts."""
    v = split_metrics(per_split["validation"], method)
    row = {k: info.get(k, "") for k in COLUMNS}
    row.update(val_pesq=v["pesq"], val_d_pesq=v["d_pesq"], val_snr=v["snr"],
               val_pesq_0db=v["pesq_0db"], val_pesq_5db=v["pesq_5db"],
               val_pesq_10db=v["pesq_10db"], val_pesq_15db=v["pesq_15db"], val_d_0db=v["d_0db"],
               val_pesq_hard=v["pesq_hard"], val_worse_pct=v["worse_pct"])
    cat_src = v
    if "train" in per_split:
        t = split_metrics(per_split["train"], method)
        row.update(train_pesq=t["pesq"], train_d_pesq=t["d_pesq"],
                   gen_gap=t["d_pesq"] - v["d_pesq"])
        cat_src = t                         # 1,296 files: categories are more reliable
    for k in ("best_category", "best_category_d", "worst_category", "worst_category_d"):
        row[k] = cat_src.get(k, "")
    row["category_split"] = "train" if "train" in per_split else "validation"
    return row


def run_facts(ckpt: dict, weights: Path) -> dict:
    """Model size and training facts from the checkpoint and its run folder."""
    meta = ckpt.get("meta", {})
    params = meta.get("parameters") or sum(int(np.prod(t.shape)) for k, t in
                                            ckpt["state_dict"].items() if k != "window")
    facts = {"model": ckpt["model"], "params": params,
             "weights_mb": round(weights.stat().st_size / 2 ** 20, 2),
             "epochs": meta.get("epochs_completed", ""), "best_epoch": meta.get("best_epoch", ""),
             "train_min": f"{meta['elapsed_seconds'] / 60:.1f}" if meta.get("elapsed_seconds") else "",
             "train_device": meta.get("device", "")}
    summary = ROOT / "runs" / str(meta.get("run_id", "")) / "summary.json"
    if meta.get("run_id") and summary.is_file():
        s = json.loads(summary.read_text(encoding="utf-8"))
        facts.update(epochs=s["epochs_completed"], best_epoch=s["best_epoch"],
                     train_min=f"{s['elapsed_seconds'] / 60:.1f}", train_device=s["device"])
    return facts


# --------------------------------------------------------------------------
# storage
def load_rows() -> list[dict]:
    if not BOARD_CSV.exists():
        return []
    with open(BOARD_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_row(row: dict) -> None:
    rows = [r for r in load_rows() if r.get("weights_sha") != str(row["weights_sha"])]
    rows.append({k: _fmt(row.get(k, "")) for k in COLUMNS})
    BOARD_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(BOARD_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _fmt(x):
    if isinstance(x, float):
        return "" if x != x else f"{x:.4f}"
    return x


# --------------------------------------------------------------------------
# rendering
def _f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def judge(rows: list[dict]) -> dict[str, tuple[list[str], list[str]]]:
    """Strengths and weaknesses of each model, relative to the others (and to the input)."""
    out = {}
    many = len(rows) > 1

    def best(key, high=True):
        vals = [_f(r, key) for r in rows if _f(r, key) == _f(r, key)]
        return (max(vals) if high else min(vals)) if vals else None

    top = best("val_pesq")
    for r in rows:
        pro, con = [], []
        vp, gap, worse = _f(r, "val_pesq"), _f(r, "gen_gap"), _f(r, "val_worse_pct")
        if many and vp == top:
            pro.append(f"검증 PESQ 최고 ({vp:.3f})")
        elif many and top is not None:
            con.append(f"최고 모델보다 {vp - top:+.3f}")
        if many and _f(r, "val_d_0db") == best("val_d_0db"):
            pro.append(f"0 dB에서 가장 강함 (+{_f(r, 'val_d_0db'):.3f})")
        if many and _f(r, "val_pesq_hard") == best("val_pesq_hard"):
            pro.append(f"어려운 잡음(hard)에서 가장 강함 ({_f(r, 'val_pesq_hard'):.3f})")
        if worse == worse and worse <= 2:
            pro.append(f"입력보다 나빠지는 파일이 거의 없음 ({worse:.1f}%)")
        elif worse == worse and worse > 5:
            con.append(f"입력보다 나빠진 파일 {worse:.1f}%")
        if gap == gap:
            if many and gap == best("gen_gap", high=False):
                pro.append(f"일반화 차가 가장 작음 ({gap:+.3f})")
            if gap >= 0.08:
                con.append(f"과적합 경향: train 개선이 validation보다 {gap:+.3f} 큼")
        if many and _f(r, "params") == best("params", high=False):
            pro.append(f"가장 가벼움 ({int(_f(r, 'params')):,} 파라미터)")
        rtf = _f(r, "rtf")
        if rtf == rtf and "cpu" in str(r.get("infer_device", "")) and rtf > 0.5:
            con.append(f"CPU 추론이 느림 (실시간의 {rtf:.2f}배 시간)")
        if r.get("best_category"):
            pro.append(f"가장 잘 줄이는 잡음: {r['best_category']} (+{_f(r, 'best_category_d'):.2f})")
        if r.get("worst_category"):
            con.append(f"가장 약한 잡음: {r['worst_category']} ({_f(r, 'worst_category_d'):+.2f})")
        d0 = _f(r, "val_d_0db")
        if d0 == d0 and d0 < 0.25:
            con.append(f"0 dB 개선이 작음 (+{d0:.3f})")
        out[r["weights_sha"]] = (pro, con)
    return out


def render(rows: list[dict]) -> str:
    if not rows:
        return "_모델 비교표: 아직 기록이 없습니다 (evaluate.py로 validation 전체를 채점하면 추가됩니다)._"
    rows = sorted(rows, key=lambda r: r["date"])
    hi = {k: max((_f(r, k) for r in rows if _f(r, k) == _f(r, k)), default=None)
          for k in ("val_pesq", "val_d_pesq", "val_snr", "val_pesq_0db", "val_pesq_15db",
                    "val_pesq_hard", "train_pesq")}
    lo = {k: min((_f(r, k) for r in rows if _f(r, k) == _f(r, k)), default=None)
          for k in ("val_worse_pct", "gen_gap", "rtf")}

    def cell(r, k, fmt="{:.3f}", hi_best=True):
        x = _f(r, k)
        if x != x:
            return "-"
        s = fmt.format(x)
        target = hi.get(k) if hi_best else lo.get(k)
        return f"**{s}**" if len(rows) > 1 and target is not None and x == target else s

    lines = ["### 모델 비교표 (Validation 164개 · 16비트 출력 기준, **굵게** = 가장 좋은 값)", "",
             "| # | 태그 | 모델 | 파라미터 | Val PESQ | ΔPESQ | Val SNR | 0 dB | 15 dB | hard "
             "| 나빠진 파일 | Train PESQ | 일반화 차 | 학습 | 추론 속도 | 날짜 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        params = f"{int(_f(r, 'params')) / 1000:.0f}k" if _f(r, "params") == _f(r, "params") else "-"
        train = (f"{r['epochs']}ep/{r['best_epoch']} · {r['train_min']}분 · {r['train_device']}"
                 if r.get("epochs") else "-")
        lines.append(" | ".join([
            f"| {i}", r["tag"] or r["run_id"], r["model"], params, cell(r, "val_pesq"),
            cell(r, "val_d_pesq", "{:+.3f}"), cell(r, "val_snr", "{:.2f}"), cell(r, "val_pesq_0db"),
            cell(r, "val_pesq_15db"), cell(r, "val_pesq_hard"),
            cell(r, "val_worse_pct", "{:.1f}%", hi_best=False), cell(r, "train_pesq"),
            cell(r, "gen_gap", "{:+.3f}", hi_best=False), train,
            (cell(r, "rtf", "{:.3f}", hi_best=False) + f" ({r['infer_device']})")
            if r.get("rtf") else "-", r["date"][:10]]) + " |")
    lines += ["", "| # | 태그 | 장점 | 단점 | 메모 |", "|---|---|---|---|---|"]
    verdict = judge(rows)
    for i, r in enumerate(rows, 1):
        pro, con = verdict[r["weights_sha"]]
        lines.append(f"| {i} | {r['tag'] or r['run_id']} | {'<br>'.join(pro) or '-'} | "
                     f"{'<br>'.join(con) or '-'} | {r.get('note') or '-'} |")
    lines += ["", "ΔPESQ = 입력(noisy) 대비 개선 · 0 dB/15 dB/hard = 해당 조건의 Val PESQ · "
              "나빠진 파일 = 입력보다 PESQ가 낮아진 validation 파일 비율 · 일반화 차 = train 개선 − "
              "validation 개선 (클수록 학습 화자에 치우침) · 학습 = 실행 epoch/선택 epoch · "
              "추론 속도 = 추론 시간 ÷ 음성 길이 (낮을수록 빠름) · 범주 장단점은 train 1,296개 기준"]
    return "\n".join(lines)


def show() -> str:
    text = render(load_rows())
    BOARD_MD.parent.mkdir(parents=True, exist_ok=True)
    BOARD_MD.write_text(text + "\n", encoding="utf-8")
    return text


def add_run(run_id: str, note: str | None = None) -> dict:
    """Rebuild a board row from an evaluate.py run folder (results/runs/<run_id>/)."""
    from se.checkpoint import read_checkpoint
    d = ROOT / "results" / "runs" / run_id
    per_split = {}
    for split in ("validation", "train"):
        if (d / split / "per_file.csv").exists():
            with open(d / split / "per_file.csv", encoding="utf-8-sig") as f:
                per_split[split] = list(csv.DictReader(f))
    if "validation" not in per_split:
        sys.exit(f"{d} has no validation/per_file.csv")
    info = json.loads((d / "validation" / "run_config.json").read_text(encoding="utf-8"))
    if info.get("subset") != "all":
        sys.exit("only runs over all validation files go on the board")
    method = [m for m in info["methods"] if m != "noisy_input"][-1]
    weights = ROOT / info["weights"]
    facts = run_facts(read_checkpoint(weights), weights) if weights.exists() else {}
    row = make_row(per_split, method, {
        **facts, "date": run_id[:8], "run_id": run_id, "tag": run_id.split("_", 1)[-1],
        "model": info.get("model", facts.get("model", "")), "weights_sha": info["weights_sha"],
        "rtf": info.get("rtf", ""), "infer_device": info.get("device", ""),
        "note": note if note is not None else info.get("note", "")})
    row["date"] = f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
    save_row(row)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", metavar="RUN_ID")
    ap.add_argument("--note")
    args = ap.parse_args(argv)
    if args.add:
        add_run(args.add, args.note)
    print(show())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
