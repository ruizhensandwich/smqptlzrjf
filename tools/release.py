"""Submission rounds: record them, tag them, push them, keep README in sync.

    python tools/release.py prepare 1 --summary "MaskCNN (course example settings)"
    python tools/release.py tag 1                  # after committing
    python tools/release.py push 1                 # secret scan, then push main + submission-1
    python tools/release.py submitted 1 [--date 2026-11-24]   # after the Learn Us post
    python tools/release.py leaderboard 1 --pesq 1.70 [--snr 12.9]
    python tools/release.py readme                 # only re-render README blocks

prepare  checks results/submissions/roundN/ (validation) and roundN/train/ were
         scored with the current code and weights (fingerprint), then writes the
         round into submissions.json and README: the results table and the
         reproduction block the notice asks for (seed, epochs, selected epoch and
         its validation score, settings, weight path). Commit afterwards.
tag      requires a clean working tree and a matching fingerprint, creates the
         annotated tag submission-N on HEAD.
push     runs tools/secret_scan.py, then `git push origin main` and the tag.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import (CHALLENGE, DEADLINE, REPO_URL, ROOT, TEAM, WEIGHTS,  # noqa: E402
                          fingerprint, get_round, git, load_submissions, round_dir,
                          save_submissions, tag_name)

README = ROOT / "README.md"
BLOCKS = {"submissions": ("<!-- submissions:start -->", "<!-- submissions:end -->"),
          "reproduce": ("<!-- reproduce:start -->", "<!-- reproduce:end -->")}


def round_result(n: int, split: str = "validation") -> dict:
    d = round_dir(n) if split == "validation" else round_dir(n) / split
    if not (d / "run_config.json").exists():
        sys.exit(f"{d.relative_to(ROOT)} missing. Run:  python evaluate.py --round {n}")
    info = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    with open(d / "summary.csv", encoding="utf-8-sig") as f:
        rows = {r["method"]: r for r in csv.DictReader(f)}
    return {"files": info["files"], "pesq": round(float(rows["submission"]["pesq"]), 4),
            "snr_db": round(float(rows["submission"]["snr_db"]), 3),
            "noisy_pesq": round(float(rows["noisy_input"]["pesq"]), 4),
            "noisy_snr_db": round(float(rows["noisy_input"]["snr_db"]), 3),
            "fingerprint": info.get("code_fingerprint")}


def require_fingerprint(n: int) -> dict:
    now = fingerprint()
    res = {}
    for split in ("validation", "train"):
        res[split] = round_result(n, split)
        if res[split]["fingerprint"] != now:
            sys.exit(f"round {n} {split} results were scored with code/weights "
                     f"{res[split]['fingerprint']}, current is {now}.\n"
                     f"Re-run:  python evaluate.py --round {n}")
    return res


def weights_info() -> dict:
    from se.checkpoint import read_checkpoint
    ck = read_checkpoint(WEIGHTS)
    m = ck["meta"]
    cfg = m.get("config") or {}
    t = {**cfg.get("train", {}), **{f"data.{k}": v for k, v in cfg.get("data", {}).items()}}
    if cfg.get("discriminator"):
        t["discriminator"] = cfg["discriminator"]
    return {"model": ck["model"], "model_args": ck["model_args"],
            "parameters": m.get("parameters"), "loss": m.get("loss"),
            "seed": m.get("seed"), "epochs_completed": m.get("epochs_completed"),
            "best_epoch": m.get("best_epoch"), "stop_reason": m.get("stop_reason"),
            "best_val_pesq_training": m.get("best_val_pesq"),
            "config_path": m.get("config_path"), "overrides": m.get("overrides") or [],
            "run_id": m.get("run_id"), "device": m.get("device"), "torch": m.get("torch"),
            "source": m.get("source"),
            "train_settings": {k: t.get(k) for k in ("batch_size", "lr", "adam_betas",
                                                     "weight_decay", "max_epochs", "min_epochs",
                                                     "patience", "min_delta", "lr_factor",
                                                     "lr_patience", "min_lr", "optimizer",
                                                     "scheduler", "lr_gamma", "discriminator",
                                                     "data.segment_seconds", "data.remix_prob",
                                                     "data.remix_snr_db", "data.gain_db")
                               if k in t},
            "weights_mb": round(WEIGHTS.stat().st_size / 2 ** 20, 2)}


def render_submissions(data: dict) -> str:
    lines = [f"팀명: **{data['team']}** · 저장소: {data['repository']}", "",
             "| 회차 | 태그 | 커밋 | 런어스 제출일 | 모델 | Validation PESQ | Validation SNR (dB) "
             "| Train PESQ | Train SNR (dB) | 히든 테스트 PESQ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(data["rounds"], key=lambda r: r["round"]):
        v, tr, lb = r.get("validation") or {}, r.get("train") or {}, r.get("leaderboard") or {}
        lines.append("| {n} | `{tag}` | {c} | {sub} | {m} | {vp} | {vs} | {tp} | {ts} | {lb} |".format(
            n=r["round"], tag=r["tag"], c=f"`{r['commit'][:7]}`" if r.get("commit") else "태그 참조",
            sub=r.get("submitted_at") or "-", m=r.get("method_summary") or r["weights"]["model"],
            vp=f"{v['pesq']:.3f}" if v else "-", vs=f"{v['snr_db']:.2f}" if v else "-",
            tp=f"{tr['pesq']:.3f}" if tr else "-", ts=f"{tr['snr_db']:.2f}" if tr else "-",
            lb=f"{lb['pesq']:.3f}" if lb.get("pesq") is not None else "-"))
    lines += ["", "평가할 코드 버전 = 해당 회차 **태그**가 가리키는 커밋 (`git checkout <태그>`). "
              "점수는 `denoise.py`가 쓰는 16비트 WAV 그대로 채점한 Validation 164개·Train 1,296개 "
              "전체의 파일별 평균이며, 상세 결과는 [`results/submissions/`](results/submissions/) "
              "(train은 `roundN/train/`)."]
    return "\n".join(lines)


def render_reproduce(data: dict) -> str:
    if not data["rounds"]:
        return "_아직 제출 기록이 없습니다._"
    r = max(data["rounds"], key=lambda r: r["round"])
    w, v = r["weights"], r.get("validation") or {}
    cmd = "python train.py" + (f" --config {w['config_path']}" if w.get("config_path") else "") \
        + "".join(f" --set {s}" for s in w.get("overrides") or [])
    ts = w.get("train_settings") or {}
    settings = ", ".join(f"{k}={v}" for k, v in ts.items()) or "-"
    rows = [
        ("제출 회차", f"{r['round']}회차 (`{r['tag']}`)"),
        ("모델", f"`{w['model']}` {json.dumps(w['model_args'], ensure_ascii=False)}, "
                f"파라미터 {w['parameters']:,}개" if w.get("parameters") else f"`{w['model']}`"),
        ("가중치 파일", f"`weights/submission.pt` ({w['weights_mb']} MB, 설정 포함)"),
        ("손실 함수", f"`{w.get('loss')}`"),
        ("seed", str(w.get("seed"))),
        ("학습 epoch", f"{w.get('epochs_completed')} epoch 실행 ({w.get('stop_reason')}), "
                     f"**{w.get('best_epoch')} epoch 가중치 선택** (Validation PESQ 최고)"),
        ("선택 가중치의 검증 점수", f"학습 중 Validation PESQ-WB {w.get('best_val_pesq_training'):.4f} "
                             f"(float 출력) · 제출 경로(16비트 WAV) {v.get('pesq', float('nan')):.4f}"
         if w.get("best_val_pesq_training") is not None else f"{v.get('pesq')}"),
        ("학습 설정", settings),
        ("학습 명령", f"`{cmd}`"),
        ("학습 환경", f"PyTorch {w.get('torch')}, {w.get('device')}"),
        ("출처", w.get("source") or w.get("run_id") or "-"),
    ]
    return "\n".join(["| 항목 | 값 |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows])


def write_readme(data: dict) -> None:
    text = README.read_text(encoding="utf-8")
    for name, render in (("submissions", render_submissions), ("reproduce", render_reproduce)):
        start, end = BLOCKS[name]
        if start not in text or end not in text:
            sys.exit(f"README.md has no {start} ... {end} block")
        a, b = text.index(start) + len(start), text.index(end)
        text = text[:a] + "\n" + render(data) + "\n" + text[b:]
    README.write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("round", type=int)
    p.add_argument("--summary", help="one-line description of the model for the table")
    for name in ("tag", "push"):
        p = sub.add_parser(name); p.add_argument("round", type=int)
    p = sub.add_parser("submitted"); p.add_argument("round", type=int)
    p.add_argument("--date", default=date.today().isoformat())
    p = sub.add_parser("leaderboard"); p.add_argument("round", type=int)
    p.add_argument("--pesq", type=float, required=True); p.add_argument("--snr", type=float)
    sub.add_parser("readme")
    args = ap.parse_args(argv)

    data = load_submissions()
    data.update({"challenge": CHALLENGE, "deadline": DEADLINE, "repository": REPO_URL})
    data["team"] = data.get("team") or TEAM

    if args.cmd == "prepare":
        both = require_fingerprint(args.round)
        r = get_round(data, args.round)
        if r is None:
            r = {"round": args.round}
            data["rounds"].append(r)
        keys = ("files", "pesq", "snr_db", "noisy_pesq", "noisy_snr_db")
        r.update({"tag": tag_name(args.round), "weights": weights_info(),
                  "results": f"results/submissions/round{args.round}/",
                  "validation": {k: both["validation"][k] for k in keys},
                  "train": {k: both["train"][k] for k in keys},
                  "code_fingerprint": both["validation"]["fingerprint"]})
        if args.summary:
            r["method_summary"] = args.summary
        for k in ("commit", "submitted_at", "leaderboard"):
            r.setdefault(k, None)
        save_submissions(data)
        write_readme(data)
        print(f"round {args.round} recorded: validation PESQ {r['validation']['pesq']:.4f} "
              f"SNR {r['validation']['snr_db']:.2f} dB, train PESQ {r['train']['pesq']:.4f}\n"
              f"Next: commit, then  python tools/release.py tag {args.round}")

    elif args.cmd == "tag":
        if get_round(data, args.round) is None:
            sys.exit(f"round {args.round} not in submissions.json; run prepare first")
        require_fingerprint(args.round)
        if git("status", "--porcelain"):
            sys.exit("working tree is not clean: commit first (the tag must match what is pushed)")
        tag = tag_name(args.round)
        if git("tag", "--list", tag):
            sys.exit(f"tag {tag} already exists ({git('rev-list', '-n1', tag)[:7]})")
        git("tag", "-a", tag, "-m", f"{TEAM} submission round {args.round}")
        print(f"created {tag} -> {git('rev-parse', 'HEAD')[:7]}\n"
              f"Next:  python tools/release.py push {args.round}")

    elif args.cmd == "push":
        tag = tag_name(args.round)
        if not git("tag", "--list", tag):
            sys.exit(f"no tag {tag}; run  python tools/release.py tag {args.round}")
        scan = subprocess.run([sys.executable, str(ROOT / "tools" / "secret_scan.py"), "--staged"])
        if scan.returncode:
            sys.exit("secret scan failed: nothing pushed")
        for ref in ("main", tag):
            subprocess.run(["git", "push", "origin", ref], cwd=ROOT, check=True)
        print(f"pushed main and {tag}. Next: python tools/check_submission.py --round {args.round}")

    elif args.cmd == "submitted":
        r = get_round(data, args.round)
        tag = tag_name(args.round)
        if r is None or not git("tag", "--list", tag):
            sys.exit(f"round {args.round} has no tag {tag} yet")
        r["commit"] = git("rev-list", "-n1", tag)
        r["submitted_at"] = args.date
        save_submissions(data)
        write_readme(data)
        print(f"round {args.round}: commit {r['commit'][:7]}, submitted {args.date}")

    elif args.cmd == "leaderboard":
        r = get_round(data, args.round)
        if r is None:
            sys.exit(f"round {args.round} not in submissions.json")
        r["leaderboard"] = {"pesq": args.pesq, "snr_db": args.snr}
        save_submissions(data)
        write_readme(data)
        print(f"round {args.round}: hidden-test PESQ {args.pesq}")

    else:
        save_submissions(data)
        write_readme(data)
        print("README blocks updated")


if __name__ == "__main__":
    main()
