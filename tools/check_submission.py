"""Pre-submission check: the 2nd challenge's rules, item by item, in the TA's place.

    python tools/check_submission.py --round 1 [--python "py -3.12"] [--n 0] [--keep]

  1. requirements.txt   fresh venv, `pip install -r requirements.txt` only
  2. 실행 방법          README's commands, `python denoise.py` in the repo root
  3. input -> output    every validation noisy WAV gets exactly one output directly
                        in output/ with the same name; no missing and no extra WAV
  4. 형식               16 kHz, mono, same sample count (and format) as the input
     4+                 re-score the outputs = results/submissions/roundN/
  5. GitHub / 제출 버전 public access, tag submission-N pushed = local tag,
                        README team / tag / reproduction info (seed, epochs, score),
                        saved results made with this code and these weights
  6. 가중치·공개 범위   weights/submission.pt in the tag, under 95 MB, no download
                        in the inference code, secret scan (keys, data, reports)
  7. AI 활용 내역서     docs/ai_usage/roundN.json complete, hwpx built
  8. 제출 횟수          at most one per week (first one not counted)

Code source: the tag submission-N if it exists (exactly what the TA clones),
otherwise the working tree as it would be committed, with a warning.
Exit code 0 = no FAIL. Needs network for pip and GitHub.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import (AI_USAGE, MAX_FILE_MB, REPO_URL, fingerprint, get_round,  # noqa: E402
                          git, load_submissions, round_dir, tag_name)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
DOWNLOAD_HINTS = re.compile(r"urllib|requests\.|torch\.hub|hf_hub|from_pretrained|wget|curl|"
                            r"https?://")


class Report:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []

    def add(self, item: str, status: str, detail: str = ""):
        self.items.append((item, status, detail))
        print(f"[{status}] {item}" + (f" — {detail}" if detail else ""), flush=True)

    def print(self) -> int:
        print("\n" + "=" * 72)
        for item, status, detail in self.items:
            print(f"[{status}] {item}")
            if detail and status != PASS:
                for line in detail.split("\n"):
                    print(f"         {line}")
        fails = sum(s == FAIL for _, s, _ in self.items)
        warns = sum(s == WARN for _, s, _ in self.items)
        print("=" * 72)
        print("READY TO SUBMIT" if not fails else "NOT READY", f"(FAIL {fails}, WARN {warns})")
        return fails


def sh(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def checkout_code(tag: str, dest: Path, rep: Report) -> str:
    if git("tag", "--list", tag):
        sh(["git", "clone", "-q", "--no-local", str(ROOT), str(dest)])
        sh(["git", "checkout", "-q", tag], cwd=dest)
        return f"tag {tag} ({git('rev-list', '-n1', tag)[:7]})"
    for rel in filter(None, git("ls-files", "--cached", "--others", "--exclude-standard").split("\n")):
        if (ROOT / rel).is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest / rel)
    rep.add("코드 출처", WARN, f"태그 {tag} 없음 → 작업 트리(커밋 예정 상태)로 점검"
            + (" · 커밋 안 된 변경 있음" if git("status", "--porcelain") else ""))
    return "working tree"


def check_formats(noisy: list[Path], out_dir: Path, rep: Report):
    import soundfile as sf
    produced = sorted(p.name for p in out_dir.iterdir() if p.is_file()) if out_dir.is_dir() else []
    nested = [p for p in out_dir.rglob("*") if p.is_file() and p.parent != out_dir] \
        if out_dir.is_dir() else []
    wavs = [p for p in produced if p.lower().endswith(".wav")]
    expected = sorted(f.name for f in noisy)
    missing = sorted(set(expected) - set(wavs))
    extra = sorted(set(wavs) - set(expected)) + [p.relative_to(out_dir).as_posix() for p in nested]
    rep.add("3. input의 모든 WAV → output/ 바로 아래 같은 이름 (누락·추가 없음)",
            PASS if not missing and not extra else FAIL,
            f"{len(expected)}개 입력, {len(wavs)}개 출력"
            + (f"\n누락: {missing[:5]}" if missing else "") + (f"\n추가: {extra[:5]}" if extra else ""))
    bad = []
    for f in noisy:
        o = out_dir / f.name
        if o.exists():
            a, b = sf.info(str(f)), sf.info(str(o))
            if b.samplerate != 16000 or b.channels != 1 or \
                    (a.samplerate, a.channels, a.frames, a.subtype) != \
                    (b.samplerate, b.channels, b.frames, b.subtype):
                bad.append(f"{f.name}: {a.samplerate}/{a.channels}/{a.frames}/{a.subtype} -> "
                           f"{b.samplerate}/{b.channels}/{b.frames}/{b.subtype}")
    rep.add("4. 16 kHz 모노, 입력과 같은 표본 수·포맷", PASS if not bad else FAIL,
            f"불일치 {len(bad)}개" + ("\n" + "\n".join(bad[:5]) if bad else ""))


def rescore(noisy: list[Path], out_dir: Path, round_no: int, rep: Report):
    item = "4+. 제출 코드 출력 재채점 = 저장된 Validation 결과 (PESQ 오류 없음)"
    summary = round_dir(round_no) / "summary.csv"
    if not summary.exists():
        rep.add(item, FAIL, f"{summary.relative_to(ROOT)} 없음 → python evaluate.py --round {round_no}")
        return
    try:
        import numpy as np
        import soundfile as sf
        from se.data import data_root, read_manifest
        from se.metrics import pesq_score, snr_db
    except ImportError as e:
        rep.add(item, WARN, f"개발 환경에 {e.name} 없음 → 재채점 생략")
        return
    root = data_root()
    clean_of = {Path(r["noisy_path"]).name: root / r["clean_path"]
                for r in read_manifest("validation", root)}
    ps, ss, errors = [], [], []
    for f in noisy:
        clean, _ = sf.read(str(clean_of[f.name]), dtype="float64")
        y, _ = sf.read(str(out_dir / f.name), dtype="float64")
        p, err = pesq_score(clean, y)
        ps.append(p)
        ss.append(snr_db(clean, y))
        if err:
            errors.append(f"{f.name}: {err}")
    with open(summary, encoding="utf-8-sig") as fh:
        saved = {r["method"]: r for r in csv.DictReader(fh)}["submission"]
    p_out, s_out = float(np.nanmean(ps)), float(np.mean(ss))
    detail = (f"재채점 PESQ {p_out:.4f} / SNR {s_out:.2f} dB, 저장된 결과 PESQ "
              f"{float(saved['pesq']):.4f} / SNR {float(saved['snr_db']):.2f} dB")
    if errors:
        rep.add(item, FAIL, detail + "\nPESQ 오류: " + "; ".join(errors[:5]))
    elif len(noisy) != int(saved["n"]):
        rep.add(item, WARN, detail + f" (일부 {len(noisy)}/{saved['n']}개만 — 전체는 --n 0)")
    else:
        ok = abs(p_out - float(saved["pesq"])) < 0.005 and abs(s_out - float(saved["snr_db"])) < 0.05
        rep.add(item, PASS if ok else FAIL, detail)


def check_remote(round_no: int, rep: Report):
    tag = tag_name(round_no)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    try:
        out = subprocess.run(["git", "-c", "credential.helper=", "ls-remote", REPO_URL,
                              "refs/heads/main", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
                             capture_output=True, text=True, env=env, timeout=60)
    except Exception as e:  # noqa: BLE001
        rep.add("5a. GitHub 저장소 공개 접근", FAIL, f"{type(e).__name__}: {e}")
        return
    if out.returncode != 0:
        rep.add("5a. GitHub 저장소 공개 접근", FAIL, f"로그인 없이 접근 불가: {out.stderr.strip()[:200]}")
        return
    rep.add("5a. GitHub 저장소 공개 접근", PASS, REPO_URL)
    refs = dict(reversed(line.split("\t")) for line in out.stdout.strip().split("\n") if line)
    remote = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
    local = git("rev-list", "-n1", tag, check=False) if git("tag", "--list", tag) else None
    if not local:
        rep.add(f"5b. 제출 버전 태그 {tag}", FAIL, f"로컬 태그 없음 → python tools/release.py tag {round_no}")
    elif not remote:
        rep.add(f"5b. 제출 버전 태그 {tag}", FAIL, f"GitHub에 없음 → python tools/release.py push {round_no}")
    elif remote != local:
        rep.add(f"5b. 제출 버전 태그 {tag}", FAIL, f"GitHub {remote[:7]} ≠ 로컬 {local[:7]}")
    else:
        rep.add(f"5b. 제출 버전 태그 {tag}", PASS, f"GitHub = 로컬 = {local[:7]}")


def check_readme(round_no: int, code_dir: Path, rep: Report):
    tag = tag_name(round_no)
    readme = (code_dir / "README.md").read_text(encoding="utf-8")
    sub_path = code_dir / "submissions.json"
    sub = json.loads(sub_path.read_text(encoding="utf-8")) if sub_path.exists() else {"rounds": []}
    problems = []
    if get_round(sub, round_no) is None:
        problems.append(f"submissions.json에 {round_no}회차 없음 → tools/release.py prepare {round_no}")
    if not sub.get("team") or sub["team"] not in readme:
        problems.append("README에 팀명 없음")
    if f"`{tag}`" not in readme:
        problems.append(f"README 표에 {tag} 없음")
    for need in ("seed", "학습 epoch", "선택 가중치의 검증 점수", "weights/submission.pt",
                 "pip install -r requirements.txt", "python denoise.py"):
        if need not in readme:
            problems.append(f"README에 '{need}' 없음")
    rep.add("5c. README: 팀명·태그·실행 명령·가중치 경로·seed·epoch·검증 점수",
            PASS if not problems else FAIL, "\n".join(problems))
    fp_code = fingerprint(code_dir)
    problems = []
    for split, folder in (("validation", round_dir(round_no)), ("train", round_dir(round_no) / "train")):
        saved = folder / "run_config.json"
        if not saved.exists():
            problems.append(f"{split}: {folder.relative_to(ROOT)} 없음")
            continue
        info = json.loads(saved.read_text(encoding="utf-8"))
        if info.get("code_fingerprint") != fp_code:
            problems.append(f"{split}: 결과 {info.get('code_fingerprint')} / 코드 {fp_code}")
        if info.get("subset") != "all":
            problems.append(f"{split}: 일부 파일({info.get('files')}개)만 평가됨")
    rep.add("5d. 저장된 Validation·Train 결과 = 이 코드와 가중치", PASS if not problems else FAIL,
            f"지문 {fp_code}" if not problems else
            "\n".join(problems) + f"\n→ python evaluate.py --round {round_no}")


def check_weights_and_privacy(code_dir: Path, rep: Report):
    w = code_dir / "weights" / "submission.pt"
    if not w.is_file():
        rep.add("6a. 제출 가중치 weights/submission.pt", FAIL, "제출 코드에 없음")
    else:
        mb = w.stat().st_size / 2 ** 20
        rep.add("6a. 제출 가중치 weights/submission.pt", PASS if mb <= MAX_FILE_MB else FAIL,
                f"{mb:.2f} MB")
    hits = []
    for p in [code_dir / "denoise.py", *sorted((code_dir / "se").glob("*.py"))]:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if DOWNLOAD_HINTS.search(code):
                hits.append(f"{p.relative_to(code_dir).as_posix()}:{i}: {line.strip()[:80]}")
    rep.add("6b. 추론 코드에 다운로드 없음", PASS if not hits else WARN, "\n".join(hits[:5]))
    from tools.secret_scan import scan
    problems = scan(git("ls-files", "--cached").split("\n"))
    rep.add("6c. 공개 범위: 키·데이터·내역서·100 MB 초과 파일 없음", PASS if not problems else FAIL,
            "\n".join(problems[:8]))


def check_ai_usage(round_no: int, rep: Report):
    spec_path = AI_USAGE / f"round{round_no}.json"
    if not spec_path.exists():
        rep.add("7. AI 활용 내역서", FAIL, f"{spec_path.relative_to(ROOT)} 없음")
        return
    from tools.build_ai_usage import validate_spec
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    problems = validate_spec(spec)
    hwpx = AI_USAGE / spec.get("output", "")
    if not hwpx.is_file():
        problems.append(f"{spec.get('output')} 미생성 → python tools/build_ai_usage.py "
                        f"docs/ai_usage/round{round_no}.json")
    elif hwpx.stat().st_mtime < spec_path.stat().st_mtime:
        problems.append(f"{hwpx.name}가 round{round_no}.json보다 오래됨 → 다시 생성")
    rep.add("7. AI 활용 내역서", PASS if not problems else FAIL, "\n".join(problems))


def check_weekly(round_no: int, rep: Report):
    done = sorted((r for r in load_submissions()["rounds"]
                   if r.get("submitted_at") and r["round"] != round_no), key=lambda r: r["round"])
    if not done:
        rep.add("8. 제출 횟수 (주 1회, 최초 제출 제외)", PASS, "최초 제출")
        return
    week = date.today().isocalendar()[:2]
    counted = [r for r in done[1:] if date.fromisoformat(r["submitted_at"]).isocalendar()[:2] == week]
    rep.add("8. 제출 횟수 (주 1회, 최초 제출 제외)", PASS if not counted else WARN,
            "" if not counted else f"이번 주에 이미 {counted[0]['round']}회차 제출 — 완화 공지가 없으면 다음 주에")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n", type=int, default=0, help="validation files to run (0 = all)")
    ap.add_argument("--python", default=sys.executable, help='interpreter for the venv, e.g. "py -3.12"')
    ap.add_argument("--skip-github", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    args = ap.parse_args(argv)

    rep = Report()
    tmp = Path(tempfile.mkdtemp(prefix="se_submit_"))
    repo = tmp / "repo"
    repo.mkdir()
    try:
        print("code:", checkout_code(tag_name(args.round), repo, rep))
        venv = tmp / "venv"
        try:
            sh([*shlex.split(args.python), "-m", "venv", str(venv)])
            py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            sh([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check",
                "-r", "requirements.txt"], cwd=repo)
            ver = sh([str(py), "-c", "import sys,torch,numpy,scipy,soundfile;print(sys.version.split()[0],"
                      "torch.__version__,numpy.__version__,scipy.__version__,soundfile.__version__)"]
                     ).stdout.split()
            rep.add("1. requirements.txt로 패키지 설치", PASS,
                    f"Python {ver[0]}: torch {ver[1]}, numpy {ver[2]}, scipy {ver[3]}, soundfile {ver[4]}")
        except subprocess.CalledProcessError as e:
            rep.add("1. requirements.txt로 패키지 설치", FAIL, (e.stderr or e.stdout)[-800:])
            return rep.print()

        from se.data import data_root
        noisy = sorted((data_root() / "validation" / "noisy").glob("*.wav"))
        noisy = noisy[: args.n] if args.n else noisy
        (repo / "input").mkdir(exist_ok=True)
        for f in noisy:
            shutil.copy2(f, repo / "input" / f.name)
        out_dir = repo / "output"
        if out_dir.is_dir():                      # keep .gitkeep, nothing else expected
            for p in out_dir.iterdir():
                if p.name != ".gitkeep":
                    rep.add("출력 폴더 초기 상태", WARN, f"저장소의 output/에 {p.name}가 있음")
        readme = (repo / "README.md").read_text(encoding="utf-8")
        documented = all(c in readme for c in ("pip install -r requirements.txt", "python denoise.py"))
        try:
            run = sh([str(py), "denoise.py"], cwd=repo)
            warn = [ln for ln in run.stderr.splitlines() if ln.startswith(("[WARN]", "[FAIL]"))]
            rep.add("2. 안내한 실행 방법대로 동작", PASS if documented and not warn else
                    (WARN if documented else FAIL), run.stdout.strip().splitlines()[-1]
                    + ("" if documented else "\nREADME에 설치·실행 명령이 없음")
                    + ("\n" + "\n".join(warn[:5]) if warn else ""))
        except subprocess.CalledProcessError as e:
            rep.add("2. 안내한 실행 방법대로 동작", FAIL, (e.stderr or e.stdout)[-800:])
            return rep.print()
        if (out_dir / ".gitkeep").exists():
            (out_dir / ".gitkeep").unlink()       # the TA's output/ holds only WAVs
        check_formats(noisy, out_dir, rep)
        rescore(noisy, out_dir, args.round, rep)
        if args.skip_github:
            rep.add("5a/5b. GitHub 공개 접근·태그", WARN, "--skip-github 로 생략")
        else:
            check_remote(args.round, rep)
        check_readme(args.round, repo, rep)
        check_weights_and_privacy(repo, rep)
        check_ai_usage(args.round, rep)
        check_weekly(args.round, rep)
        return rep.print()
    finally:
        if args.keep:
            print("kept:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
