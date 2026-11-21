"""Block what must never reach the public repository.

    python tools/secret_scan.py            # files git would publish (tracked + staged + untracked)
    python tools/secret_scan.py --staged   # only what is tracked or staged (pre-push hook)

Fails (exit 1) on:
  * credentials: private keys, cloud service-account JSON, API keys/tokens,
    SSH keys (google_compute_engine*, id_*), .env files
  * data: any audio file, anything under data/ or results/listening/ (students' recordings)
  * the AI usage report folder docs/ai_usage/ (submitted privately)
  * files over 95 MB (GitHub refuses them)
Installed as a git pre-push hook by `git config core.hooksPath .githooks`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import MAX_FILE_MB, git  # noqa: E402

SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "service account JSON": re.compile(rb'"type"\s*:\s*"service_account"'),
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_\-]{35}"),
    "GitHub token": re.compile(rb"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{36}\b|github_pat_[0-9A-Za-z_]{40,}"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "Anthropic/OpenAI key": re.compile(rb"\bsk-(?:ant-)?[0-9A-Za-z_\-]{20,}"),
    "OAuth refresh token": re.compile(rb'"refresh_token"\s*:\s*"1//'),
}
BAD_NAMES = re.compile(r"(^|/)(\.env(\..*)?|google_compute_engine.*|id_(rsa|ed25519|ecdsa)(\.pub)?"
                       r"|credentials\.json|application_default_credentials\.json|.*\.pem|.*\.ppk"
                       r"|.*\.p12|client_secret.*\.json)$", re.I)
AUDIO = re.compile(r"\.(wav|flac|mp3|ogg|m4a|npy)$", re.I)
SKIP_CONTENT = {".pt", ".png", ".jpg", ".hwpx", ".zip"}


def candidate_files(staged_only: bool) -> list[str]:
    files = git("ls-files", "--cached").split("\n")
    if not staged_only:
        files += git("ls-files", "--others", "--exclude-standard").split("\n")
    return sorted({f for f in files if f})


def scan(files: list[str], root: Path = ROOT) -> list[str]:
    problems = []
    for rel in files:
        p = root / rel
        if not p.is_file():
            continue
        if rel.startswith("data/") or AUDIO.search(rel):
            problems.append(f"{rel}: 데이터·음성 파일 (수강생 녹음, 공개 금지)")
            continue
        if rel.startswith(("results/listening/", "results/figures/")):
            problems.append(f"{rel}: 청취·스펙트로그램 결과 (수강생 녹음 포함)")
            continue
        if rel.startswith("docs/ai_usage/"):
            problems.append(f"{rel}: AI 활용 내역서는 비공개")
            continue
        if BAD_NAMES.search(rel):
            problems.append(f"{rel}: 인증·키 파일 이름")
            continue
        size_mb = p.stat().st_size / 2 ** 20
        if size_mb > MAX_FILE_MB:
            problems.append(f"{rel}: {size_mb:.0f} MB (GitHub 한도 100 MB)")
        if p.suffix.lower() in SKIP_CONTENT or rel == "tools/secret_scan.py" \
                or rel.startswith("tests/"):
            continue
        data = p.read_bytes()
        for name, pat in SECRET_PATTERNS.items():
            if pat.search(data):
                problems.append(f"{rel}: {name}로 보이는 내용")
    return problems


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    files = candidate_files("--staged" in args)
    problems = scan(files)
    for p in problems:
        print("[BLOCK]", p)
    print(f"secret scan: {len(files)} files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
