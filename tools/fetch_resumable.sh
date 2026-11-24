#!/usr/bin/env bash
# Download one large file so that a dropped connection can neither stop it nor pass as "done".
#
#   bash tools/fetch_resumable.sh <url> <output file> <expected size in bytes>
#
# - every dropped connection is resumed from where it stopped (curl -C -), up to 200 times;
# - a connection that stalls below 20 kB/s for 60 s is closed and resumed;
# - each attempt is logged with curl's exit code to <output file>.log;
# - exit code 0 ONLY if the size equals the expected size AND (for .gz) the archive passes
#   gzip -t. Anything else exits 1, so a caller that reports "finished" also sees the failure.
set -u
url="$1"; out="$2"; want="$3"; log="$out.log"
size() { stat -c %s "$out" 2>/dev/null || echo 0; }

for attempt in $(seq 1 200); do
    have=$(size)
    [ "$have" -ge "$want" ] && break
    curl -sS -L -C - --retry 5 --retry-delay 5 --retry-all-errors \
         --speed-limit 20000 --speed-time 60 -o "$out" "$url" 2>>"$log"
    code=$?
    echo "$(date '+%F %T') attempt $attempt: curl exit $code, $(size) of $want bytes" | tee -a "$log"
    [ "$code" -ne 0 ] && sleep 5
done

have=$(size)
if [ "$have" != "$want" ]; then
    echo "FAILED: $have of $want bytes after $attempt attempts" | tee -a "$log"
    exit 1
fi
case "$out" in
    *.gz|*.tgz)
        if ! gzip -t "$out" 2>>"$log"; then
            echo "FAILED: size is right but the archive is damaged (gzip -t)" | tee -a "$log"
            exit 1
        fi ;;
esac
echo "COMPLETE: $have bytes, verified" | tee -a "$log"
