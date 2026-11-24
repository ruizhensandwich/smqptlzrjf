#!/usr/bin/env bash
# Download one large file in parallel byte ranges (for servers that are slow per connection).
#
#   bash tools/fetch_segments.sh <url> <output file> <expected size in bytes> [segments=8]
#
# Each segment is resumed from where it stopped until it has exactly its length; a stalled
# connection (< 20 kB/s for 60 s) is closed and resumed. The parts are then joined.
# Exit code 0 ONLY if the joined file has the expected size AND (for .gz) passes gzip -t;
# anything else exits 1. Progress and every attempt go to <output file>.log.
set -u
url="$1"; out="$2"; want="$3"; n="${4:-8}"; log="$out.log"
part=$(( (want + n - 1) / n ))

fetch_part() {
    local k="$1" first=$(( $1 * part )) last=$(( ($1 + 1) * part - 1 )) f="$out.part$1" have code
    [ "$last" -ge "$want" ] && last=$(( want - 1 ))
    local need=$(( last - first + 1 ))
    for attempt in $(seq 1 200); do
        have=$(stat -c %s "$f" 2>/dev/null || echo 0)
        [ "$have" -ge "$need" ] && return 0
        curl -sS -L -r "$(( first + have ))-$last" --speed-limit 20000 --speed-time 60 "$url" >> "$f" 2>>"$log"
        code=$?
        echo "$(date '+%F %T') part $k attempt $attempt: curl exit $code, $(stat -c %s "$f") of $need" >> "$log"
        [ "$code" -ne 0 ] && sleep 3
    done
    return 1
}

for k in $(seq 0 $(( n - 1 ))); do fetch_part "$k" & done
wait

: > "$out"
for k in $(seq 0 $(( n - 1 ))); do cat "$out.part$k" >> "$out"; done
have=$(stat -c %s "$out")
if [ "$have" != "$want" ]; then
    echo "FAILED: $have of $want bytes" | tee -a "$log"
    exit 1
fi
case "$out" in
    *.gz|*.tgz)
        if ! gzip -t "$out" 2>>"$log"; then
            echo "FAILED: size is right but the archive is damaged (gzip -t)" | tee -a "$log"
            exit 1
        fi ;;
esac
rm -f "$out".part*
echo "COMPLETE: $have bytes, verified" | tee -a "$log"
