#!/usr/bin/env bash
# Koi bhi tracked file had se bari ho to fail. Wazan Release par jaate hain, git mein nahi.
# Had 25 MB rakhi hai — GitHub ki 100 MB se kaafi neeche, taake warning waqt par mile.
set -euo pipefail
LIMIT_MB="${LIMIT_MB:-25}"
limit=$(( LIMIT_MB * 1024 * 1024 ))
bad=0
while IFS= read -r f; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f")
  if [ "$sz" -gt "$limit" ]; then
    echo "::error file=${f}::${f} is $((sz / 1048576)) MB — repo limit is ${LIMIT_MB} MB. Publish it as a Release asset, not a commit."
    bad=1
  fi
done < <(git ls-files)
if [ "$bad" -eq 0 ]; then echo "guard: no tracked file over ${LIMIT_MB} MB"; fi
exit "$bad"
