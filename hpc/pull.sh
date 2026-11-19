#!/usr/bin/env bash
# Pull a branch (default main) into this checkout on the H20, over HTTPS, with retries:
#   ssh -p PORT root@HOST 'cd /2023533015/VCC_Project && hpc/pull.sh'
# No SSH agent forwarding and no GitHub key: the repository is public. The container's DNS sometimes
# answers github.com with an address that does not connect, and a connection sometimes times out, so a
# single pull fails about one time in two. Each try is short and gets a new DNS answer.
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
branch=${1:-main}
# fetch over HTTPS; a separate push address stays as it is
git remote set-url origin https://github.com/ENNCELADUS/gene-perturbation-prediction.git
for try in $(seq 1 12); do
  if timeout 40 git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=20 pull --ff-only origin "$branch"; then
    git log -1 --oneline
    exit 0
  fi
  echo "pull failed (try $try of 12)" >&2
  sleep 2
done
echo "every try failed" >&2
exit 1
