#!/usr/bin/env bash
# Pull a branch (default main) into this checkout on the H20, over HTTPS, with retries:
#   ssh -p PORT root@HOST 'cd /2023533015/VCC_Project && hpc/pull.sh'
# No SSH agent forwarding and no GitHub key: the repository is public. The container's DNS sometimes
# answers github.com with an address that does not connect, so each try goes to one of GitHub's own
# addresses in turn (every third try asks the DNS), and a try that hangs stops after 40 s.
# shortcut: fixed addresses; if they stop working, take new ones from https://api.github.com/meta ("web").
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
branch=${1:-main}
# git 2.56.1 on the shared disk. A container's own git (2.25.1) ignores the address and asks the DNS.
git=/2023533015/.local/git-2.56.1/bin/git
[[ -x $git ]] || git=git
# fetch over HTTPS; a separate push address stays as it is
"$git" remote set-url origin https://github.com/ENNCELADUS/gene-perturbation-prediction.git
try=0
for address in 20.27.177.113 20.205.243.166 dns 20.200.245.247 140.82.112.3 dns \
    20.27.177.113 140.82.114.4 dns 20.205.243.166 20.200.245.247 dns; do
  try=$((try + 1))
  pin=http.curloptResolve=github.com:443:$address
  [[ $address == dns ]] && pin=http.curloptResolve=
  if timeout 40 "$git" -c "$pin" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=20 \
      pull --ff-only origin "$branch"; then
    "$git" log -1 --oneline
    exit 0
  fi
  echo "pull failed (try $try of 12, address: $address)" >&2
done
echo "every try failed" >&2
exit 1
