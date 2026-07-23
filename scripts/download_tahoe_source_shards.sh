#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR [WORKERS]" >&2
  exit 2
fi

output_dir=$1
workers=${2:-32}
shard_count=3388
revision="2dc57900b7981cfcf5e211527169a0b006546a95"
base_url="https://hf-mirror.com/datasets/tahoebio/Tahoe-100M/resolve/$revision/data"

mkdir -p "$output_dir"

valid_parquet() {
  local path=$1
  [[ -s "$path" ]] \
    && [[ $(head -c 4 "$path") == "PAR1" ]] \
    && [[ $(tail -c 4 "$path") == "PAR1" ]]
}

download_shard() {
  local shard=$1
  local name destination partial url
  name=$(printf "train-%05d-of-%05d.parquet" "$shard" "$shard_count")
  destination="$output_dir/$name"
  partial="$destination.part"
  url="$base_url/$name"
  if valid_parquet "$destination"; then
    return
  fi
  local attempt
  for attempt in 1 2 3 4 5; do
    if wget --quiet --continue --tries=10 --timeout=60 -O "$partial" "$url"; then
      break
    fi
    if [[ "$attempt" -eq 5 ]]; then
      echo "failed after 5 attempts: $url" >&2
      return 1
    fi
    sleep $((2 ** (attempt - 1)))
  done
  mv "$partial" "$destination"
}

export output_dir shard_count base_url
export -f download_shard valid_parquet
seq 0 $((shard_count - 1)) \
  | xargs -n 1 -P "$workers" bash -c 'download_shard "$1"' _

actual_count=$(find "$output_dir" -maxdepth 1 -type f -name '*.parquet' | wc -l)
if [[ "$actual_count" -ne "$shard_count" ]]; then
  echo "expected $shard_count shards, found $actual_count" >&2
  exit 1
fi
while IFS= read -r -d '' path; do
  if ! valid_parquet "$path"; then
    echo "invalid parquet file: $path" >&2
    exit 1
  fi
done < <(find "$output_dir" -maxdepth 1 -type f -name '*.parquet' -print0)

hashes_partial="$output_dir/source_sha256.txt.part"
hash_workers=$workers
if [[ "$hash_workers" -gt 16 ]]; then
  hash_workers=16
fi
find "$output_dir" -maxdepth 1 -type f -name '*.parquet' -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "$hash_workers" sha256sum \
  | sort > "$hashes_partial"
mv "$hashes_partial" "$output_dir/source_sha256.txt"
printf '%s\n' "$actual_count" > "$output_dir/source_shard_count.txt"
printf '%s\n' "$revision" > "$output_dir/source_revision.txt"
