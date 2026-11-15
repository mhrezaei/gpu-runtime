#!/usr/bin/env bash
# Registry download speed test: how fast a host gets the GPU image's layers from each registry, without docker.
# It reads the image manifest anonymously, then times plain HTTPS downloads of layer blobs (the same bytes docker pull
# fetches, through the same CDN redirects): one stream on the largest layer, then three streams at once (docker pulls
# at most 3 layers in parallel). Nothing is written to disk.
#
# usage: registry-speed.sh <image-ref> [<image-ref> ...]
#   e.g. registry-speed.sh docker.io/mhadirezaei/oneart-gpu:bc560f3cb410 ghcr.io/<user>/oneart-gpu:bc560f3cb410
# env:   MB=1024   megabytes read per stream (a Range request; 0 = the whole blob)
#        SECS=120  per-download time cap (the speed is still reported)
# Public images only (anonymous pull, as Clore uses them). Needs bash, curl, python3.
set -euo pipefail
MB=${MB:-1024}
SECS=${SECS:-120}
ACCEPT='application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json'

# registry host, repository and reference (tag or digest) of an image ref
split() {
  local ref=$1 host rest
  host=${ref%%/*}; rest=${ref#*/}
  [[ $host == docker.io ]] && host=registry-1.docker.io
  if [[ $rest == *@* ]]; then echo "$host" "${rest%@*}" "${rest#*@}"; else echo "$host" "${rest%:*}" "${rest##*:}"; fi
}

# anonymous pull token from the registry's own challenge (Docker Hub, GHCR and Distribution all answer the same way)
token() {
  local host=$1 repo=$2 challenge realm service
  challenge=$(curl -sI "https://$host/v2/" | tr -d '\r' | grep -i '^www-authenticate:' || true)
  [[ -z $challenge ]] && { echo ""; return; }
  realm=$(sed -E 's/.*realm="([^"]+)".*/\1/' <<<"$challenge")
  service=$(sed -nE 's/.*service="([^"]+)".*/\1/p' <<<"$challenge")
  curl -s "$realm?service=$service&scope=repository:$repo:pull" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("token") or d.get("access_token") or "")'
}

# one timed download: prints "<MB/s> <MB read> <seconds>"
fetch() {
  local url=$1 auth=$2 range=() out
  ((MB > 0)) && range=(-r "0-$((MB * 1024 * 1024 - 1))")
  out=$(curl -sL --max-time "$SECS" "${range[@]}" -H "Authorization: Bearer $auth" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$url" || true)
  read -r bytes secs code <<<"$out"
  python3 -c "b=$bytes;s=max($secs,0.001);print(f'{b/1e6/s:8.1f} MB/s  {b/1e6:7.0f} MB  {s:6.1f} s  http {\"$code\"}')"
}

for ref in "$@"; do
  read -r host repo tag < <(split "$ref")
  echo "== $ref"
  auth=$(token "$host" "$repo")
  manifest=$(curl -sL -H "Accept: $ACCEPT" -H "Authorization: Bearer $auth" "https://$host/v2/$repo/manifests/$tag")
  # an index (multi-arch) points at the linux/amd64 manifest
  sub=$(python3 -c 'import sys,json;m=json.loads(sys.stdin.read());print(next((x["digest"] for x in m.get("manifests",[]) if x.get("platform",{}).get("architecture")=="amd64"),""))' <<<"$manifest")
  [[ -n $sub ]] && manifest=$(curl -sL -H "Accept: $ACCEPT" -H "Authorization: Bearer $auth" "https://$host/v2/$repo/manifests/$sub")
  mapfile -t layers < <(python3 -c 'import sys,json;[print(x["digest"],x["size"]) for x in sorted(json.loads(sys.stdin.read())["layers"],key=lambda x:-x["size"])[:3]]' <<<"$manifest")
  ((${#layers[@]})) || { echo "   no layers (private image or wrong ref?)"; continue; }
  first=${layers[0]%% *}
  echo "   one stream, largest layer (${layers[0]##* } bytes):"
  echo "   $(fetch "https://$host/v2/$repo/blobs/$first" "$auth")"
  echo "   three streams at once (the three largest layers):"
  pids=()
  for l in "${layers[@]}"; do
    (echo "   $(fetch "https://$host/v2/$repo/blobs/${l%% *}" "$auth")") &
    pids+=($!)
  done
  wait "${pids[@]}"
done
