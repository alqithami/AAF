#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ARCHIVE="aaf_jii_github_min_bundle.tar.xz"
cat source_bundle/aaf_jii_github_min_bundle.tar.xz.b64.part-* | tr -d '\r\n\t ' | base64 --decode > "$ARCHIVE"
echo "dbc7c58dcc10c77839dcfc35e5e44af8df6975ed7bc1752d2033541c9a49ccc9  $ARCHIVE" | sha256sum --check
rm -rf restored_source
mkdir -p restored_source
tar -xJf "$ARCHIVE" -C restored_source
printf 'Restored and verified: %s/restored_source\n' "$ROOT"
