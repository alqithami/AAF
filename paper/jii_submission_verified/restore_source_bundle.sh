#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ARCHIVE="aaf_jii_verified_text_bundle.tar.xz"
tr -d '\r\n\t ' < source_bundle/aaf_jii_verified_text_bundle.tar.xz.b64 | base64 --decode > "$ARCHIVE"
echo "d936fe292fd20b67e8ece7f01680e30d64698b13ef6563fb08aa0069003b8434  $ARCHIVE" | sha256sum --check
rm -rf restored_source
mkdir -p restored_source
tar -xJf "$ARCHIVE" -C restored_source
printf 'Restored and verified: %s/restored_source\n' "$ROOT"
