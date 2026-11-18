#!/usr/bin/env bash
set -euo pipefail

# OpenSSL_jll 3.5.4 needs its matching libcrypto loaded before system OpenSSL.
# Prefer the project artifact directory for this Julia process only.
OPENSSL_LIB="${JULIA_OPENSSL_LIB:-}"
if [[ -z "$OPENSSL_LIB" ]]; then
    DEPOTS="${JULIA_DEPOT_PATH:-${HOME}/.julia}"
    IFS=: read -r -a DEPOT_LIST <<< "$DEPOTS"
    for depot in "${DEPOT_LIST[@]}"; do
        candidate="$depot/artifacts/7f39a18d94f87b5135df6731a327b61b8c463af6/lib"
        if [[ -r "$candidate/libcrypto.so.3" ]]; then
            OPENSSL_LIB="$candidate"
            break
        fi
    done
fi
JULIA_REAL_BIN="${JULIA_REAL_BIN:-julia}"

if [[ -n "$OPENSSL_LIB" && -r "$OPENSSL_LIB/libcrypto.so.3" ]]; then
    export LD_LIBRARY_PATH="$OPENSSL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$JULIA_REAL_BIN" "$@"
