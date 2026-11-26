#!/usr/bin/env bash
# Download and compile the original POPSTAR program of Resende & Werneck.
#
#   tools/get_popstar.sh                 download the distribution and build third_party/popstar/popstar
#   tools/get_popstar.sh popstar.tar.gz  build from a copy of the distribution you already have
#
# POPSTAR is not part of this repository: it is the authors' program and must be obtained from
# their page, https://mauricio.resende.info/popstar/downloads.html.  The source is compiled
# without any change.  The only compiler option added to the authors' makefile is -std=gnu++98,
# because one call in a test routine that this study never executes (popstar.cpp, line 459)
# passes `false` as a null pointer, which later C++ standards reject.
#
# The pre-compiled Linux binary offered on the same page is a 32-bit executable built in 2002;
# it is not used here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/popstar"
URL="https://mauricio.resende.info/popstar/files/popstar.tar.gz"
SHA256="37f0627c35481f5b76ed50756289f472579a918e1d2bb12803f06a0e2ec2f43d"   # distribution used in the paper (version 20041104)
mkdir -p "$DEST"
TARBALL="${1:-$DEST/popstar.tar.gz}"
if [ ! -f "$TARBALL" ]; then
    echo "Downloading $URL"
    if command -v curl >/dev/null; then curl -fsSL -o "$TARBALL" "$URL"; else wget -q -O "$TARBALL" "$URL"; fi
fi
if command -v sha256sum >/dev/null; then
    GOT="$(sha256sum "$TARBALL" | cut -d' ' -f1)"
    if [ "$GOT" != "$SHA256" ]; then
        echo "WARNING: the archive differs from the one used in the paper (sha256 $GOT)." >&2
    fi
fi
rm -rf "$DEST/src"
tar -xzf "$TARBALL" -C "$DEST"
make -C "$DEST/src" FLAGS="-O3 -std=gnu++98 -w" LIBS="-lm"
cp "$DEST/src/popstar" "$DEST/popstar"
echo "Built $DEST/popstar"
"$DEST/popstar" 2>&1 | head -3 || true
