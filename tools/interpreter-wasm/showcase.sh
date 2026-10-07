#!/usr/bin/env bash
# Builds and previews the GitHub Pages showcase in `docs/`.
#
#   tools/interpreter-wasm/showcase.sh build          # Wasm module + docs/
#   tools/interpreter-wasm/showcase.sh check          # docs/ vs. native tool
#   tools/interpreter-wasm/showcase.sh preview [port] # serve docs/ like Pages
#
# `build` uses an activated emsdk, or the one in $EMSDK or third_party/emsdk.
# It builds native TableGen (release preset), the Wasm module (wasm preset),
# and replaces docs/ with the page, the module and the examples. Examples lose
# their RUN and CHECK lines, and their RUN budget goes to examples/index.json.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
web="$root/tools/interpreter-wasm/web"
docs="$root/docs"
prefix="mlir-interpreter"

activate_emsdk() {
  command -v emcc > /dev/null && [[ -n "${EMSDK:-}" ]] && return
  local emsdk="${EMSDK:-$root/third_party/emsdk}"
  if [[ ! -f "$emsdk/emsdk_env.sh" ]]; then
    echo "error: no emsdk in $emsdk. Install it with:" >&2
    echo "  git clone https://github.com/emscripten-core/emsdk.git third_party/emsdk" >&2
    echo "  third_party/emsdk/emsdk install 6.0.11 && third_party/emsdk/emsdk activate 6.0.11" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$emsdk/emsdk_env.sh" > /dev/null 2>&1
}

package_examples() {
  mkdir -p "$docs/examples"
  local entries=() file name budget
  for file in "$root"/examples/*.mlir; do
    name="$(basename "$file" .mlir)"
    budget="$(sed -n 's/^\/\/ RUN:.*--budget=\([0-9]\+\).*/\1/p' "$file" | head -n 1)"
    entries+=("{\"name\": \"$name\", \"budget\": ${budget:-1000}}")
    # Drop lit directives, then the blank lines they leave at the top.
    grep -Ev '^[[:space:]]*// (RUN|CHECK)' "$file" | sed '/./,$!d' \
      > "$docs/examples/$name.mlir"
  done
  local IFS=,
  printf '[%s]\n' "${entries[*]}" > "$docs/examples/index.json"
}

build() {
  cd "$root"
  activate_emsdk
  # Configure each preset only once: configuring repoints the root
  # compile_commands.json, which should stay on the developer's build.
  [[ -f build/release/build.ninja ]] || cmake --preset release > /dev/null
  cmake --build --preset release --target llvm-min-tblgen mlir-tblgen
  [[ -f build/wasm/build.ninja ]] || cmake --preset wasm > /dev/null
  cmake --build --preset wasm --target interpreter-wasm

  rm -rf "$docs"
  mkdir -p "$docs"
  cp "$web"/{index.html,style.css,main.js,worker.js} "$docs/"
  cp build/wasm/tools/interpreter-wasm/interpreter.{js,wasm} "$docs/"
  package_examples
  # Serve files as they are instead of through Jekyll.
  touch "$docs/.nojekyll"
  echo "Built $docs"
}

check() {
  cd "$root"
  [[ -f "$docs/interpreter.js" ]] || { echo "error: run '$0 build' first" >&2; exit 1; }
  cmake --build --preset release --target interpreter
  node tools/interpreter-wasm/parity.mjs "$docs" build/release/tools/interpreter/interpreter
}

preview() {
  local port="${1:-8000}" site
  [[ -f "$docs/index.html" ]] || { echo "error: run '$0 build' first" >&2; exit 1; }
  # Pages serves the repository under /$prefix/, so serve docs/ there too.
  site="$(mktemp -d)"
  trap 'rm -rf "$site"' EXIT
  ln -s "$docs" "$site/$prefix"
  echo "Serving http://127.0.0.1:$port/$prefix/"
  python3 -m http.server --bind 127.0.0.1 --directory "$site" "$port"
}

case "${1:-}" in
  build) build ;;
  check) check ;;
  preview) shift; preview "$@" ;;
  *) echo "usage: $0 build | check | preview [port]" >&2; exit 1 ;;
esac
