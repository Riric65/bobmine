#!/usr/bin/env bash
#
# bobmine - compile le mineur en une seule commande (Linux / macOS)
#
#   1. Installe Rust une fois : https://rustup.rs
#   2. Lance :  ./build.sh
#   3. Lance le mineur avec ton user_id (voir le message de fin)
#
set -e
cd "$(dirname "$0")"

echo "=================================================="
echo "  bobmine - compilation"
echo "=================================================="
echo

# --- Rust ---------------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
    echo "ERREUR : Rust n'est pas installe ('cargo' introuvable)."
    echo
    echo "Installe-le une seule fois, puis relance ce script :"
    echo "    https://rustup.rs"
    echo "  ou, sur Arch/Artix :  sudo pacman -S rust"
    echo "  ou, sur Debian/Ubuntu : sudo apt install cargo"
    exit 1
fi

# --- CUDA (optionnel : ~14x plus rapide) --------------------------------
NVCC=""
for cand in nvcc /opt/cuda/bin/nvcc /usr/local/cuda/bin/nvcc; do
    if command -v "$cand" >/dev/null 2>&1; then
        NVCC="$cand"
        break
    fi
done

if [ -z "$NVCC" ]; then
    echo "[i] Pas de CUDA detecte : le mineur sera en CPU (~230 M H/s)."
    echo "    Avec un GPU NVIDIA, installe le CUDA Toolkit et relance"
    echo "    ce script : le minage GPU (~3 GH/s, soit ~14x plus rapide)"
    echo "    s'active tout seul."
    echo "        https://developer.nvidia.com/cuda-downloads"
    echo
fi

# --- Build --------------------------------------------------------------
echo "Compilation (premier lancement : ~1 min)..."
echo
cargo build --release

echo
echo "Termine : ./target/release/bobminer"
if [ -n "$NVCC" ]; then
    echo "[OK] CUDA detecte : version GPU active (~3 GH/s, ~4000 BC/h)."
else
    echo "[i] Version CPU (~230 M H/s, ~260 BC/h)."
    echo "    -> CUDA Toolkit + relance de ce script pour le GPU (x14)."
fi
echo
echo "Pour lancer :"
echo "    ./target/release/bobminer --user-id TON_ID"
echo
echo "    TON_ID  = identifiant de ton compte BobPhone"
echo "    options : --help"
