#!/bin/bash

# --- LibTorch のヘッダとライブラリパスを設定 ---
export CXXFLAGS="-I$HOME/libtorch/include -I$HOME/libtorch/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=0"
export LDFLAGS="-L$HOME/libtorch/lib -ltorch -ltorch_cpu -lc10"
export LD_LIBRARY_PATH=$HOME/libtorch/lib:$LD_LIBRARY_PATH

# --- ビルド処理 ---
cd ~/rcss/teams/policy_gradient/helios-base-policy-gradient/build

# 必要であれば CMake を再実行（初回 or キャッシュクリア後）
cmake \
  -DCMAKE_PREFIX_PATH=$HOME/libtorch \
  -DCMAKE_CXX_FLAGS="-I$HOME/libtorch/include -I$HOME/libtorch/include/torch/csrc/api/include" \
  -DLIBRCSC_INSTALL_DIR=$HOME/rcss/teams/policy_gradient \
  ..

# クリーン & 再ビルド
make clean
make -j$(nproc)
