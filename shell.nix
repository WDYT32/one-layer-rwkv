{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    ninja
    gcc
  ];

  # Необхідно для того, щоб встановлені через uv (PyPI) пакети, 
  # такі як PyTorch, могли знайти системні бібліотеки
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];
}
