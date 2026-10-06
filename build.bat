@echo off
chcp 65001 >nul 2>nul
title bobmine - build
cd /d "%~dp0"

echo ==================================================
echo   bobmine - compilation
echo ==================================================
echo.

rem --- Rust ------------------------------------------------------------
where cargo >nul 2>nul
if errorlevel 1 (
    echo ERREUR : Rust n'est pas installe ^(cargo introuvable^).
    echo.
    echo Installe-le une seule fois, puis relance ce fichier :
    echo     https://rustup.rs
    echo ^(telecharge rustup-init.exe, valide avec Enter^)
    echo.
    pause
    exit /b 1
)

rem --- CUDA (optionnel : ~14x plus rapide) -----------------------------
set NVCC_FOUND=0
where nvcc >nul 2>nul
if not errorlevel 1 set NVCC_FOUND=1
if "%NVCC_FOUND%"=="0" if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set NVCC_FOUND=1

if "%NVCC_FOUND%"=="0" (
    echo [i] Pas de CUDA detecte : le mineur sera en CPU ^(environ 230 M H/s^).
    echo     Avec un GPU NVIDIA, installe le NVIDIA CUDA Toolkit puis
    echo     relance ce fichier : le minage GPU ^(environ 3 GH/s, soit
    echo     14x plus rapide^) s'active tout seul.
    echo         https://developer.nvidia.com/cuda-downloads
    echo.
)

rem --- Build -----------------------------------------------------------
echo Compilation ^(premier lancement : environ 1 min^)...
echo.
cargo build --release
if errorlevel 1 (
    echo.
    echo ERREUR pendant la compilation - voir le message ci-dessus.
    echo.
    echo Si lerreur parle de "linker", il manque les C++ Build Tools :
    echo     https://aka.ms/vs/17/release/vs_BuildTools.exe
    echo ^(coche "Desktop development with C++", puis relance^)
    echo.
    pause
    exit /b 1
)

echo.
echo Termine : target\release\bobminer.exe
echo.
if "%NVCC_FOUND%"=="1" (
    echo [OK] CUDA detecte : version GPU active ^(environ 3 GH/s^).
) else (
    echo [i] Version CPU uniquement ^(environ 230 M H/s^).
    echo     Pour 14x plus rapide : installe le CUDA Toolkit puis
    echo     relance ce fichier .bat.
)
echo.
echo Pour lancer :
echo     target\release\bobminer.exe --user-id TON_ID
echo.
echo     TON_ID = identifiant de ton compte BobPhone
echo     options : --help
echo.
pause
