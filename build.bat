@echo off
setlocal
REM Crear carpeta bin si no existe
if not exist bin mkdir bin

REM Compilar (C++17, optimizado) con include a Eigen e include/
g++ -std=c++17 -O3 -march=native -DEIGEN_NO_DEBUG ^
 -I "include" -I "third_party\eigen" ^
 src\main.cpp src\io.cpp src\split.cpp src\als.cpp src\sgd.cpp src\metrics.cpp src\utils.cpp ^
 -o bin\recsys.exe

if %errorlevel% neq 0 (
  echo [BUILD] Error al compilar.
  exit /b 1
) else (
  echo [BUILD] OK: bin\recsys.exe
)
