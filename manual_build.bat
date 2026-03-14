@echo off
set "MSVC_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
set "SITE_PACKAGES=C:\Users\ROG\miniforge3\Lib\site-packages"
set "PY_INC=C:\Users\ROG\miniforge3\Include"
set "PY_LIB_DIR=C:\Users\ROG\miniforge3\libs"
set "T_INC=%SITE_PACKAGES%\torch\include"
set "T_LIB=%SITE_PACKAGES%\torch\lib"

echo [1/1] Compiling for Blackwell with Raw Headers...

"%CUDA_PATH%\nvcc.exe" ^
    -shared D:\AIEngineer\fused_act.cu ^
    -o fused_op.pyd ^
    --compiler-bindir "%MSVC_PATH%" ^
    -std=c++17 ^
    -O3 --use_fast_math ^
    -gencode=arch=compute_120,code=sm_120 ^
    -I"%PY_INC%" ^
    -I"%T_INC%" ^
    -I"%T_INC%\torch\csrc\api\include" ^
    -L"%T_LIB%" ^
    -L"%PY_LIB_DIR%" ^
    -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lc10 -lcuda ^
    -DNOMINMAX ^
    -D__CS_CUDA_COMPILATION__ ^
    -Xcompiler /MD ^
    -Xcompiler /wd4348 ^
    -Xcompiler /wd4018 ^
    -Xcompiler /wd4267 ^
    -Xcompiler /wd4244 ^
    -Xcompiler /wd4251 ^
    -Xcompiler /wd4275 ^
    -Xcompiler /wd4819

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Build complete.
) else (
    echo [FAILED] Compilation failed.
)
pause