@echo off
:: 1. Point to your VS Build Tools / Community installation
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "VCVARS=%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS%" (
    echo [ERROR] Could not find vcvars64.bat at %VCVARS%
    exit /b 1
)

:: 2. Initialize the x64 Developer Environment
echo Initializing MSVC Environment...
call "%VCVARS%"

:: 3. Run the Python build
echo Starting Build...
python setup.py build_ext --inplace

echo Done!
pause