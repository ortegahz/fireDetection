@echo off
chcp 65001
setlocal

set CURRENT_PATH=%cd%
echo ????: %CURRENT_PATH%

set PYTHONPATH=%CURRENT_PATH%;%PYTHONPATH%
echo PYTHONPATH: %PYTHONPATH%

set VIDEO_FOLDER=D:\vids
set PYTHON_SCRIPT=demos\demo_detect_mp.py
set SAVE_ROOT=D:\fire_test_results

echo VIDEO_FOLDER: %VIDEO_FOLDER%
echo PYTHON_SCRIPT: %PYTHON_SCRIPT%
echo SAVE_ROOT: %SAVE_ROOT%

if exist "%SAVE_ROOT%" (
    rd /s /q "%SAVE_ROOT%"
)

mkdir "%SAVE_ROOT%"

for /r "%VIDEO_FOLDER%" %%f in (*.mp4 *.avi *.mov) do (
    echo Processing video: %%f
    echo python "%PYTHON_SCRIPT%" --path_video "%%f" --save_root "%SAVE_ROOT%"

    start /wait python "%PYTHON_SCRIPT%" --path_video "%%f" --save_root "%SAVE_ROOT%"

    echo Waiting for 1 seconds...
    timeout /t 1 /nobreak
)

endlocal
