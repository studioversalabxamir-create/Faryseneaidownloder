@echo off
echo Installing Spleeter for vocal separation...

REM Activate virtual environment if it exists
if exist "spleeter-env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call spleeter-env\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv spleeter-env
    call spleeter-env\Scripts\activate.bat
)

REM Install required packages
echo Installing TensorFlow...
pip install tensorflow==2.10.0

echo Installing Spleeter...
pip install spleeter

echo Installing additional dependencies...
pip install pydub
pip install librosa
pip install numpy

echo Installation complete!
echo.
echo To test the installation, run:
echo python -c "from spleeter.separator import Separator; print('Spleeter installed successfully!')"

pause