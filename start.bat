@echo off
title YaLlegue - Checador Facial
echo.
echo  =============================================
echo   YaLlegue - Checador de Asistencia Facial
echo   http://localhost:5252
echo   http://localhost:5252/pantalla
echo  =============================================
echo.

cd /d "%~dp0"

REM Usa el venv de tuplatillofue si existe, si no busca venv local
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "..\tuplatillofue\.venv\Scripts\activate.bat" (
    call ..\tuplatillofue\.venv\Scripts\activate.bat
    echo [OK] Usando venv de tuplatillofue
) else if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
) else (
    echo [WARN] No se encontro venv, usando Python del sistema
)

echo [INFO] Instalando dependencias si faltan...
pip install flask insightface onnxruntime opencv-python numpy werkzeug -q

echo.
echo [INFO] Iniciando YaLlegue en puerto 5252...
echo [INFO] Pantalla TV:  http://localhost:5252/pantalla
echo [INFO] Admin panel:  http://localhost:5252/admin
echo.

python server.py

pause
