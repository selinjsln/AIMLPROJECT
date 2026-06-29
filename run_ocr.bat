@echo off
call venv\Scripts\activate

python ocr_engine.py %1

pause