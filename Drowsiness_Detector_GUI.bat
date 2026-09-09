@echo off
cd /d "%~dp0"
title Driver Drowsiness Detection - GUI
echo Starting Driver Drowsiness Detection GUI ...
python -u src\gui.py
pause
