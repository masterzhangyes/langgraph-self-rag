@echo off
chcp 65001 >nul
title 智能问答系统 - 一键启动
setlocal
cd /d "%~dp0"

echo ========================================
echo    智能问答系统 v2.1 - 一键启动
echo ========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+ 并勾选 "Add to PATH"
    echo.
    pause
    exit /b 1
)

echo 启动中，请稍候（服务首次启动/装依赖时会较慢）...
echo 日志文件: logs\uvicorn.out.log 与 logs\vite.out.log
echo.
echo 提示: 关闭本窗口不会停止服务
echo       停止服务请在项目目录执行:  python start_all.py --stop
echo.
python start_all.py

echo.
echo ----------------------------------------
pause
