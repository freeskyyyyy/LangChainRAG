@echo off
title RAG 知识库问答系统 - 后端服务
cd /d "%~dp0backend"

echo.
echo ============================================================
echo    企业级 RAG 知识库问答系统
echo ============================================================
echo.
echo    启动后，浏览器打开下面任意一个地址：
echo.
echo      问答界面    http://127.0.0.1:8000/demo
echo      接口文档    http://127.0.0.1:8000/docs
echo.
echo    要停止服务，直接关闭这个黑色窗口即可。
echo ============================================================
echo.

"..\.venv\Scripts\python.exe" run.py

echo.
echo    服务已停止。
echo.
pause
