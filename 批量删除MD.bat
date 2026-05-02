@echo off
:: 设置字符集为 UTF-8，防止控制台中文乱码
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 正在扫描并删除 .md 文件...
echo [已启用排除规则: 跳过 "prompts" 文件夹 和 "default.md" 文件]
echo ---------------------------------------------------------

:: 核心逻辑说明：
:: 1. dir /s /b *.md -> 找到所有的 .md 文件
:: 2. findstr /v /i -> 反向过滤（/v）并忽略大小写（/i）
:: 3. /c:"\prompts\" -> 排除路径中包含 \prompts\ 的文件（即跳过该文件夹）
:: 4. /c:"\default.md" -> 排除名为 default.md 的文件

for /f "delims=" %%F in ('dir /s /b *.md 2^>nul ^| findstr /v /i /c:"\prompts\" /c:"\default.md"') do (
    echo [删除] "%%F"
    del /q /f "%%F"
)

echo ---------------------------------------------------------
echo 删除操作已完成！
pause