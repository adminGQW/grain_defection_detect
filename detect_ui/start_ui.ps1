$ErrorActionPreference = 'Stop'
$python = $env:GRAIN_UI_PYTHON
$url = 'http://127.0.0.1:8765'
$expectedCheckpoint = 'best_model_original.pth'
$server = $null

# 路径从脚本位置和环境变量读取，避免把中文路径拼进多层引号命令。
function Get-ModelInfo {
    try {
        Invoke-RestMethod -Uri "$url/api/info" -TimeoutSec 2
    } catch {
        return $null
    }
}

try {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Python environment not found: $python. Edit GRAIN_UI_PYTHON in the BAT file."
    }
    $info = Get-ModelInfo
    if ($info -and $info.checkpoint -ne $expectedCheckpoint) {
        throw 'An older/different UI is using port 8765. Stop its server window with Ctrl+C, then launch again.'
    }
    if (-not $info) {
        # 在当前启动窗口显示服务输出，模型加载失败时可直接看到原因。
        $server = Start-Process -FilePath $python -ArgumentList '-u', 'app.py' -WorkingDirectory $PSScriptRoot -NoNewWindow -PassThru
        $deadline = (Get-Date).AddSeconds(90)
        while (-not $info -and -not $server.HasExited -and (Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 500
            $info = Get-ModelInfo
        }
        if (-not $info) {
            throw 'The UI server did not become ready. See the Python error above.'
        }
    }
    Start-Process $url
    if ($server) {
        Write-Host 'Keep this window open. Press Ctrl+C to stop the UI server.'
        $server.WaitForExit()
        if ($server.ExitCode -ne 0) { throw "UI server exited: $($server.ExitCode)" }
    }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    # 只清理本次启动的进程，不停止其他窗口已有的服务。
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -ErrorAction SilentlyContinue }
}
