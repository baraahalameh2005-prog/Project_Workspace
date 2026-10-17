# Starts the avatar backend (port 8002), the frontend (port 5173) and the
# clothing server (port 8010, the try-on page) in their own windows, waits
# until the backend has loaded Anny, then opens the page.
# Close the windows to stop them.

$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Missing $python - create the avatar environment first (see docs/API.md)." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

function Test-Url($url) {
    try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

$free = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
if ($free -lt 2500) {
    Write-Host "Only $free MB of memory is free. The avatar backend needs about 2 GB;" -ForegroundColor Yellow
    Write-Host "close Chrome tabs and other large programs if it stops or becomes slow." -ForegroundColor Yellow
}

if (Test-Url "http://127.0.0.1:8002/health") {
    Write-Host "Backend already running on port 8002."
} else {
    Write-Host "Starting the backend (loading Anny takes up to a minute)..."
    $env:TMP = Join-Path $root "var\tmp"; $env:TEMP = $env:TMP
    New-Item -ItemType Directory -Force $env:TMP | Out-Null
    Start-Process -FilePath $python -ArgumentList "-B", "api.py" -WorkingDirectory (Join-Path $root "backend")
}

if (Test-Url "http://127.0.0.1:5173/") {
    Write-Host "Frontend already running on port 5173."
} else {
    Write-Host "Starting the frontend..."
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "npm run dev" -WorkingDirectory (Join-Path $root "frontend")
}

$clothingPython = Join-Path $root "clothing\.venv\Scripts\python.exe"
if (Test-Url "http://127.0.0.1:8010/viewer/") {
    Write-Host "Clothing server already running on port 8010."
} elseif (Test-Path $clothingPython) {
    Write-Host "Starting the clothing server..."
    Start-Process -FilePath $clothingPython -ArgumentList "-B", "backend\server.py" -WorkingDirectory (Join-Path $root "clothing")
} else {
    Write-Host "Missing $clothingPython - the try-on page will not work (see clothing/README.md)." -ForegroundColor Yellow
}

# Baraah's Dress3D prototype (a dress alone in 3D), if its repo is next to this one.
$dress3d = Join-Path (Split-Path $root -Parent) "clothes"
$dress3dPython = Join-Path $dress3d ".venv\Scripts\python.exe"
if (Test-Url "http://127.0.0.1:8765/") {
    Write-Host "Dress3D already running on port 8765."
} elseif (Test-Path $dress3dPython) {
    Write-Host "Starting Dress3D..."
    Start-Process -FilePath $dress3dPython -ArgumentList "prototype\server.py" -WorkingDirectory $dress3d
} else {
    Write-Host "Dress3D not found at $dress3d - the 3D dress button will not work." -ForegroundColor Yellow
}

Write-Host "Waiting for the backend to be ready..."
$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8002/health" -TimeoutSec 2
        if ($health.model_loaded) { break }
    } catch { }
    Start-Sleep -Seconds 2
}
if ((Get-Date) -ge $deadline) {
    Write-Host "The backend did not become ready in 5 minutes; check its window for errors." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Ready. Opening http://127.0.0.1:5173/" -ForegroundColor Green
Start-Process "http://127.0.0.1:5173/"
