param(
    [string]$Python = "python",
    [string]$InnoCompiler = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VendorBin = Join-Path $PSScriptRoot "vendor\bin"
New-Item -ItemType Directory -Path $VendorBin -Force | Out-Null

foreach ($ToolName in @("ffmpeg.exe", "ffprobe.exe")) {
    $Tool = Get-Command $ToolName -ErrorAction SilentlyContinue
    if ($Tool) {
        Copy-Item -LiteralPath $Tool.Source -Destination (Join-Path $VendorBin $ToolName) -Force
    }
}

& $Python -m pip install --upgrade pyinstaller
& $Python -m PyInstaller --noconfirm --clean --distpath (Join-Path $RepoRoot "dist") --workpath (Join-Path $RepoRoot "build") (Join-Path $PSScriptRoot "Editto.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

if (-not (Test-Path -LiteralPath $InnoCompiler)) {
    throw "Inno Setup 6 not found. Install it or pass -InnoCompiler with ISCC.exe path."
}

& $InnoCompiler (Join-Path $PSScriptRoot "Editto.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed." }

Write-Host "Installer created under $(Join-Path $RepoRoot 'installer_output')"
