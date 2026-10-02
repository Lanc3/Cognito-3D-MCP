#Requires -Version 5.1
param(
    [ValidateSet('Hunyuan', 'SF3D', 'SPAR3D', 'TRELLIS')][string]$Backend,
    [string]$SettingsPath = (Join-Path (Split-Path -Parent $PSScriptRoot) '.cognito-install.json')
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
if (-not (Test-Path -LiteralPath $SettingsPath)) {
    $SettingsPath = Join-Path (Get-CognitoDataRoot) 'installation.json'
}
if (-not (Test-Path -LiteralPath $SettingsPath)) { throw 'Installation settings are missing. Run Install.cmd first.' }
$settings = Get-Content -LiteralPath $SettingsPath -Raw | ConvertFrom-Json
$Backend = Resolve-CognitoBackend -Requested $Backend -Installed @($settings.backends)
$python = switch ($Backend) {
    'Hunyuan' { Join-Path $settings.runtime_root 'hunyuan3d-2mv\venv\Scripts\python.exe' }
    'SF3D' { Join-Path $settings.runtime_root 'sf3d\venv\Scripts\python.exe' }
    'SPAR3D' { Join-Path $settings.runtime_root 'spar3d\venv\Scripts\python.exe' }
    'TRELLIS' { Join-Path $settings.runtime_root 'trellis-venv\Scripts\python.exe' }
}
Invoke-Checked -FilePath $python -Arguments @((Join-Path $PSScriptRoot 'launch-server.py'), '--settings', $SettingsPath, '--backend', $Backend)
