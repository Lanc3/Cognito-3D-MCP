param([string]$SettingsPath)
$parameters = @{ Backend = 'Hunyuan' }
if ($SettingsPath) { $parameters.SettingsPath = $SettingsPath }
& (Join-Path $PSScriptRoot 'run.ps1') @parameters
