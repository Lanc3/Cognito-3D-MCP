param([string]$SettingsPath)
$parameters = @{ Backend = 'TRELLIS' }
if ($SettingsPath) { $parameters.SettingsPath = $SettingsPath }
& (Join-Path $PSScriptRoot 'run.ps1') @parameters
