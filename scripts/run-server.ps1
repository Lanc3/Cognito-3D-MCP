param([ValidateSet('SF3D', 'SPAR3D')][string]$Backend = 'SF3D', [string]$SettingsPath)
$parameters = @{ Backend = $Backend }
if ($SettingsPath) { $parameters.SettingsPath = $SettingsPath }
& (Join-Path $PSScriptRoot 'run.ps1') @parameters
