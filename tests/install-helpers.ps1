$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts\common.ps1')
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('cognito-install-test-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $testRoot | Out-Null
$compilerRoot = Join-Path $testRoot "Compiler's path & spaces"
New-Item -ItemType Directory -Force -Path (Join-Path $compilerRoot 'Common7\Tools') | Out-Null
@"
@echo off
set COGNITO_ENV_TEST=expected
"@ | Set-Content -LiteralPath (Join-Path $compilerRoot 'Common7\Tools\VsDevCmd.bat') -Encoding ascii
function Get-VisualStudio2022Path { return $compilerRoot }
Import-VisualStudio2022Environment
if ($env:COGNITO_ENV_TEST -ne 'expected') { throw 'Compiler environment import did not survive special path characters.' }
$zipSource = Join-Path $testRoot 'archive-source'
New-Item -ItemType Directory -Force -Path $zipSource | Out-Null
[IO.File]::WriteAllText((Join-Path $zipSource 'tool.exe'), 'verified executable fixture')
[IO.File]::WriteAllText((Join-Path $zipSource 'dependency.dll'), 'verified dependency fixture')
$zip = Join-Path $testRoot 'fixture.zip'
Compress-Archive -Path (Join-Path $zipSource '*') -DestinationPath $zip -Force
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
Get-VerifiedDownload -Url 'https://invalid.example/not-downloaded' -Destination $zip -Sha256 $hash
$destination = Join-Path $testRoot 'expanded'
Expand-VersionedArchive -Archive $zip -Destination $destination -AnchorFile 'tool.exe'
Expand-VersionedArchive -Archive $zip -Destination $destination -AnchorFile 'tool.exe'
[IO.File]::WriteAllText((Join-Path $destination 'dependency.dll'), 'corrupted fixture')
$caught = $false
try { Expand-VersionedArchive -Archive $zip -Destination $destination -AnchorFile 'tool.exe' } catch { $caught = $true }
if (-not $caught) { throw 'Corrupt installed DLL was not detected.' }
$exitCaught = $false
try { Invoke-Checked -FilePath cmd.exe -Arguments @('/d', '/c', 'exit 7') } catch { $exitCaught = $true }
if (-not $exitCaught) { throw 'Non-zero native command was not detected.' }
Invoke-Checked -FilePath cmd.exe -Arguments @('/d', '/c', 'echo Expected stderr fixture 1>&2 & exit 0')
if ($ErrorActionPreference -ne 'Stop') { throw 'Native command changed the caller error policy.' }
if ((Resolve-CognitoBackend -Installed @('SF3D', 'Hunyuan')) -ne 'Hunyuan') {
    throw 'A multiview install did not prefer Hunyuan.'
}
if ((Resolve-CognitoBackend -Installed @('SF3D', 'SPAR3D')) -ne 'SF3D') {
    throw 'A legacy install did not select the first installed backend.'
}
if ((Resolve-CognitoBackend -Installed @('SPAR3D')) -ne 'SPAR3D') {
    throw 'A SPAR3D-only install did not select SPAR3D.'
}
$backendCaught = $false
try { Resolve-CognitoBackend -Requested 'Hunyuan' -Installed @('SPAR3D') | Out-Null } catch {
    $backendCaught = $_.Exception.Message.Contains('Hunyuan is not installed')
}
if (-not $backendCaught) { throw 'An explicitly uninstalled backend was not rejected clearly.' }
Write-Host 'PASS: compiler environment quoting, cached archive checksum, repeat extraction verification, corrupted DLL rejection, native exit-code checking.'
