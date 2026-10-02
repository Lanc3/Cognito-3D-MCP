param(
    [string]$Destination = (Join-Path (Join-Path $env:LOCALAPPDATA 'Cognito-3D-mcp') 'runtime\autoremesher')
)
$ErrorActionPreference = 'Stop'
$release = '1.2.0'
$commit = 'd9ef96bd72f0b134dd7e51acf5904f32a5679704'
$expectedHash = 'f6184622cef84f0bcf032a0474df44e2076f4c10c159f1af0cb7230259af7476'
$archiveUrl = 'https://github.com/huxingyi/autoremesher/releases/download/1.2.0/autoremesher-1.2.0-win32-x86_64.zip'
$target = [System.IO.Path]::GetFullPath($Destination)
$exe = Join-Path $target 'autoremesher.exe'
if (Test-Path -LiteralPath $exe) {
    $manifestPath = Join-Path $target 'install-manifest.json'
    if (Test-Path -LiteralPath $manifestPath) {
        $installed = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $exeHash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($installed.archive_sha256 -eq $expectedHash -and $installed.exe_sha256 -eq $exeHash) {
            Write-Output "Verified existing AutoRemesher $release at $exe"
            return
        }
    }
    throw "An unverified AutoRemesher already exists at $exe. Choose an empty destination."
}
if ((Test-Path -LiteralPath $target) -and @(Get-ChildItem -LiteralPath $target -Force).Count -gt 0) {
    throw "Destination is not empty: $target. Choose an empty destination."
}
$parent = Split-Path -Parent $target
$downloadDir = Join-Path $parent 'autoremesher-downloads'
New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
$archive = Join-Path $downloadDir 'autoremesher-1.2.0-win32-x86_64.zip'
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archive
}
$actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    throw "AutoRemesher archive checksum mismatch. Expected $expectedHash, received $actualHash."
}
New-Item -ItemType Directory -Path $target -Force | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $target
if (-not (Test-Path -LiteralPath $exe)) { throw 'Release did not contain autoremesher.exe.' }
foreach ($notice in @('LICENSE', 'ACKNOWLEDGEMENTS.html')) {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Lanc3/autoremesher/$commit/$notice" `
        -OutFile (Join-Path $target $notice)
}
$manifest = [ordered]@{
    release = $release
    source_repository = 'https://github.com/Lanc3/autoremesher'
    source_commit = $commit
    binary_repository = 'https://github.com/huxingyi/autoremesher'
    provenance = 'GitHub comparison verified upstream tag 1.2.0 identical to Lanc3 master commit.'
    archive_url = $archiveUrl
    archive_sha256 = $expectedHash
    exe_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    installed_utc = [DateTime]::UtcNow.ToString('o')
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $target 'install-manifest.json') -Encoding utf8
Write-Output "Installed and hash-verified AutoRemesher $release at $exe"
Write-Output "CODEX_AUTOREMESHER_EXE=$exe"
Write-Output 'Remeshing is CPU-only. The MCP wrapper enforces CPU affinity, memory caps and low priority.'
