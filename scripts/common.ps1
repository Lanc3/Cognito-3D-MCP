# Shared Windows installer helpers. Dot-source this file; it performs no installation.
Set-StrictMode -Version 3

function Get-CognitoDataRoot {
    if ($env:LOCALAPPDATA) { return Join-Path $env:LOCALAPPDATA 'Cognito-3D-mcp' }
    return Join-Path ([Environment]::GetFolderPath('UserProfile')) 'Cognito-3D-mcp'
}

function Resolve-CognitoBackend {
    param([string]$Requested, [string[]]$Installed)
    if (-not $Installed -or $Installed.Count -eq 0) {
        throw 'No installed 3D backend is recorded. Run Install.cmd first.'
    }
    if (-not $Requested) {
        $Requested = if ('Hunyuan' -in $Installed) { 'Hunyuan' } else { $Installed[0] }
    }
    if ($Requested -notin @('Hunyuan', 'SF3D', 'SPAR3D', 'TRELLIS')) {
        throw 'The saved backend setting is invalid. Rerun Install.cmd.'
    }
    if ($Requested -notin $Installed) {
        throw "$Requested is not installed. Installed backends: $($Installed -join ', '). Run install.ps1 -Backend $Requested to add it."
    }
    return $Requested
}

function Assert-HunyuanTerms {
    param([switch]$Accepted)
    if ($Accepted) { return }
    Write-Host 'Hunyuan3D has separate upstream terms; the integration MIT license does not cover the model.'
    Write-Host 'The pinned Hunyuan license excludes the European Union, United Kingdom, and South Korea.'
    Write-Host 'Read: https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/f8db63096c8282cb27354314d896feba5ba6ff8a/LICENSE'
    $answer = Read-Host 'Confirm you have reviewed the upstream terms and are eligible to use Hunyuan (type YES)'
    if ($answer -cne 'YES') { throw 'Hunyuan installation cancelled before downloading its runtime or models.' }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [int[]]$SuccessCodes = @(0)
    )
    if (-not (Test-Path -LiteralPath $FilePath)) {
        Get-Command $FilePath -ErrorAction Stop | Out-Null
    }
    $previousErrorAction = $ErrorActionPreference
    try {
        # Windows PowerShell turns harmless native stderr (such as pip warnings)
        # into ErrorRecords. Keep stderr visible and judge success by the exit code.
        $ErrorActionPreference = 'Continue'
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($exitCode -notin $SuccessCodes) {
        # Avoid printing arguments, which can contain credentials.
        throw "Command failed with exit code ${exitCode}: $FilePath"
    }
}

function Resolve-InstallerPython {
    param([string]$Python = 'py -3.11')
    # Preserve the old documented launcher value without evaluating shell text.
    if ($Python -eq 'py -3.11') {
        $executable = & py.exe -3.11 -c 'import sys; print(sys.executable)'
        if ($LASTEXITCODE -ne 0 -or -not $executable) { throw 'Python 3.11 is required.' }
        $Python = ($executable | Select-Object -Last 1).Trim()
    }
    $command = Get-Command $Python -ErrorAction Stop
    $resolved = $command.Source
    Invoke-Checked -FilePath $resolved -Arguments @('-c', 'import sys; assert sys.version_info[:2] == (3, 11), "The GPU backends require Python 3.11"')
    return $resolved
}

function Get-VisualStudio2022Path {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { return $null }
    $result = & $vswhere -latest -version '[17.0,18.0)' -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $result) { return $null }
    return ($result | Select-Object -Last 1).Trim()
}

function Import-VisualStudio2022Environment {
    $vsPath = Get-VisualStudio2022Path
    if (-not $vsPath) { throw 'Install Visual Studio 2022 Build Tools with the C++ workload (or run Install.cmd).' }
    $vsDevCmd = Join-Path $vsPath 'Common7\Tools\VsDevCmd.bat'
    # Start the command with CALL, avoiding cmd's special stripping of a leading
    # quoted executable path (which differs between Windows PowerShell and pwsh).
    $lines = & cmd.exe /d /c "call `"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0) { throw 'Could not activate the Visual Studio 2022 build environment.' }
    foreach ($line in $lines) {
        $separator = $line.IndexOf('=')
        if ($separator -gt 0) {
            [Environment]::SetEnvironmentVariable($line.Substring(0, $separator), $line.Substring($separator + 1), 'Process')
        }
    }
    $env:DISTUTILS_USE_SDK = '1'
    $env:MSSdk = '1'
}

function Assert-CudaRuntime {
    param([Parameter(Mandatory = $true)][string]$PythonExe)
    Invoke-Checked -FilePath $PythonExe -Arguments @('-c', 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable. Check the NVIDIA driver and PyTorch wheel."; print("GPU:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda)')
    # Compile for the actual installed GPU(s), unless the user explicitly selected architectures.
    if (-not $env:TORCH_CUDA_ARCH_LIST) {
        $architectures = & $PythonExe -c 'import torch; print(";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))'
        if ($LASTEXITCODE -ne 0 -or -not $architectures) { throw 'Could not determine CUDA GPU architectures.' }
        $env:TORCH_CUDA_ARCH_LIST = ($architectures | Select-Object -Last 1).Trim()
    }
    if (-not $env:CUDA_PATH -or -not (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
        throw 'CUDA Toolkit is required to compile the GPU extensions. Run Install.cmd or set CUDA_PATH.'
    }
    $torchCuda = (& $PythonExe -c 'import torch; print(torch.version.cuda)').Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not query the PyTorch CUDA version.' }
    $nvccVersion = & (Join-Path $env:CUDA_PATH 'bin\nvcc.exe') --version
    if ($LASTEXITCODE -ne 0 -or ($nvccVersion -join ' ') -notmatch "release $([regex]::Escape($torchCuda))[, ]") {
        throw "CUDA Toolkit must match the PyTorch wheel ($torchCuda). CUDA_PATH currently points to $env:CUDA_PATH."
    }
}

function Assert-FreeSpace {
    param([string]$Path, [int]$MinimumGB)
    $driveRoot = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path))
    $drive = [IO.DriveInfo]::new($driveRoot)
    if ($drive.AvailableFreeSpace -lt ($MinimumGB * 1GB)) {
        throw "At least ${MinimumGB}GB free is required on $driveRoot."
    }
}

function Get-VerifiedDownload {
    param([string]$Url, [string]$Destination, [string]$Sha256)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    if (-not (Test-Path -LiteralPath $Destination)) {
        $partial = "$Destination.partial"
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial
        $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
        if ($actual -ne $Sha256) { throw "SHA-256 mismatch for downloaded archive: $Destination" }
        Move-Item -LiteralPath $partial -Destination $Destination
    }
    if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -ne $Sha256) {
        throw "SHA-256 mismatch for cached archive: $Destination"
    }
}

function Expand-VersionedArchive {
    param([string]$Archive, [string]$Destination, [string]$AnchorFile)
    # Use a new staging directory so recovery never recursively removes a user path.
    $staging = Join-Path (Split-Path -Parent $Archive) ('extract-' + [Guid]::NewGuid().ToString('N'))
    Expand-Archive -LiteralPath $Archive -DestinationPath $staging
    $anchor = Get-ChildItem -LiteralPath $staging -Recurse -File -Filter $AnchorFile | Select-Object -First 1
    if (-not $anchor) { throw "$AnchorFile was not found inside $Archive." }
    if (Test-Path -LiteralPath (Join-Path $Destination $AnchorFile)) {
        foreach ($file in (Get-ChildItem -LiteralPath $anchor.Directory.FullName -Recurse -File)) {
            $relative = $file.FullName.Substring($anchor.Directory.FullName.Length).TrimStart('\')
            $installed = Join-Path $Destination $relative
            if (-not (Test-Path -LiteralPath $installed) -or
                (Get-FileHash -LiteralPath $installed -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash) {
                throw "Installed runtime differs from the verified archive: $installed. Choose a new destination or repair this runtime explicitly."
            }
        }
        return
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $anchor.Directory.FullName -Force | Copy-Item -Destination $Destination -Recurse -Force
}

function Get-BlenderExecutable {
    if ($env:CODEX_BLENDER_EXE -and (Test-Path -LiteralPath $env:CODEX_BLENDER_EXE)) { return $env:CODEX_BLENDER_EXE }
    $command = Get-Command blender.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $blenderRoot = Join-Path $env:ProgramFiles 'Blender Foundation'
    if (Test-Path -LiteralPath $blenderRoot) {
        $found = Get-ChildItem -LiteralPath $blenderRoot -Filter blender.exe -Recurse -File | Sort-Object FullName -Descending | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return $null
}
