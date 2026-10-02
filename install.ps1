#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Hunyuan', 'SF3D', 'SPAR3D', 'TRELLIS', 'All')][string]$Backend = 'Hunyuan',
    [string]$DataRoot = (Join-Path $env:LOCALAPPDATA 'Cognito-3D-mcp'),
    [string]$RuntimeRoot,
    [string]$ModelRoot,
    [string]$OutputRoot,
    [string]$Python,
    [string]$CodexConfig,
    [string]$SkillsRoot,
    [string]$HuggingFaceTokenPath,
    [switch]$SkipPrerequisiteInstall,
    [switch]$NoRegisterCodex,
    [switch]$NoInstallSkills,
    [switch]$AcceptHunyuanLicense,
    [switch]$Interactive,
    [switch]$InstallCuMesh,
    [ValidateSet(2048,2560,3072,3584,4096)][int]$CuMeshBuildMemoryMiB = 2048,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts\common.ps1')
if ($Interactive -and -not $DryRun -and -not $PSBoundParameters.ContainsKey('Backend')) {
    Write-Host 'Cognito-3D-mcp: choose a 3D engine to install.'
    Write-Host '1. Hunyuan (default multiview engine; licence excludes EU, UK and South Korea)'
    Write-Host '2. SF3D (single-image; Hugging Face model access required)'
    Write-Host '3. SPAR3D (single-image; Hugging Face model access required)'
    Write-Host '4. TRELLIS (bidirectional engine; separate model licences apply)'
    Write-Host '5. All engines (all upstream licence requirements apply)'
    do {
        $selection = Read-Host 'Enter 1, 2, 3, 4, or 5'
    } until ($selection -in @('1', '2', '3', '4', '5'))
    $Backend = @('Hunyuan', 'SF3D', 'SPAR3D', 'TRELLIS', 'All')[[int]$selection - 1]
}
$DataRoot = [IO.Path]::GetFullPath($DataRoot)
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $DataRoot 'runtime' }
if (-not $ModelRoot) { $ModelRoot = Join-Path $DataRoot 'models' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $DataRoot 'outputs' }
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$ModelRoot = [IO.Path]::GetFullPath($ModelRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if (-not $HuggingFaceTokenPath) {
    if ($env:HF_TOKEN_PATH) { $HuggingFaceTokenPath = $env:HF_TOKEN_PATH }
    else {
        $originalHfHome = if ($env:HF_HOME) { $env:HF_HOME }
            elseif ($env:XDG_CACHE_HOME) { Join-Path $env:XDG_CACHE_HOME 'huggingface' }
            else { Join-Path ([Environment]::GetFolderPath('UserProfile')) '.cache\huggingface' }
        $HuggingFaceTokenPath = Join-Path $originalHfHome 'token'
    }
}
$HuggingFaceTokenPath = [IO.Path]::GetFullPath($HuggingFaceTokenPath)
$codexDirectory = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path ([Environment]::GetFolderPath('UserProfile')) '.codex' }
if (-not $CodexConfig) { $CodexConfig = Join-Path $codexDirectory 'config.toml' }
# Codex's current user-skill discovery path is ~/.agents/skills.
if (-not $SkillsRoot) { $SkillsRoot = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.agents\skills' }
[string[]]$backends = if ($Backend -eq 'All') { @('Hunyuan', 'SF3D', 'SPAR3D', 'TRELLIS') } else { @($Backend) }
$plan = [ordered]@{
    brand = 'Cognito-3D-mcp'
    backends = $backends
    runtime_root = $RuntimeRoot
    model_root = $ModelRoot
    output_root = $OutputRoot
    codex_config = $CodexConfig
    skills_root = $SkillsRoot
    huggingface_token_path = $HuggingFaceTokenPath
    register_codex = -not $NoRegisterCodex
    install_skills = -not $NoInstallSkills
    prerequisite_installs = -not $SkipPrerequisiteInstall
    python = '3.11'
    cuda_toolkit = '13.0'
    pytorch = '2.9.1+cu130'
    requires_existing_nvidia_driver = $true
    hunyuan_license_acknowledgement = ('Hunyuan' -in $backends)
    optional_cumesh_build = [bool]$InstallCuMesh
    cumesh_build_memory_mib = $CuMeshBuildMemoryMiB
}
if ($DryRun) {
    $plan | ConvertTo-Json -Depth 5
    return
}
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'The automated installer requires 64-bit Windows. Other platforms can run the MCP wrapper, but native GPU setup is manual.'
}
if ($InstallCuMesh -and 'Hunyuan' -notin $backends) { throw '-InstallCuMesh requires -Backend Hunyuan or All.' }
if (-not $InstallCuMesh -and $CuMeshBuildMemoryMiB -ne 2048) { throw '-CuMeshBuildMemoryMiB only applies with -InstallCuMesh.' }
if ('Hunyuan' -in $backends) { Assert-HunyuanTerms -Accepted:$AcceptHunyuanLicense }
$env:HF_TOKEN_PATH = $HuggingFaceTokenPath

function Refresh-InstallerEnvironment {
    # Keep current task-specific paths while adding newly installed software.
    $env:PATH = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:PATH
    $gitBin = Join-Path $env:ProgramFiles 'Git\cmd'
    if (Test-Path -LiteralPath $gitBin) { $env:PATH = "$gitBin;$env:PATH" }
    $cudaPath = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v13.0'
    if (Test-Path -LiteralPath (Join-Path $cudaPath 'bin\nvcc.exe')) {
        $env:CUDA_PATH = $cudaPath
        $env:PATH = (Join-Path $cudaPath 'bin') + ';' + $env:PATH
    }
}

function Install-WingetPrerequisite {
    param([string]$Id, [string[]]$ExtraArguments = @())
    if ($SkipPrerequisiteInstall) { throw "Missing prerequisite: $Id. Install it, then run the installer again." }
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw 'Windows Package Manager (winget) is missing. Install Microsoft App Installer from Microsoft Store, then rerun Install.cmd.'
    }
    Write-Host "Installing prerequisite: $Id (Windows may show an administrator approval dialog)."
    Invoke-Checked -FilePath $winget.Source -Arguments (@('install', '--exact', '--id', $Id, '--source', 'winget', '--accept-source-agreements', '--accept-package-agreements', '--disable-interactivity') + $ExtraArguments)
    Refresh-InstallerEnvironment
}

function Find-Python311 {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Join-Path $env:ProgramFiles 'Python311\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            try {
                & $candidate -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>$null
                if ($LASTEXITCODE -eq 0) { return $candidate }
            } catch {
                # Windows PowerShell treats native stderr as an error with Stop.
                # A broken interpreter must not prevent bootstrapping Python.
            }
        }
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $found = & $launcher.Source -3.11 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { return ($found | Select-Object -Last 1).Trim() }
        } catch {
            # py.exe may exist before Python 3.11 is installed.
        }
    }
    return $null
}

Refresh-InstallerEnvironment
$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    throw 'An NVIDIA CUDA GPU and NVIDIA driver are required. Install the current driver from https://www.nvidia.com/Download/index.aspx and rerun Install.cmd.'
}
$gpuInformation = & $nvidiaSmi.Source --query-gpu=name,driver_version --format=csv,noheader
if ($LASTEXITCODE -ne 0 -or -not $gpuInformation) { throw 'No working NVIDIA GPU/driver was detected.' }
Write-Host "Detected GPU: $($gpuInformation -join '; ')"
if ('TRELLIS' -notin $backends -or $backends.Count -gt 1) {
    foreach ($gpu in $gpuInformation) {
        $driver = ($gpu -split ',')[-1].Trim()
        if ([version]$driver -lt [version]'580.0') { throw 'The pinned CUDA 13.0 stack needs NVIDIA driver 580 or newer. Update the driver and rerun Install.cmd.' }
    }
}
Assert-FreeSpace -Path $RuntimeRoot -MinimumGB 45
Assert-FreeSpace -Path $ModelRoot -MinimumGB 45
if (-not $Python) {
    $Python = Find-Python311
    if (-not $Python) {
        Install-WingetPrerequisite -Id 'Python.Python.3.11'
        $Python = Find-Python311
    }
}
if (-not $Python) { throw 'Python 3.11 installation could not be located. Rerun with -Python C:\path\to\python.exe.' }
$Python = Resolve-InstallerPython -Python $Python
if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) { Install-WingetPrerequisite -Id 'Git.Git' }
if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) { throw 'Git is still unavailable after installation. Restart Windows and rerun Install.cmd.' }

$needsCompiler = @($backends | Where-Object { $_ -ne 'TRELLIS' }).Count -gt 0
if ($needsCompiler) {
    if (-not (Get-VisualStudio2022Path)) {
        Install-WingetPrerequisite -Id 'Microsoft.VisualStudio.2022.BuildTools' -ExtraArguments @('--force', '--override', '--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended')
    }
    if (-not (Get-VisualStudio2022Path)) { throw 'Visual Studio 2022 C++ Build Tools are unavailable. A restart may be required; rerun Install.cmd afterwards.' }
    $cudaPath = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v13.0'
    if (-not (Test-Path -LiteralPath (Join-Path $cudaPath 'bin\nvcc.exe'))) {
        Install-WingetPrerequisite -Id 'Nvidia.CUDA' -ExtraArguments @('--version', '13.0')
    }
    Refresh-InstallerEnvironment
    if (-not $env:CUDA_PATH -or -not (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
        throw 'CUDA Toolkit 13.0 is unavailable. A restart may be required; rerun Install.cmd afterwards.'
    }
}
$blender = Get-BlenderExecutable
if (-not $blender) {
    Install-WingetPrerequisite -Id 'BlenderFoundation.Blender'
    $blender = Get-BlenderExecutable
}
if (-not $blender) { throw 'Blender could not be found after installation. Set CODEX_BLENDER_EXE to blender.exe and rerun.' }
New-Item -ItemType Directory -Force -Path $DataRoot, $RuntimeRoot, $ModelRoot, $OutputRoot, (Join-Path $DataRoot 'inputs') | Out-Null
$env:CODEX_BLENDER_EXE = $blender
$validatorRoot = Join-Path $RuntimeRoot 'gltf-validator'
$validatorArchive = Join-Path $RuntimeRoot 'downloads\gltf_validator-2.0.0-dev.3.10-win64.zip'
Get-VerifiedDownload -Url 'https://github.com/KhronosGroup/glTF-Validator/releases/download/2.0.0-dev.3.10/gltf_validator-2.0.0-dev.3.10-win64.zip' -Destination $validatorArchive -Sha256 'c5068f51205deedc28acc3529ee7e11ee60e853454f673093398eba80142202c'
Expand-VersionedArchive -Archive $validatorArchive -Destination $validatorRoot -AnchorFile 'gltf_validator.exe'

foreach ($selectedBackend in $backends) {
    Write-Host "Installing $selectedBackend. Model downloads and native compilation can take a long time."
    switch ($selectedBackend) {
        'Hunyuan' {
            & (Join-Path $PSScriptRoot 'scripts\setup-hunyuan-mv.ps1') -Python $Python -RuntimeRoot (Join-Path $RuntimeRoot 'hunyuan3d-2mv') -ModelCache (Join-Path $ModelRoot 'hunyuan3d-2mv') -AcceptHunyuanLicense
            & (Join-Path $PSScriptRoot 'scripts\setup-autoremesher.ps1') -Destination (Join-Path $RuntimeRoot 'autoremesher')
        }
        'SF3D' {
            & (Join-Path $PSScriptRoot 'scripts\setup.ps1') -Python $Python -RuntimeRoot (Join-Path $RuntimeRoot 'sf3d')
            Invoke-Checked -FilePath (Join-Path $RuntimeRoot 'sf3d\venv\Scripts\python.exe') -Arguments @(
                (Join-Path $PSScriptRoot 'scripts\prepare-models.py'), '--backend', 'sf3d',
                '--cache-root', (Join-Path $ModelRoot 'sf3d'), '--token-path', $HuggingFaceTokenPath
            )
        }
        'SPAR3D' {
            & (Join-Path $PSScriptRoot 'scripts\setup-spar3d.ps1') -Python $Python -RuntimeRoot (Join-Path $RuntimeRoot 'spar3d')
            Invoke-Checked -FilePath (Join-Path $RuntimeRoot 'spar3d\venv\Scripts\python.exe') -Arguments @(
                (Join-Path $PSScriptRoot 'scripts\prepare-models.py'), '--backend', 'spar3d',
                '--cache-root', (Join-Path $ModelRoot 'spar3d'), '--token-path', $HuggingFaceTokenPath
            )
        }
        'TRELLIS' { & (Join-Path $PSScriptRoot 'scripts\setup-trellis.ps1') -Python $Python -RuntimeRoot $RuntimeRoot -ModelRoot (Join-Path $ModelRoot 'trellis2-q8') }
    }
}
if ($InstallCuMesh) {
    $cumeshArguments = @{
        PythonExe = (Join-Path $RuntimeRoot 'hunyuan3d-2mv\venv\Scripts\python.exe')
        PackageDir = (Join-Path $RuntimeRoot 'shape-repair-packages')
        GpuLock = (Join-Path $RuntimeRoot 'gpu.lock')
    }
    $runStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    & (Join-Path $PSScriptRoot 'scripts\setup-cumesh-repair.ps1') @cumeshArguments -Build -QualificationBuildMemoryMiB $CuMeshBuildMemoryMiB -RunName "build-$runStamp"
    & (Join-Path $PSScriptRoot 'scripts\setup-cumesh-repair.ps1') @cumeshArguments -Probe -RunName "probe-$runStamp"
}
$settings = [ordered]@{
    schema_version = 1
    repository = $PSScriptRoot
    data_root = $DataRoot
    runtime_root = $RuntimeRoot
    model_root = $ModelRoot
    output_root = $OutputRoot
    blender = $blender
    codex_config = $CodexConfig
    skills_root = $SkillsRoot
    backends = $backends
    hf_token_path = $HuggingFaceTokenPath
    hf_token_paths = @{}
}
foreach ($legacyBackend in @('sf3d', 'spar3d')) {
    $modelReport = Join-Path (Join-Path $ModelRoot $legacyBackend) 'model-installation.json'
    if (Test-Path -LiteralPath $modelReport) {
        $modelInstallation = Get-Content -LiteralPath $modelReport -Raw | ConvertFrom-Json
        if ($modelInstallation.token_path) { $settings.hf_token_paths[$legacyBackend] = $modelInstallation.token_path }
    }
}
$settingsFile = Join-Path $DataRoot 'installation.json'
[IO.File]::WriteAllText($settingsFile, ($settings | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
# A pointer lets the checkout launchers discover custom locations without committing machine paths.
[IO.File]::WriteAllText((Join-Path $PSScriptRoot '.cognito-install.json'), ($settings | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
$configArguments = @((Join-Path $PSScriptRoot 'scripts\configure_codex.py'), '--settings', $settingsFile)
if ($NoRegisterCodex) { $configArguments += '--no-register' }
if ($NoInstallSkills) { $configArguments += '--no-skills' }
Invoke-Checked -FilePath $Python -Arguments $configArguments
Write-Host "Cognito-3D-mcp installation completed. Outputs: $OutputRoot"
Write-Host 'Restart Codex, then ask it to call server_status. Selected engine/model files are installed; GPU generation still needs a first-run verification.'
