#Requires -Version 5.1
param(
    [string]$PythonExe = (Join-Path (Join-Path $env:LOCALAPPDATA 'Cognito-3D-mcp') 'runtime\hunyuan3d-2mv\venv\Scripts\python.exe'),
    [string]$PackageDir = (Join-Path (Join-Path $env:LOCALAPPDATA 'Cognito-3D-mcp') 'runtime\shape-repair-packages'),
    [string]$SourceDir,
    [string]$EvidenceDir,
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$GpuLock = (Join-Path (Join-Path $env:LOCALAPPDATA 'Cognito-3D-mcp') 'runtime\gpu.lock'),
    [ValidateSet(2048,2560,3072,3584,4096)][int]$QualificationBuildMemoryMiB = 2048,
    [string]$RunName,
    [switch]$Build,
    [switch]$Probe
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
if ($Build -and $Probe) { throw 'Build and probe must run as separate serial invocations.' }
if (-not $Build -and $QualificationBuildMemoryMiB -ne 2048) { throw 'The build override cannot raise a probe/runtime cap.' }
if ($QualificationBuildMemoryMiB -gt 2048 -and -not $RunName) { throw 'An increased build cap requires an explicit unique -RunName.' }
$PythonExe = Resolve-InstallerPython -Python $PythonExe
$PackageDir = [IO.Path]::GetFullPath($PackageDir)
if (-not $SourceDir) { $SourceDir = Join-Path (Split-Path -Parent $PackageDir) 'cumesh-repair-src' }
if (-not $EvidenceDir) { $EvidenceDir = Join-Path (Split-Path -Parent $PackageDir) 'cumesh-qualification' }
$SourceDir = [IO.Path]::GetFullPath($SourceDir)
$EvidenceDir = [IO.Path]::GetFullPath($EvidenceDir)
$commit = '12289e1062f0603f2f0d0771b02e1395d247f26f'
if (-not (Test-Path -LiteralPath (Join-Path $SourceDir '.git'))) {
    if ((Test-Path -LiteralPath $SourceDir) -and @(Get-ChildItem -LiteralPath $SourceDir -Force).Count -gt 0) {
        throw "CuMesh source directory already contains files: $SourceDir"
    }
    Invoke-Checked -FilePath git.exe -Arguments @('clone', '--filter=blob:none', 'https://github.com/JeffreyXiang/CuMesh.git', $SourceDir)
}
Invoke-Checked -FilePath git.exe -Arguments @('-C', $SourceDir, 'fetch', '--depth', '1', 'origin', $commit)
Invoke-Checked -FilePath git.exe -Arguments @('-C', $SourceDir, 'checkout', '--detach', $commit)
Invoke-Checked -FilePath git.exe -Arguments @('-C', $SourceDir, 'submodule', 'update', '--init', '--recursive')
if ($Build -or $Probe) {
    if (-not $CudaRoot) { throw 'Set CUDA_PATH or pass -CudaRoot for the CUDA Toolkit 13.0 installation.' }
    $env:CUDA_HOME = $CudaRoot
    $env:CUDA_PATH = $CudaRoot
    $env:PATH = (Join-Path $CudaRoot 'bin') + ';' + $env:PATH
    Import-VisualStudio2022Environment
    Assert-CudaRuntime -PythonExe $PythonExe
}
$env:MAX_JOBS = '1'
$env:BUILD_TARGET = 'cuda'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
if ($Build) {
    Invoke-Checked -FilePath $PythonExe -Arguments @('-m', 'pip', 'install', 'ninja>=1.11,<2', 'pybind11>=2.13,<4')
}
$mode = if ($Probe) { 'probe' } elseif ($Build) { 'build' } else { 'metadata' }
$workerArgs = @('-B', (Join-Path $PSScriptRoot 'cumesh-build-worker.py'), '--mode', $mode,
    '--packages', $PackageDir, '--source', $SourceDir, '--evidence', $EvidenceDir, '--gpu-lock', $GpuLock,
    '--qualification-build-memory-mb', $QualificationBuildMemoryMiB)
if ($RunName) { $workerArgs += @('--run-name', $RunName) }
Invoke-Checked -FilePath $PythonExe -Arguments $workerArgs
Write-Host "CuMesh $mode completed. A successful CUDA probe enables experimental candidates; full common-worker qualification is still required for production readiness."
