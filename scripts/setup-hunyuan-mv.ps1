param(
    [string]$Python = "py -3.11",
    [string]$RuntimeRoot = (Join-Path (Join-Path $env:LOCALAPPDATA "Cognito-3D-mcp") "runtime\hunyuan3d-2mv"),
    [string]$ModelCache = (Join-Path (Join-Path $env:LOCALAPPDATA "Cognito-3D-mcp") "models\hunyuan3d-2mv"),
    [string]$UpstreamCommit = "f8db63096c8282cb27354314d896feba5ba6ff8a",
    [string]$ShapeRevision = "3a761b539b29fe4ff64714813aa9560fd66f5de0",
    [string]$TextureRevision = "9cd649ba6913f7a852e3286bad86bfa9a2d83dcf",
    [string]$TorchVersion = "2.9.1",
    [string]$TorchvisionVersion = "0.24.1",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
    [string]$DiffusersVersion = "0.31.0",
    [string]$TransformersVersion = "4.48.3",
    [string]$HuggingFaceHubVersion = "0.26.5",
    [string]$AccelerateVersion = "1.1.1",
    [string]$GradioVersion = "4.44.1",
    [string]$RembgVersion = "2.0.67",
    [int]$MinimumFreeGB = 45,
    [switch]$AcceptHunyuanLicense
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Assert-HunyuanTerms -Accepted:$AcceptHunyuanLicense
$Python = Resolve-InstallerPython -Python $Python
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$ModelCache = [IO.Path]::GetFullPath($ModelCache)
$Venv = Join-Path $RuntimeRoot "venv"
$PythonExe = Join-Path $Venv "Scripts\python.exe"
$Upstream = Join-Path $RuntimeRoot "Hunyuan3D-2"
$TempRoot = Join-Path $RuntimeRoot "temp"
Assert-FreeSpace -Path $RuntimeRoot -MinimumGB $MinimumFreeGB
Assert-FreeSpace -Path $ModelCache -MinimumGB $MinimumFreeGB


function Enable-Cuda13WindowsBuildWorkarounds {
    $CudaVersion = & $PythonExe -c "import torch; print(torch.version.cuda or '')"
    if ($LASTEXITCODE -ne 0 -or -not $CudaVersion.StartsWith("13.")) {
        return
    }

    $TorchHeader = & $PythonExe -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent / 'include/torch/csrc/dynamo/compiled_autograd.h')"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $TorchHeader)) {
        throw "Could not locate PyTorch's compiled_autograd.h for the CUDA 13 workaround."
    }
    $HeaderText = [IO.File]::ReadAllText($TorchHeader).Replace("`r`n", "`n")
    $HeaderNeedle = "    } else if constexpr (::std::is_same_v<T, ::std::string>) {`n      return at::StringType::get();"
    $HeaderMarker = "Disabled for nvcc on Windows: CUDA 13.x reports ::std as ambiguous here"
    if ($HeaderText.Contains($HeaderNeedle)) {
        $HeaderReplacement = "    // $HeaderMarker`n    // Hunyuan does not instantiate this branch."
        [IO.File]::WriteAllText($TorchHeader, $HeaderText.Replace($HeaderNeedle, $HeaderReplacement))
    } elseif (-not $HeaderText.Contains($HeaderMarker)) {
        throw "PyTorch's CUDA header layout changed; refusing an unknown patch."
    }

    $RasterizerSetup = Join-Path $Upstream "hy3dgen\texgen\custom_rasterizer\setup.py"
    $SetupText = [IO.File]::ReadAllText($RasterizerSetup).Replace("`r`n", "`n")
    $SetupNeedle = "custom_rasterizer_module = CUDAExtension('custom_rasterizer_kernel', [`n    'lib/custom_rasterizer_kernel/rasterizer.cpp',`n    'lib/custom_rasterizer_kernel/grid_neighbor.cpp',`n    'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',`n])"
    $SetupMarker = "-Xcompiler=/Zc:preprocessor"
    if ($SetupText.Contains($SetupNeedle)) {
        $SetupBase = $SetupNeedle.Substring(0, $SetupNeedle.Length - 2)
        $SetupReplacement = "$SetupBase,`n    extra_compile_args={'nvcc': ['$SetupMarker']},`n)"
        [IO.File]::WriteAllText($RasterizerSetup, $SetupText.Replace($SetupNeedle, $SetupReplacement))
    } elseif (-not $SetupText.Contains($SetupMarker)) {
        throw "Hunyuan rasterizer setup layout changed; refusing an unknown patch."
    }
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot, $ModelCache, $TempRoot | Out-Null
$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:HF_HOME = $ModelCache
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ModelCache "hub"
$env:TORCH_HOME = Join-Path $ModelCache "torch"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Invoke-Checked -FilePath $Python -Arguments @("-m", "venv", $Venv)
}

Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
)
Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "torch==$TorchVersion", "torchvision==$TorchvisionVersion",
    "--index-url", $TorchIndexUrl
)

if (-not (Test-Path -LiteralPath (Join-Path $Upstream ".git"))) {
    Invoke-Checked -FilePath git -Arguments @(
        "clone", "--filter=blob:none", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git", $Upstream
    )
}
Push-Location $Upstream
try {
    Invoke-Checked -FilePath git -Arguments @("fetch", "--depth", "1", "origin", $UpstreamCommit)
    Invoke-Checked -FilePath git -Arguments @("checkout", "--detach", $UpstreamCommit)
    Import-VisualStudio2022Environment
    Assert-CudaRuntime -PythonExe $PythonExe
    Enable-Cuda13WindowsBuildWorkarounds
    Invoke-Checked -FilePath $PythonExe -Arguments @(
        "-m", "pip", "install", "--no-build-isolation", "-r", "requirements.txt"
    )
    Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "install", "-e", ".")
    Push-Location "hy3dgen\texgen\custom_rasterizer"
    try {
        Invoke-Checked -FilePath $PythonExe -Arguments @("setup.py", "install")
    } finally {
        Pop-Location
    }
    Push-Location "hy3dgen\texgen\differentiable_renderer"
    try {
        Invoke-Checked -FilePath $PythonExe -Arguments @("setup.py", "install")
    } finally {
        Pop-Location
    }
} finally {
    Pop-Location
}

Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", $ProjectRoot)
$ShapeLoaderId = (& $PythonExe -B -c "from codex_3d_mcp.hunyuan_mv.safe_loader import SAFE_LOADER_ID; print(SAFE_LOADER_ID)").Trim()
if ($LASTEXITCODE -ne 0 -or -not $ShapeLoaderId) {
    throw "Could not resolve the safe Hunyuan shape loader."
}
# Hunyuan Paint is incompatible with the latest unbounded Hugging Face stack.
# In particular, Diffusers 0.39 loads its custom UNet under two module names,
# breaking CPU offload. Keep this tested set together on clean installs.
Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install",
    "diffusers==$DiffusersVersion",
    "transformers==$TransformersVersion",
    "huggingface-hub==$HuggingFaceHubVersion",
    "accelerate==$AccelerateVersion",
    "gradio==$GradioVersion",
    "rembg==$RembgVersion",
    "hf_xet"
)
& $PythonExe -m pip show hf-gradio *> $null
if ($LASTEXITCODE -eq 0) {
    Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "uninstall", "-y", "hf-gradio")
}

Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "-r", (Join-Path $ProjectRoot "requirements-repair.txt")
)
Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-c", "import numpy, scipy, trimesh, pymeshlab, manifold3d, rtree; print('CPU repair geometry dependencies imported')"
)

Invoke-Checked -FilePath $PythonExe -Arguments @(
    (Join-Path $ProjectRoot "scripts\download-hunyuan-models.py"),
    "--shape-revision", $ShapeRevision,
    "--texture-revision", $TextureRevision
)
Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-c", "import torch, hy3dgen, custom_rasterizer, mesh_processor; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
)
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "check")

$Manifest = [ordered]@{
    installed_at = [DateTime]::UtcNow.ToString("o")
    upstream_commit = $UpstreamCommit
    shape_revision = $ShapeRevision
    texture_revision = $TextureRevision
    shape_loader = $ShapeLoaderId
    torch = $TorchVersion
    torchvision = $TorchvisionVersion
    torch_index = $TorchIndexUrl
    diffusers = $DiffusersVersion
    transformers = $TransformersVersion
    huggingface_hub = $HuggingFaceHubVersion
    accelerate = $AccelerateVersion
    gradio = $GradioVersion
    rembg = $RembgVersion
    cuda_verified = $true
    gpu = [ordered]@{
        detail = (& $PythonExe -c 'import torch; print(torch.cuda.get_device_name(0))').Trim()
    }
}
$Manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RuntimeRoot "install-manifest.json") -Encoding utf8
Write-Host "Hunyuan3D-2mv setup complete. Restart Codex after MCP registration."
