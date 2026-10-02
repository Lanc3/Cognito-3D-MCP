param(
    [string]$Python = "py -3.11",
    [string]$RuntimeRoot = (Join-Path (Join-Path $env:LOCALAPPDATA "Cognito-3D-mcp") "runtime\sf3d"),
    [string]$Sf3dRef = "ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2",
    [string]$TorchVersion = "2.9.1",
    [string]$TorchvisionVersion = "0.24.1",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Python = Resolve-InstallerPython -Python $Python
$Root = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$Venv = Join-Path $RuntimeRoot "venv"
$PythonExe = Join-Path $Venv "Scripts\python.exe"
$Vendor = Join-Path $RuntimeRoot "stable-fast-3d"
$IsWindowsPlatform = $env:OS -eq "Windows_NT"



function Enable-Cuda13WindowsBuildWorkarounds {
    if (-not $IsWindowsPlatform) {
        return
    }

    $CudaVersion = & $PythonExe -c "import torch; print(torch.version.cuda or '')"
    if ($LASTEXITCODE -ne 0 -or -not $CudaVersion.StartsWith("13.")) {
        return
    }

    # PyTorch 2.9/2.11 CUDA 13 wheels expose a string-packer branch that nvcc
    # cannot parse with MSVC. SF3D's extensions do not instantiate this branch.
    $TorchHeader = & $PythonExe -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent / 'include/torch/csrc/dynamo/compiled_autograd.h')"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $TorchHeader)) {
        throw "Could not locate PyTorch's compiled_autograd.h for the CUDA 13 Windows workaround."
    }
    $HeaderText = [IO.File]::ReadAllText($TorchHeader).Replace("`r`n", "`n")
    $HeaderNeedle = "    } else if constexpr (::std::is_same_v<T, ::std::string>) {`n      return at::StringType::get();"
    $HeaderMarker = "Disabled for nvcc on Windows: CUDA 13.x reports ::std as ambiguous here"
    if ($HeaderText.Contains($HeaderNeedle)) {
        $HeaderReplacement = "    // $HeaderMarker`n    // SF3D does not instantiate this branch."
        [IO.File]::WriteAllText($TorchHeader, $HeaderText.Replace($HeaderNeedle, $HeaderReplacement))
    } elseif (-not $HeaderText.Contains($HeaderMarker)) {
        throw "PyTorch's CUDA 13 header layout has changed; refusing to apply an unknown workaround."
    }

    # Pass the standards-conforming MSVC preprocessor flag through nvcc.
    $TextureSetup = Join-Path $Vendor "texture_baker\setup.py"
    $SetupText = [IO.File]::ReadAllText($TextureSetup).Replace("`r`n", "`n")
    $SetupNeedle = "        `"nvcc`": [`n            `"-O3`" if not debug_mode else `"-O0`",`n        ],"
    $SetupMarker = "-Xcompiler=/Zc:preprocessor"
    if ($SetupText.Contains($SetupNeedle)) {
        $SetupReplacement = "        `"nvcc`": [`n            `"-O3`" if not debug_mode else `"-O0`",`n        ]`n        + ([`"$SetupMarker`"] if platform.system() == `"Windows`" else []),"
        [IO.File]::WriteAllText($TextureSetup, $SetupText.Replace($SetupNeedle, $SetupReplacement))
    } elseif (-not $SetupText.Contains($SetupMarker)) {
        throw "Stable Fast 3D's texture_baker setup layout has changed; refusing to apply an unknown workaround."
    }
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Invoke-Checked -FilePath $Python -Arguments @("-m", "venv", $Venv)
}

Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "--upgrade", "pip", "setuptools==69.5.1", "wheel"
)
Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "-r", (Join-Path $Root "requirements.txt")
)
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", $Root)

Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-m", "pip", "install", "torch==$TorchVersion", "torchvision==$TorchvisionVersion",
    "--index-url", $TorchIndexUrl
)
Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-c", "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available())"
)

if (-not (Test-Path (Join-Path $Vendor ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Vendor) | Out-Null
    Invoke-Checked -FilePath git -Arguments @(
        "clone", "https://github.com/Stability-AI/stable-fast-3d.git", $Vendor
    )
}
Push-Location $Vendor
try {
    Invoke-Checked -FilePath git -Arguments @("fetch", "--depth", "1", "origin", $Sf3dRef)
    Invoke-Checked -FilePath git -Arguments @("checkout", "--detach", $Sf3dRef)
    Import-VisualStudio2022Environment
    Assert-CudaRuntime -PythonExe $PythonExe
    Enable-Cuda13WindowsBuildWorkarounds
    Invoke-Checked -FilePath $PythonExe -Arguments @(
        "-m", "pip", "install", "--no-build-isolation", "-r", "requirements.txt"
    )
} finally {
    Pop-Location
}

$env:COGNITO_INSTALL_VENDOR = $Vendor
Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-c", "import os, sysconfig; from pathlib import Path; (Path(sysconfig.get_path('purelib')) / 'cognito_3d_vendor.pth').write_text(os.environ['COGNITO_INSTALL_VENDOR'] + chr(10), encoding='utf-8')"
)
Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-c", "import os; import texture_baker, uv_unwrapper; from pathlib import Path; import sys; sys.path.insert(0, os.environ['COGNITO_INSTALL_VENDOR']); from sf3d.system import SF3D; print('Stable Fast 3D imports OK')"
)

Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "check")
Write-Host "Setup complete. Authenticate with Hugging Face, then run scripts\run-server.ps1."
