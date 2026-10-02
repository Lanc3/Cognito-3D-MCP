param(
    [string]$Python = "py -3.11",
    [string]$RuntimeRoot = (Join-Path (Join-Path $env:LOCALAPPDATA "Cognito-3D-mcp") "runtime"),
    [string]$ModelRoot = (Join-Path (Join-Path $env:LOCALAPPDATA "Cognito-3D-mcp") "models\trellis2-q8")
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Python = Resolve-InstallerPython -Python $Python
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$Venv = Join-Path $RuntimeRoot "trellis-venv"
$PythonExe = Join-Path $Venv "Scripts\python.exe"
$RuntimeDir = Join-Path $RuntimeRoot "trellis-v0.5.4"
$RealEsrganDir = Join-Path $RuntimeRoot "realesrgan-v0.2.5.0"
$ValidatorDir = Join-Path $RuntimeRoot "gltf-validator"
$DownloadDir = Join-Path $RuntimeRoot "downloads"
$TemporaryDir = Join-Path $RuntimeRoot "temp"
$PipCache = Join-Path $RuntimeRoot "pip-cache"
$WeightsRevision = "a57397bd3d351599d9729fc144b3f87c3f87d65b"
$RuntimeSha = "f7d2912b064bf1520f03e025c5eb344df6b347ad04831ac7aa04d847581bd7ad"
$RealEsrganSha = "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d"
$ValidatorSha = "c5068f51205deedc28acc3529ee7e11ee60e853454f673093398eba80142202c"





Assert-FreeSpace -Path $RuntimeRoot -MinimumGB 20
Assert-FreeSpace -Path $ModelRoot -MinimumGB 20

New-Item -ItemType Directory -Force -Path $RuntimeRoot,$ModelRoot,$DownloadDir,$TemporaryDir,$PipCache | Out-Null
$env:TEMP = $TemporaryDir
$env:TMP = $TemporaryDir
$env:PIP_CACHE_DIR = $PipCache
$env:HF_HOME = Join-Path $ModelRoot "dino-cache"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Invoke-Checked -FilePath $Python -Arguments @("-m", "venv", $Venv)
}
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", "pip", "wheel")
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "install", "-r", (Join-Path $ProjectRoot "requirements.txt"))
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", "${ProjectRoot}[trellis]")
& $PythonExe -c "import torch, torchvision"
if ($LASTEXITCODE -ne 0) {
    Invoke-Checked -FilePath $PythonExe -Arguments @(
        "-m", "pip", "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    )
}

$RuntimeArchive = Join-Path $DownloadDir "trellis-cuda-windows-x64-v0.5.4.zip"
Get-VerifiedDownload `
    -Url "https://github.com/pwilkin/trellis.cpp/releases/download/v0.5.4/trellis-cuda-windows-x64.zip" `
    -Destination $RuntimeArchive -Sha256 $RuntimeSha
Expand-VersionedArchive -Archive $RuntimeArchive -Destination $RuntimeDir -AnchorFile "trellis-server.exe"

$WeightHashes = [ordered]@{
    "birefnet.gguf" = "10c5dd4dcac904cf81c9a16180eb66f167dd52ab55867f54a44567b0b2babbc1"
    "dinov3.gguf" = "0dd4ffd4b46a248f5b7d49c35275d68461fbf73f57ddb4c1fa8afb4f7bb45a0d"
    "shape_dec.gguf" = "0de7c7a675022dd8696d526a9279e5a50b2a35c8a79452f434a72dc53d40f169"
    "shape_flow_1024.gguf" = "997e9fc10ab95fda11c4cd1cbaf425101980ac36a2ef80e9c83e0b9c4bfc9680"
    "shape_flow_512.gguf" = "29b639f4ff22ded8f91b619376a835f64b9874b0ebcadb9ac6c305195bf5d1f9"
    "ss_dec.gguf" = "2790b5eecb261cc877d9bf175ce2bd6dd48cd65be8c042c5f5bc023dfca01cf7"
    "ss_flow.gguf" = "ea6d8a42b20661a5c6a52e5ffbdc1df7aca9b212792193873b418c69f871422c"
    "tex_dec.gguf" = "88b4fced46455e02f316664d5c43584a311921dd9a1cdc1b7b7d981cca9214d4"
    "tex_flow_1024.gguf" = "cb2cb3aee74ba09c018f918ed8c146bc7e4f96335b61fa0d1fc1ff1a7811e6da"
    "tex_flow_512.gguf" = "389a2cbdda59d53b21e5989650d9d36b7ac603266eaef06712cd07a9fc377210"
}
foreach ($Entry in $WeightHashes.GetEnumerator()) {
    $Url = "https://huggingface.co/ilintar/trellis2-gguf/resolve/$WeightsRevision/q8/$($Entry.Key)?download=true"
    Get-VerifiedDownload -Url $Url -Destination (Join-Path $ModelRoot $Entry.Key) -Sha256 $Entry.Value
}

$RealEsrganArchive = Join-Path $DownloadDir "realesrgan-ncnn-vulkan-20220424-windows.zip"
Get-VerifiedDownload `
    -Url "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip" `
    -Destination $RealEsrganArchive -Sha256 $RealEsrganSha
Expand-VersionedArchive -Archive $RealEsrganArchive -Destination $RealEsrganDir `
    -AnchorFile "realesrgan-ncnn-vulkan.exe"

$ValidatorArchive = Join-Path $DownloadDir "gltf_validator-2.0.0-dev.3.10-win64.zip"
Get-VerifiedDownload `
    -Url "https://github.com/KhronosGroup/glTF-Validator/releases/download/2.0.0-dev.3.10/gltf_validator-2.0.0-dev.3.10-win64.zip" `
    -Destination $ValidatorArchive -Sha256 $ValidatorSha
Expand-VersionedArchive -Archive $ValidatorArchive -Destination $ValidatorDir `
    -AnchorFile "gltf_validator.exe"

$env:COGNITO_INSTALL_HF_HOME = $env:HF_HOME
Invoke-Checked -FilePath $PythonExe -Arguments @(
    "-c",
    "import os; from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-small', revision='ed25f3a31f01632728cabb09d1542f84ab7b0056', cache_dir=os.environ['COGNITO_INSTALL_HF_HOME'])"
)

$Manifest = [ordered]@{
    installed_at = [DateTime]::UtcNow.ToString("o")
    runtime_version = "v0.5.4"
    runtime_commit = "ae1a63757264eec5bfba84b94cf59ddcc161e537"
    archive_sha256 = $RuntimeSha
    weights_revision = $WeightsRevision
    weights = $WeightHashes
    realesrgan_sha256 = $RealEsrganSha
    gltf_validator_sha256 = $ValidatorSha
}
$ManifestPath = Join-Path $RuntimeDir "install-manifest.json"
$ManifestJson = $Manifest | ConvertTo-Json -Depth 4
[IO.File]::WriteAllText($ManifestPath, $ManifestJson, [Text.UTF8Encoding]::new($false))

Invoke-Checked -FilePath (Join-Path $RuntimeDir "trellis-server.exe") -Arguments @("--help")
Invoke-Checked -FilePath (Join-Path $RealEsrganDir "realesrgan-ncnn-vulkan.exe") -Arguments @("-h")
Invoke-Checked -FilePath (Join-Path $ValidatorDir "gltf_validator.exe") -Arguments @("--help")

Write-Host "TRELLIS production runtime installed at $RuntimeRoot. Run scripts\run-trellis-server.ps1."
