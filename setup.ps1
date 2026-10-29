# Sets up the Python 3.12 + CUDA PyTorch + Unsloth environment (Windows, NVIDIA GPU)
Set-Location $PSScriptRoot

$have = py -0p 2>&1 | Select-String "3.12"
if (-not $have) {
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
}
$ErrorActionPreference = "Stop"

py -3.12 -m venv .venv312
$py = ".\.venv312\Scripts\python.exe"
& $py -m pip install --upgrade pip

# CUDA torch first, pinned to what Unsloth accepts
& $py -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install unsloth

# Unsloth can swap in CPU torch; restore CUDA build if so
$ok = & $py -c "import torch; print(torch.cuda.is_available())"
if ($ok -ne "True") {
    & $py -m pip uninstall -y torch torchvision
    & $py -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
}
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"