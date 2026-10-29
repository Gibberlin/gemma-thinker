Set-Location C:\dev\gemma-thinker

# 1. Python 3.12: install if missing
$have = py -0p 2>&1 | Select-String "3.12"
if (-not $have) {
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
}
$ErrorActionPreference = "Stop"
# 2. fresh venv
py -3.12 -m venv .venv312
$py = ".\.venv312\Scripts\python.exe"
& $py -m pip install --upgrade pip

# 3. CUDA torch first, pinned to what Unsloth accepts
& $py -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Unsloth
& $py -m pip install unsloth

# 5. verify; if Unsloth swapped in CPU torch, put CUDA back
$ok = & $py -c "import torch; print(torch.cuda.is_available())"
if ($ok -ne "True") {
    & $py -m pip uninstall -y torch torchvision
    & $py -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
}
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"