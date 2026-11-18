# Build the Paper 3 manuscript:  numbers/tables from results -> XeLaTeX -> BibTeX -> XeLaTeX x2.
# TeX Live 2025 is used read-only; every TeX cache/config is redirected into work\.texmf_var.
param([string]$Python = "python")
$ErrorActionPreference = "Continue"   # native tools write progress to stderr
$WORK = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$MS = Join-Path $WORK "manuscript"
$tv = Join-Path $WORK ".texmf_var"
$env:PATH = "C:\texlive\2025\bin\windows;" + $env:PATH
$env:TEXMFVAR = $tv; $env:TEXMFCACHE = $tv; $env:TEXMFCONFIG = "$tv\config"; $env:TEXMFHOME = "$tv\home"
& $Python -W ignore (Join-Path $PSScriptRoot "make_numbers.py") 2>&1 | Out-File -Encoding utf8 (Join-Path $MS "build_numbers.log")
"make_numbers exit: $LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { "make_numbers failed (see manuscript\build_numbers.log); build stopped"; exit 1 }
Set-Location $MS
foreach ($step in @("xelatex", "bibtex", "xelatex", "xelatex")) {
    if ($step -eq "bibtex") { bibtex main | Out-File -Encoding utf8 build_bibtex.log }
    else { xelatex -interaction=nonstopmode -halt-on-error main.tex | Out-File -Encoding utf8 build_xelatex.log }
    "$step exit: $LASTEXITCODE"
}
Select-String -Path main.log -Pattern "undefined|Undefined number key|multiply defined|Overfull" | Select-Object -First 40 | ForEach-Object { $_.Line }
