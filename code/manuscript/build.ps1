# Build the CosCert manuscript and standalone Supplementary Material:
#   numbers/tables from results -> XeLaTeX -> BibTeX -> XeLaTeX x2.
# TeX caches/config are redirected into .texmf_var inside the project folder.
param([string]$Python = "python", [string]$TexBin = "")
$ErrorActionPreference = "Stop"
$WORK = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$MS = Join-Path $WORK "manuscript"
$tv = Join-Path $WORK ".texmf_var"
if ($TexBin -and (Test-Path $TexBin)) { $env:PATH = "$TexBin;" + $env:PATH }
$env:TEXMFVAR = $tv; $env:TEXMFCACHE = $tv; $env:TEXMFCONFIG = "$tv\config"; $env:TEXMFHOME = "$tv\home"
& $Python -W ignore (Join-Path $PSScriptRoot "make_numbers.py") 2>&1 | Out-File -Encoding utf8 (Join-Path $MS "build_numbers.log")
"make_numbers exit: $LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { "make_numbers failed (see manuscript\build_numbers.log); build stopped"; exit 1 }
Set-Location $MS
foreach ($doc in @("main", "supplementary")) {
    if (-not (Test-Path "$doc.tex")) { continue }
    foreach ($step in @("xelatex", "bibtex", "xelatex", "xelatex")) {
        if ($step -eq "bibtex") { bibtex $doc | Out-File -Encoding utf8 "build_${doc}_bibtex.log" }
        else { xelatex -interaction=nonstopmode -halt-on-error "$doc.tex" | Out-File -Encoding utf8 "build_${doc}_xelatex.log" }
        if ($LASTEXITCODE -ne 0) { throw "$step failed for $doc (exit $LASTEXITCODE)" }
    }
    Select-String -Path "$doc.log" -Pattern "undefined|Undefined number key|multiply defined|Overfull" | Select-Object -First 40 | ForEach-Object { $_.Line }
}
