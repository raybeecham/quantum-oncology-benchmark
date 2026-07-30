param(
    [string]$Owner = "raybeecham",
    [string]$Repository = "quantum-oncology-benchmark"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is required. Install it, then run: gh auth login"
}

gh auth status | Out-Host

if (-not (Test-Path .git)) {
    git init -b main
    git add .
    git commit -m "Initialize quantum oncology benchmark"
}

$remote = git remote get-url origin 2>$null
if (-not $remote) {
    gh repo create "$Owner/$Repository" --public --source . --remote origin --push --description "Reproducible classical and quantum machine-learning benchmarks for cancer classification research."
} else {
    git push -u origin main
}
