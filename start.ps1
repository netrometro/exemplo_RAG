if (-not (Test-Path ".venv")) {
    Write-Host "Criando o ambiente virtual (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "Ativando o ambiente virtual..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1
