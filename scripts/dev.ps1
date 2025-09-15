# RAG Knowledge Assistant - Development Server
Write-Host "🚀 Starting RAG Knowledge Assistant in development mode..." -ForegroundColor Green

# Check if virtual environment exists
if (!(Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found. Run setup first:" -ForegroundColor Red
    Write-Host "   .\scripts\setup.ps1" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

# Check if .env file exists
if (!(Test-Path ".env")) {
    Write-Host "⚠️  .env file not found. Creating from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "🔑 Please edit .env and add your API keys before continuing" -ForegroundColor Yellow
}

# Set development environment variables
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
$env:DEBUG = "true"
$env:LOG_LEVEL = "DEBUG"

# Create logs directory if it doesn't exist
if (!(Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
}

Write-Host ""
Write-Host "🌐 Starting FastAPI development server..." -ForegroundColor Green
Write-Host ""
Write-Host "📋 Server Information:" -ForegroundColor Yellow
Write-Host "   🌍 URL: http://localhost:8000" -ForegroundColor Cyan
Write-Host "   📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "   📖 ReDoc: http://localhost:8000/redoc" -ForegroundColor Cyan
Write-Host "   ❤️  Health: http://localhost:8000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔧 Development Features:" -ForegroundColor Yellow
Write-Host "   ♻️  Auto-reload: Enabled" -ForegroundColor Green
Write-Host "   🐛 Debug mode: Enabled" -ForegroundColor Green
Write-Host "   📝 Detailed logs: Enabled" -ForegroundColor Green
Write-Host ""
Write-Host "🚫 Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host "=" * 60

# Change to backend directory and start server
Set-Location backend
try {
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
} catch {
    Write-Host ""
    Write-Host "❌ Server failed to start. Check the error above." -ForegroundColor Red
    Write-Host ""
    Write-Host "🔍 Common issues:" -ForegroundColor Yellow
    Write-Host "   • Port 8000 already in use" -ForegroundColor Cyan
    Write-Host "   • Missing dependencies (run .\scripts\setup.ps1)" -ForegroundColor Cyan
    Write-Host "   • Invalid configuration in .env file" -ForegroundColor Cyan
    Write-Host "   • Missing API keys in .env file" -ForegroundColor Cyan
} finally {
    Set-Location ..
}

Write-Host ""
Write-Host "👋 Development server stopped" -ForegroundColor Yellow
