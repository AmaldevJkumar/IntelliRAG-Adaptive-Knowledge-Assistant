# RAG Knowledge Assistant - Setup Script
Write-Host "🚀 RAG Knowledge Assistant - Setup & Installation" -ForegroundColor Green
Write-Host "=" * 60

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow

# Check Python version
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    
    # Verify Python 3.11+
    if ($pythonVersion -match "Python 3\.(\d+)") {
        $minorVersion = [int]$matches[1]
        if ($minorVersion -lt 11) {
            Write-Host "❌ Python 3.11+ required. Current: $pythonVersion" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+" -ForegroundColor Red
    Write-Host "   Download from: https://www.python.org/downloads/" -ForegroundColor Cyan
    exit 1
}

# Check Git
try {
    $gitVersion = git --version 2>&1
    Write-Host "✅ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git not found. Please install Git" -ForegroundColor Red
    Write-Host "   Download from: https://git-scm.com/downloads" -ForegroundColor Cyan
    exit 1
}

# Check Docker (optional)
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
    $dockerAvailable = $true
} catch {
    Write-Host "⚠️  Docker not found (optional for local development)" -ForegroundColor Yellow
    $dockerAvailable = $false
}

Write-Host ""
Write-Host "🔧 Setting up Python environment..." -ForegroundColor Yellow

# Create virtual environment
if (Test-Path "venv") {
    Write-Host "📁 Virtual environment already exists" -ForegroundColor Cyan
} else {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Download spaCy model
Write-Host "🤖 Downloading spaCy language model..." -ForegroundColor Cyan
python -m spacy download en_core_web_sm
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Failed to download spaCy model (continuing anyway)" -ForegroundColor Yellow
}

# Setup environment file
Write-Host ""
Write-Host "⚙️  Setting up configuration..." -ForegroundColor Yellow

if (!(Test-Path ".env")) {
    Write-Host "📝 Creating .env file from template..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
    Write-Host "🔑 Please edit .env file and add your API keys:" -ForegroundColor Yellow
    Write-Host "   - OPENAI_API_KEY=your-openai-api-key" -ForegroundColor Cyan
    Write-Host "   - PINECONE_API_KEY=your-pinecone-api-key (optional)" -ForegroundColor Cyan
} else {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
}

# Create necessary directories
Write-Host "📁 Creating necessary directories..." -ForegroundColor Cyan
$directories = @("logs", "data/faiss_index", "data/temp", "data/uploads")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   Created: $dir" -ForegroundColor Gray
    }
}

# Initialize database (if needed)
Write-Host ""
Write-Host "🗄️  Database setup..." -ForegroundColor Yellow
Write-Host "   For production: Setup PostgreSQL and update DATABASE_URL in .env" -ForegroundColor Cyan
Write-Host "   For development: SQLite will be used automatically" -ForegroundColor Cyan

# Run tests to verify setup
Write-Host ""
Write-Host "🧪 Running setup verification tests..." -ForegroundColor Yellow
try {
    pytest tests/test_setup.py -v
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Setup verification passed!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some setup verification tests failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not run setup verification tests" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Edit .env file with your API keys"
Write-Host "   2. Run: .\scripts\dev.ps1 (start development server)"
Write-Host "   3. Run: .\scripts\test.ps1 (run tests)"
Write-Host "   4. Visit: http://localhost:8000/docs (API documentation)"
Write-Host ""
Write-Host "🔗 Useful commands:" -ForegroundColor Yellow
Write-Host "   Development server: .\scripts\dev.ps1"
Write-Host "   Run tests: .\scripts\test.ps1"
Write-Host "   Docker setup: .\scripts\docker.ps1"
Write-Host "   API docs: http://localhost:8000/docs"
Write-Host ""

if ($dockerAvailable) {
    Write-Host "🐳 Docker is available for containerized deployment" -ForegroundColor Green
    Write-Host "   Run: .\scripts\docker.ps1 for containerized setup"
}

Write-Host "✨ Happy coding!" -ForegroundColor Cyan
