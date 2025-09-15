# RAG Knowledge Assistant - Docker Setup
Write-Host "🐳 RAG Knowledge Assistant - Docker Deployment" -ForegroundColor Green
Write-Host "=" * 55

# Check if Docker is available
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not found. Please install Docker Desktop" -ForegroundColor Red
    Write-Host "   Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Cyan
    exit 1
}

# Check if docker-compose is available
try {
    $composeVersion = docker-compose --version 2>&1
    Write-Host "✅ Docker Compose found: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose not found" -ForegroundColor Red
    exit 1
}

# Check if .env file exists
if (!(Test-Path ".env")) {
    Write-Host "⚠️  .env file not found. Creating from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "🔑 Please edit .env and add your API keys:" -ForegroundColor Yellow
    Write-Host "   - OPENAI_API_KEY=your-openai-api-key" -ForegroundColor Cyan
    Write-Host "   - PINECONE_API_KEY=your-pinecone-api-key (optional)" -ForegroundColor Cyan
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 0
    }
}

Write-Host ""
Write-Host "🚀 Starting RAG Knowledge Assistant with Docker..." -ForegroundColor Yellow

# Parse command line arguments
$command = $args[0]
$detached = $args -contains "-d" -or $args -contains "--detach"

switch ($command) {
    "build" {
        Write-Host "🔨 Building Docker images..." -ForegroundColor Cyan
        docker-compose build --no-cache
    }
    "stop" {
        Write-Host "🛑 Stopping services..." -ForegroundColor Cyan
        docker-compose down
    }
    "restart" {
        Write-Host "🔄 Restarting services..." -ForegroundColor Cyan
        docker-compose restart
    }
    "logs" {
        Write-Host "📋 Showing logs..." -ForegroundColor Cyan
        docker-compose logs -f
    }
    "status" {
        Write-Host "📊 Service status..." -ForegroundColor Cyan
        docker-compose ps
    }
    "clean" {
        Write-Host "🧹 Cleaning up Docker resources..." -ForegroundColor Cyan
        docker-compose down -v --remove-orphans
        docker system prune -f
    }
    default {
        Write-Host "🏗️  Building and starting services..." -ForegroundColor Cyan
        
        if ($detached) {
            docker-compose up --build -d
        } else {
            docker-compose up --build
        }
    }
}

if ($LASTEXITCODE -eq 0 -and ($command -eq "" -or $command -eq "start")) {
    Start-Sleep -Seconds 5
    
    Write-Host ""
    Write-Host "✅ Services started successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Available services:" -ForegroundColor Yellow
    Write-Host "   📱 RAG API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "   📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "   ❤️  Health Check: http://localhost:8000/health" -ForegroundColor Cyan
    Write-Host "   📊 Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor Cyan
    Write-Host "   🔍 Prometheus: http://localhost:9090" -ForegroundColor Cyan
    Write-Host "   🧪 MLflow: http://localhost:5000" -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "🔧 Docker commands:" -ForegroundColor Yellow
    Write-Host "   View logs: .\scripts\docker.ps1 logs" -ForegroundColor Cyan
    Write-Host "   Stop services: .\scripts\docker.ps1 stop" -ForegroundColor Cyan
    Write-Host "   Restart: .\scripts\docker.ps1 restart" -ForegroundColor Cyan
    Write-Host "   Status: .\scripts\docker.ps1 status" -ForegroundColor Cyan
    Write-Host "   Clean up: .\scripts\docker.ps1 clean" -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "🧪 Testing the API..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 10
        if ($response.status -eq "healthy") {
            Write-Host "✅ API is responding correctly!" -ForegroundColor Green
        } else {
            Write-Host "⚠️  API responded but status is: $($response.status)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  API might still be starting up..." -ForegroundColor Yellow
        Write-Host "   Check status with: .\scripts\docker.ps1 status" -ForegroundColor Cyan
        Write-Host "   View logs with: .\scripts\docker.ps1 logs" -ForegroundColor Cyan
    }
}

if ($command -ne "logs") {
    Write-Host ""
    Write-Host "📋 Quick commands:" -ForegroundColor Yellow
    Write-Host "   .\scripts\docker.ps1         # Start services" -ForegroundColor Cyan
    Write-Host "   .\scripts\docker.ps1 logs    # View logs" -ForegroundColor Cyan
    Write-Host "   .\scripts\docker.ps1 stop    # Stop services" -ForegroundColor Cyan
    Write-Host "   .\scripts\docker.ps1 clean   # Clean up" -ForegroundColor Cyan
}
