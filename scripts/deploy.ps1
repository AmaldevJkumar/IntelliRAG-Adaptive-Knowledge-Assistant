# RAG Knowledge Assistant - Deployment Script
Write-Host "🚀 RAG Knowledge Assistant - Production Deployment" -ForegroundColor Green
Write-Host "=" * 60

# Parse command line arguments
$environment = $args[0]
$skipTests = $args -contains "--skip-tests"
$dryRun = $args -contains "--dry-run"

if (!$environment) {
    Write-Host "❌ Environment not specified!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage: .\scripts\deploy.ps1 <environment> [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Environments:" -ForegroundColor Yellow
    Write-Host "   staging    - Deploy to staging environment" -ForegroundColor Cyan
    Write-Host "   production - Deploy to production environment" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "   --skip-tests - Skip running tests before deployment" -ForegroundColor Cyan
    Write-Host "   --dry-run    - Show what would be deployed without deploying" -ForegroundColor Cyan
    exit 1
}

if ($environment -notin @("staging", "production")) {
    Write-Host "❌ Invalid environment: $environment" -ForegroundColor Red
    Write-Host "Valid environments: staging, production" -ForegroundColor Yellow
    exit 1
}

Write-Host "🎯 Target environment: $environment" -ForegroundColor Cyan
if ($dryRun) {
    Write-Host "👀 DRY RUN MODE - No actual deployment will occur" -ForegroundColor Yellow
}

# Pre-deployment checks
Write-Host ""
Write-Host "🔍 Pre-deployment checks..." -ForegroundColor Yellow

# Check if kubectl is available
try {
    $kubectlVersion = kubectl version --client --short 2>&1
    Write-Host "✅ kubectl found: $kubectlVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ kubectl not found. Please install kubectl" -ForegroundColor Red
    exit 1
}

# Check if Docker is available
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not found. Please install Docker" -ForegroundColor Red
    exit 1
}

# Check Kubernetes context
$currentContext = kubectl config current-context 2>&1
Write-Host "🎯 Current Kubernetes context: $currentContext" -ForegroundColor Cyan

# Confirm deployment to production
if ($environment -eq "production" -and !$dryRun) {
    Write-Host ""
    Write-Host "⚠️  PRODUCTION DEPLOYMENT WARNING!" -ForegroundColor Red
    Write-Host "You are about to deploy to PRODUCTION environment." -ForegroundColor Yellow
    Write-Host "Context: $currentContext" -ForegroundColor Yellow
    Write-Host ""
    $confirm = Read-Host "Are you sure you want to continue? Type 'DEPLOY' to confirm"
    
    if ($confirm -ne "DEPLOY") {
        Write-Host "❌ Deployment cancelled" -ForegroundColor Red
        exit 0
    }
}

# Run tests unless skipped
if (!$skipTests) {
    Write-Host ""
    Write-Host "🧪 Running tests before deployment..." -ForegroundColor Yellow
    
    & ".\scripts\test.ps1" "fast"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Tests failed! Deployment aborted." -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Tests passed!" -ForegroundColor Green
}

# Build and tag Docker image
$imageTag = "rag-knowledge-assistant:$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$latestTag = "rag-knowledge-assistant:latest"

Write-Host ""
Write-Host "🏗️  Building Docker image..." -ForegroundColor Yellow
Write-Host "Image tag: $imageTag" -ForegroundColor Cyan

if (!$dryRun) {
    docker build -t $imageTag -t $latestTag .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker build failed!" -ForegroundColor Red
        exit 1
    }
}

# Push to registry (if configured)
$registry = $env:DOCKER_REGISTRY
if ($registry) {
    Write-Host ""
    Write-Host "📤 Pushing to Docker registry..." -ForegroundColor Yellow
    
    $registryImage = "$registry/$imageTag"
    $registryLatest = "$registry/$latestTag"
    
    if (!$dryRun) {
        docker tag $imageTag $registryImage
        docker tag $latestTag $registryLatest
        docker push $registryImage
        docker push $registryLatest
    } else {
        Write-Host "Would push: $registryImage" -ForegroundColor Gray
        Write-Host "Would push: $registryLatest" -ForegroundColor Gray
    }
}

# Deploy to Kubernetes
Write-Host ""
Write-Host "☸️  Deploying to Kubernetes..." -ForegroundColor Yellow

# Create namespace if it doesn't exist
if (!$dryRun) {
    kubectl apply -f deployment/kubernetes/namespace.yaml
} else {
    Write-Host "Would apply: deployment/kubernetes/namespace.yaml" -ForegroundColor Gray
}

# Apply configurations
$kubernetesFiles = @(
    "deployment/kubernetes/configmap.yaml",
    "deployment/kubernetes/secrets.yaml",
    "deployment/kubernetes/deployment.yaml",
    "deployment/kubernetes/service.yaml",
    "deployment/kubernetes/ingress.yaml",
    "deployment/kubernetes/hpa.yaml"
)

foreach ($file in $kubernetesFiles) {
    if (Test-Path $file) {
        if (!$dryRun) {
            Write-Host "Applying: $file" -ForegroundColor Cyan
            kubectl apply -f $file
        } else {
            Write-Host "Would apply: $file" -ForegroundColor Gray
        }
    }
}

# Wait for deployment to complete
if (!$dryRun) {
    Write-Host ""
    Write-Host "⏳ Waiting for deployment to complete..." -ForegroundColor Yellow
    kubectl rollout status deployment/rag-backend -n rag-system --timeout=300s
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Deployment failed or timed out!" -ForegroundColor Red
        exit 1
    }
}

# Verify deployment
Write-Host ""
Write-Host "🔍 Verifying deployment..." -ForegroundColor Yellow

if (!$dryRun) {
    # Check pod status
    kubectl get pods -n rag-system -l app=rag-backend
    
    # Check service status
    kubectl get services -n rag-system
    
    # Run health check
    Write-Host ""
    Write-Host "🏥 Running health check..." -ForegroundColor Yellow
    
    $healthCheckUrl = if ($environment -eq "production") {
        "https://api.yourdomain.com/health"
    } else {
        "https://staging-api.yourdomain.com/health"
    }
    
    try {
        $response = Invoke-RestMethod -Uri $healthCheckUrl -Method Get -TimeoutSec 30
        if ($response.status -eq "healthy") {
            Write-Host "✅ Health check passed!" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Health check returned: $($response.status)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Health check failed: $_" -ForegroundColor Red
    }
}

# Deployment summary
Write-Host ""
Write-Host "📋 Deployment Summary" -ForegroundColor Green
Write-Host "=" * 30
Write-Host "Environment: $environment" -ForegroundColor Cyan
Write-Host "Image: $imageTag" -ForegroundColor Cyan
Write-Host "Namespace: rag-system" -ForegroundColor Cyan
Write-Host "Deployment time: $(Get-Date)" -ForegroundColor Cyan

if ($dryRun) {
    Write-Host ""
    Write-Host "👀 This was a DRY RUN - no actual deployment occurred" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Deployment process completed!" -ForegroundColor Green
