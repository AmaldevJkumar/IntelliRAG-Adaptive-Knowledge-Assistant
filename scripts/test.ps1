# RAG Knowledge Assistant - Test Runner
Write-Host "🧪 RAG Knowledge Assistant - Test Suite" -ForegroundColor Green
Write-Host "=" * 50

# Check if virtual environment exists
if (!(Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found. Run setup first:" -ForegroundColor Red
    Write-Host "   .\scripts\setup.ps1" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

# Set test environment variables
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
$env:ENVIRONMENT = "testing"
$env:DEBUG = "true"

Write-Host "📊 Running comprehensive test suite..." -ForegroundColor Yellow
Write-Host ""

# Parse command line arguments
$testType = $args[0]
$coverage = $args -contains "--coverage" -or $args -contains "-c"
$verbose = $args -contains "--verbose" -or $args -contains "-v"

# Build pytest command
$pytestArgs = @()

if ($verbose) {
    $pytestArgs += "-v"
} else {
    $pytestArgs += "-q"
}

if ($coverage) {
    $pytestArgs += "--cov=backend/app"
    $pytestArgs += "--cov-report=html"
    $pytestArgs += "--cov-report=term-missing"
    Write-Host "📈 Coverage reporting enabled" -ForegroundColor Green
}

# Determine which tests to run
switch ($testType) {
    "unit" {
        Write-Host "🔬 Running unit tests only..." -ForegroundColor Cyan
        $pytestArgs += "tests/test_unit/"
    }
    "integration" {
        Write-Host "🔗 Running integration tests only..." -ForegroundColor Cyan
        $pytestArgs += "tests/test_integration/"
    }
    "api" {
        Write-Host "🌐 Running API tests only..." -ForegroundColor Cyan
        $pytestArgs += "tests/test_api.py"
    }
    "fast" {
        Write-Host "⚡ Running fast tests only..." -ForegroundColor Cyan
        $pytestArgs += "-m", "not slow"
    }
    "slow" {
        Write-Host "🐌 Running slow tests only..." -ForegroundColor Cyan
        $pytestArgs += "-m", "slow"
    }
    default {
        Write-Host "🎯 Running all tests..." -ForegroundColor Cyan
        $pytestArgs += "tests/"
    }
}

# Add pytest-html for nice reports
$pytestArgs += "--html=test-results.html"
$pytestArgs += "--self-contained-html"

Write-Host "🏃 Executing: pytest $($pytestArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Run tests
try {
    $testStart = Get-Date
    pytest @pytestArgs
    $testEnd = Get-Date
    $duration = $testEnd - $testStart
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ All tests passed!" -ForegroundColor Green
        Write-Host "⏱️  Duration: $($duration.TotalSeconds.ToString('F2')) seconds" -ForegroundColor Gray
        
        if ($coverage) {
            Write-Host "📊 Coverage report generated: htmlcov/index.html" -ForegroundColor Cyan
        }
        
        Write-Host "📄 Test report: test-results.html" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "❌ Some tests failed!" -ForegroundColor Red
        Write-Host "📄 Check test-results.html for detailed results" -ForegroundColor Yellow
        
        if ($coverage) {
            Write-Host "📊 Coverage report: htmlcov/index.html" -ForegroundColor Cyan
        }
    }
} catch {
    Write-Host ""
    Write-Host "💥 Test execution failed!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🔧 Test commands:" -ForegroundColor Yellow
Write-Host "   All tests: .\scripts\test.ps1" -ForegroundColor Cyan
Write-Host "   Unit tests: .\scripts\test.ps1 unit" -ForegroundColor Cyan
Write-Host "   Integration: .\scripts\test.ps1 integration" -ForegroundColor Cyan
Write-Host "   API tests: .\scripts\test.ps1 api" -ForegroundColor Cyan
Write-Host "   With coverage: .\scripts\test.ps1 --coverage" -ForegroundColor Cyan
Write-Host "   Fast tests: .\scripts\test.ps1 fast" -ForegroundColor Cyan
Write-Host ""

exit $LASTEXITCODE
