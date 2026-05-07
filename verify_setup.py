"""
Setup verification script
Checks if all components are properly configured
"""
import sys
from pathlib import Path
import importlib


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'statsmodels',
        'prophet',
        'xgboost',
        'tensorflow',
        'fastapi',
        'uvicorn',
        'pydantic',
        'openpyxl',
        'matplotlib',
        'seaborn',
        'joblib',
        'holidays'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_files():
    """Check if all required files exist"""
    print("\nChecking project files...")
    
    required_files = [
        'config.py',
        'data_loader.py',
        'feature_engineering.py',
        'models.py',
        'model_trainer.py',
        'api.py',
        'train.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'Forecasting Case- Study.xlsx'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            missing.append(file)
    
    if missing:
        print(f"\n  Missing files: {', '.join(missing)}")
        return False
    
    return True


def check_directories():
    """Check if all required directories exist"""
    print("\nChecking directories...")
    
    required_dirs = ['data', 'models', 'logs', 'results']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            dir_path.mkdir(exist_ok=True)
            print(f"    Created {dir_name}/")
    
    return True


def check_data_file():
    """Check if data file exists and is readable"""
    print("\nChecking data file...")
    
    data_file = Path("Forecasting Case- Study.xlsx")
    
    if not data_file.exists():
        print(f"  ✗ Data file not found: {data_file}")
        print("    Please ensure the Excel file is in the project directory")
        return False
    
    try:
        import pandas as pd
        df = pd.read_excel(data_file)
        print(f"  ✓ Data file readable")
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {df.columns.tolist()}")
        return True
    except Exception as e:
        print(f"  ✗ Error reading data file: {e}")
        return False


def check_imports():
    """Check if all project modules can be imported"""
    print("\nChecking project modules...")
    
    modules = [
        'config',
        'data_loader',
        'feature_engineering',
        'models',
        'model_trainer',
        'api'
    ]
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}.py")
        except Exception as e:
            print(f"  ✗ {module}.py - Error: {e}")
            return False
    
    return True


def print_summary(checks):
    """Print summary of checks"""
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    total = len(checks)
    passed = sum(checks.values())
    
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {check_name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Run training: python train.py")
        print("  2. Start API: python api.py")
        print("  3. Test API: python test_api.py")
        print("  4. Or run complete pipeline: python run_pipeline.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Ensure data file is present")
        print("  - Check Python version (3.8+ required)")


def main():
    """Run all verification checks"""
    print("="*70)
    print("SALES FORECASTING SYSTEM - SETUP VERIFICATION")
    print("="*70)
    
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Project Files": check_files(),
        "Directories": check_directories(),
        "Data File": check_data_file(),
        "Module Imports": check_imports()
    }
    
    print_summary(checks)


if __name__ == "__main__":
    main()
