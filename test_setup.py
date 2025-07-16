#!/usr/bin/env python3
"""
Comprehensive installation test for Particle Accelerator Simulation Project
Tests all required packages and provides detailed feedback
"""

import sys
import subprocess

def test_import(package_name, import_statement=None, version_attr=None):
    """Test importing a package and optionally get its version"""
    if import_statement is None:
        import_statement = package_name
    
    try:
        # Execute the import
        exec(f"import {import_statement}")
        
        # Get version if specified
        if version_attr:
            try:
                version = eval(f"{import_statement}.{version_attr}")
                print(f"✅ {package_name}: {version}")
            except:
                print(f"✅ {package_name}: Available (version unknown)")
        else:
            print(f"✅ {package_name}: Available")
        return True
        
    except ImportError as e:
        print(f"❌ {package_name}: NOT AVAILABLE - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name}: Available but error getting version - {e}")
        return True

def test_physics_functionality():
    """Test basic physics functionality"""
    print("\n🔬 Testing Physics Functionality:")
    
    try:
        import ROOT
        # Test basic ROOT functionality
        h = ROOT.TH1F("test", "Test Histogram", 100, 0, 10)
        print("✅ ROOT: Basic histogram creation works")
    except Exception as e:
        print(f"❌ ROOT: Basic functionality failed - {e}")
    
    try:
        import pythia8
        # Test basic PYTHIA functionality
        pythia = pythia8.Pythia()
        print("✅ PYTHIA8: Basic initialization works")
    except Exception as e:
        print(f"❌ PYTHIA8: Basic functionality failed - {e}")

def test_ai_functionality():
    """Test basic AI functionality"""
    print("\n🤖 Testing AI/ML Functionality:")
    
    try:
        import torch
        # Test tensor creation
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✅ PyTorch: Tensor creation works, CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch: Basic functionality failed - {e}")

def main():
    print("🚀 Testing Particle Accelerator Simulation Setup")
    print("=" * 60)
    
    print(f"🐍 Python Version: {sys.version}")
    print("=" * 60)
    
    # Core Python packages
    print("\n📦 Core Scientific Packages:")
    test_import("NumPy", "numpy", "__version__")
    test_import("SciPy", "scipy", "__version__")
    test_import("Pandas", "pandas", "__version__")
    test_import("Matplotlib", "matplotlib", "__version__")
    test_import("Seaborn", "seaborn", "__version__")
    test_import("Plotly", "plotly", "__version__")
    test_import("Scikit-learn", "sklearn", "__version__")
    
    # ML Frameworks
    print("\n🧠 Machine Learning Frameworks:")
    test_import("PyTorch", "torch", "__version__")
    test_import("TorchVision", "torchvision", "__version__")
    test_import("TorchAudio", "torchaudio", "__version__")
    
    # Physics packages
    print("\n⚛️  Physics Packages:")
    test_import("ROOT")
    test_import("PYTHIA8", "pythia8")
    test_import("Uproot", "uproot", "__version__")
    test_import("Awkward", "awkward", "__version__")
    test_import("Particle", "particle", "__version__")
    test_import("HEP Units", "hepunits")
    
    # AI Enhancement packages
    print("\n🔍 AI Enhancement Libraries:")
    test_import("SHAP", "shap", "__version__")
    test_import("LIME", "lime")
    test_import("Transformers", "transformers", "__version__")
    
    # Web Development
    print("\n🌐 Web Development:")
    test_import("Flask", "flask", "__version__")
    test_import("FastAPI", "fastapi", "__version__")
    test_import("Uvicorn", "uvicorn", "__version__")
    test_import("WebSockets", "websockets", "__version__")
    
    # Development tools
    print("\n🔧 Development Tools:")
    test_import("Jupyter", "jupyter")
    test_import("IPython", "IPython", "__version__")
    test_import("Requests", "requests", "__version__")
    test_import("Pillow", "PIL", "__version__")
    
    # Test functionality
    test_physics_functionality()
    test_ai_functionality()
    
    print("\n" + "=" * 60)
    print("🎯 Installation test complete!")
    print("=" * 60)
    
    # Final summary
    print("\n📋 Next Steps:")
    print("✅ If all packages show as available, you're ready to start Phase 1!")
    print("❌ If any packages are missing, install them with:")
    print("   pip install [package-name]")
    print("   conda install -c conda-forge [package-name]")

if __name__ == "__main__":
    main()