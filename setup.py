# Advanced Chatbot - Setup Script

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_environment():
    """Set up environment file"""
    print("🔧 Setting up environment...")
    if not os.path.exists(".env"):
        if os.path.exists("env_example.txt"):
            with open("env_example.txt", "r") as src:
                with open(".env", "w") as dst:
                    dst.write(src.read())
            print("✅ Environment file created from template!")
        else:
            print("⚠️  No environment template found. Creating basic .env file...")
            with open(".env", "w") as f:
                f.write("# Basic environment configuration\n")
                f.write("MODEL_NAME=microsoft/DialoGPT-medium\n")
                f.write("MAX_LENGTH=100\n")
                f.write("TEMPERATURE=0.7\n")
    else:
        print("✅ Environment file already exists!")

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected!")
    return True

def main():
    """Main setup function"""
    print("🤖 Advanced Chatbot Setup")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    print("\n🎉 Setup complete!")
    print("\nTo run the chatbot:")
    print("  Web Interface: streamlit run app.py")
    print("  CLI Interface: python 0548.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
