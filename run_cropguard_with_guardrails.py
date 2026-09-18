import os
import sys
import subprocess
import time

def main():
    print("🌱 Starting CropGuard Full MVP (with NeMo Guardrails) 🌱\n")
    
    # 1. Ask for API Key securely
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        api_key = input("Please enter your NVIDIA API Key (starts with nvapi-) to enable Guardrails: ").strip()
        if not api_key:
            print("Error: API Key is required. Exiting.")
            sys.exit(1)
        os.environ["NVIDIA_API_KEY"] = api_key
    
    # Configure fastembed cache so it doesn't redownload
    cache_dir = os.path.join(os.path.dirname(__file__), "hf_cache")
    os.environ["FASTEMBED_CACHE_PATH"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    print("\n[1/2] Starting FastAPI Backend...")
    # Start FastAPI backend
    fastapi_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8002"],
        env=os.environ.copy()
    )
    
    # Wait for it to spin up
    time.sleep(5)
    
    print("\n[2/2] Starting Streamlit UI...")
    # Start Streamlit UI
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port", "8599"],
        [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8599"],
        env=os.environ.copy()
    )
    
    try:
        fastapi_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down CropGuard...")
        fastapi_process.terminate()
        streamlit_process.terminate()

if __name__ == "__main__":
    main()

