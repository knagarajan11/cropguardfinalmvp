
import os

import sys

import subprocess

import time



api_key = input("Please enter your NVIDIA API Key (starts with nvapi-) to enable Guardrails: ").strip()

if api_key:

    os.environ["NVIDIA_API_KEY"] = api_key



print("[1/2] Starting FastAPI Backend...")

fastapi_process = subprocess.Popen(

    [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8002"]

)



print("Waiting 10 seconds for backend to initialize...")

time.sleep(10)



print("[2/2] Starting Streamlit UI...")

streamlit_process = subprocess.Popen(

    [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8599"]

)



try:

    fastapi_process.wait()

    streamlit_process.wait()

except KeyboardInterrupt:

    fastapi_process.terminate()

    streamlit_process.terminate()

