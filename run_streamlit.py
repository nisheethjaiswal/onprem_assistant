import os
import subprocess
import sys

def run_streamlit_app():
    """
    Launches the Streamlit app programmatically.
    """
    try:
        # Get the current working directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Name of your Streamlit app script
        streamlit_app_file = "rag_chatbot.py"  # Change this to your app filename
        
        # Construct the command to run the Streamlit app
        command = [sys.executable, "-m", "streamlit", "run", os.path.join(current_dir, streamlit_app_file)]
        
        # Execute the command
        subprocess.run(command, check=True)
    except Exception as e:
        print(f"Error running Streamlit app: {e}")

if __name__ == "__main__":
    run_streamlit_app()
