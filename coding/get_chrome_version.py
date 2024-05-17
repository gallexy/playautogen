# filename: get_chrome_version.py
import subprocess

# Get Chrome version
result = subprocess.run(['chromium-browser', '--version'], stdout=subprocess.PIPE)
version_output = result.stdout.decode('utf-8').strip()
version = version_output.split(' ')[1]

print(f"Chrome Version: {version}")