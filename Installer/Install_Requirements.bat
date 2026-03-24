winget install -e --id Python.Python.3.12 --silent --accept-source-agreements --accept-package-agreements
cd "%1"
pip install -r requirements.txt