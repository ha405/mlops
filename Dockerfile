FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Force the path to include the local site-packages
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages:/code

WORKDIR /code

RUN groupadd -r appuser && useradd -r -g appuser appuser

# 1. Upgrade pip using the module approach
RUN python -m pip install --upgrade pip

# 2. Install setuptools and mlflow using 'python -m pip' 
# This ensures they land in the exact same path python uses
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /code

USER appuser

EXPOSE 8000

# Use 'python -m uvicorn' to ensure the module is loaded from the correct path
CMD ["python", "-m", "uvicorn", "app.serve:app", "--host", "0.0.0.0", "--port", "8000"]