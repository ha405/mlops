FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /code
RUN useradd -m -u 1000 appuser
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools
RUN python -m pip install --no-cache-dir --default-timeout=1000 --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
RUN chown -R appuser:appuser /code
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.serve:app", "--host", "0.0.0.0", "--port", "8000"]