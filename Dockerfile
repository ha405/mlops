FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.serve:app", "--host", "0.0.0.0", "--port", "8000"]

RUN chown -R appuser:appuser /code

USER appuser

EXPOSE 8000

# Use 'python -m uvicorn' to ensure the module is loaded from the correct path
CMD ["python", "-m", "uvicorn", "app.serve:app", "--host", "0.0.0.0", "--port", "8000"]