# Dockerfile para predictor custom de Vertex AI
# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    WORKERS=2

# Directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos necesarios
COPY predictor.py .
COPY achilles_moe_export/ ./achilles_moe_export/

# Exponer puerto
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Comando de inicio con gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "--timeout", "120", "predictor:app"]
