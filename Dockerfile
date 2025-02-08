# Usa uma imagem base do Python
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de dependências
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Expõe a porta 8000 (usada pela API)
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]