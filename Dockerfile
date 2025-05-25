# Use Python 3.11 slim como base
FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p API/Features/Artefatos API/Modelo/Artefatos mlruns

# Expor portas
EXPOSE 8000 5000

# Comando padrão (pode ser sobrescrito)
CMD ["python", "API/main.py"] 