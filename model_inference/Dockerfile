FROM python:3

# Permitir que mensagens sejam enviadas imediatamente aos logs
ENV PYTHONUNBUFFERED True

# Define a variável de ambiente para a porta (caso deseje forçar 8000; o Cloud Run geralmente define PORT automaticamente)
ENV PORT 8000

EXPOSE 8000

# Copiar o código para dentro do container
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Instalar as dependências necessárias
RUN pip install -r requirements.txt

# Inicializa a aplicação com uvicorn, definindo explicitamente host e porta
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info