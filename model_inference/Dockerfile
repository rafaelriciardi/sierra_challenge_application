FROM python:3

# Allow statements and log messages to immediately appear in the native logs
ENV PYTHONUNBUFFERED True

ENV PORT 8000
EXPOSE 8000

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt

# Start uvicorn application, setting the host and port
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info