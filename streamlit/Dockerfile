FROM python:3

# Allow statements and log messages to immediately appear in the native logs
ENV PYTHONUNBUFFERED True

EXPOSE 8080

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt

# Start streamlit application, setting the port
CMD streamlit run --server.port 8080 --server.enableCORS false app.py