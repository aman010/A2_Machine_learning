FROM python:3.8.6-slim-buster

# set working directory in container
WORKDIR /usr/src/app

# Copy and install packages
COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

# Copy app folder to app folder in container
COPY ./app/ /usr/src/app

COPY app/* /app/

EXPOSE 8080


# Changing to non-root user
#RUN useradd -m appUser
#USER appUser

ENV NGINX_WORKER_PROCESSES auto

# Run locally on port 8050
#CMD gunicorn --bind 0.0.0.0:8089 app:serve
CMD ["python", "app.py"]
