# syntax = docker/dockerfile:experimental
FROM python:3.11-slim

WORKDIR /home/app

COPY app.py requirements.txt /home/app/

RUN mkdir templates static

COPY templates/index.html /home/app/templates
COPY static/image.png /home/app/static
# COPY . /tmp/src
# need the below packages for opencv library
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
                                             apt-get install build-essential cmake ffmpeg libsm6 libxext6 -y && \
                                             rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements.txt

ENTRYPOINT ["python", "app.py"]

# FROM python:3.11-slim

# WORKDIR /home/app

# COPY app.py requirements.txt /home/app/

# RUN mkdir templates

# COPY templates/index.html /home/app/templates

# # need the below packages for opencv library
# RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
#                                              apt-get install ffmpeg libsm6 libxext6 -y && \
#                                              rm -rf /var/lib/apt/lists/*

# # RUN pip install --no-cache-dir -r requirements.txt

# # Below cache mount results in checksum error on aws, so commenting it for aws build
# # RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements.txt

# ENTRYPOINT ["python", "app.py"]
