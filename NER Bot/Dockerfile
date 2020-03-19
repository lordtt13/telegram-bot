FROM python:3.7-stretch
MAINTAINER Tanmay Thakur <tanmaythakurbrn2rule@gmail.com>

# Install Build Utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# Check Python Environment
RUN python --version
RUN pip --version

# set the working directory for containers
WORKDIR  /usr/src

# Installing Dependencies
COPY requirements.txt .
RUN pip install python-telegram-bot
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copying Files
COPY . .

# Running Python Application
CMD ["python", "ner_bot.py"] 
