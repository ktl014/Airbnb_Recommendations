FROM continuumio/miniconda3
MAINTAINER Kevin Le <kevin.le@gmail.com>

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# set the working directory for containers
WORKDIR /app

# Installing python dependencies
COPY env.yml .
RUN conda env create -f env.yml

# Make RUN commands use the new environment:
RUN ["conda","update", "-n","base", "-c", "defaults", "conda"]
SHELL ["conda", "run", "-n", "airbnb_env", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure streamlit is installed:"
RUN python -c "import streamlit"

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# Copy all the files from the project’s root to the working directory
COPY . /app
RUN ls -la /app/*

# Make port 8000 available to the world outside this container
EXPOSE 8501

# The code to run when container is started:
SHELL ["conda", "run", "-n", "airbnb_env", "/bin/bash", "-c"]
CMD [ "streamlit", "run", "src/app.py" ]