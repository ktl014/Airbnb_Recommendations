FROM continuumio/miniconda3
MAINTAINER Kevin Le <kevin.le@gmail.com>

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# set the working directory for containers
WORKDIR /usr/src/app

# Installing python dependencies
COPY env.yml .

# Make RUN commands use the new environment:
RUN ["conda","update", "-n","base", "-c", "defaults", "conda"]
RUN [ "conda", "env", "update" ,"-n","base","-f","env.yml"]

# Copy all the files from the project’s root to the working directory
COPY . .
RUN ls -la ./

ENTRYPOINT ["streamlit", "run"]
CMD ["src/navigation_app.py"]