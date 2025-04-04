# Use the official miniconda3 base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        gdal-bin \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies via conda
RUN conda update -y -n base -c defaults conda
RUN conda install -y -c conda-forge libgdal gdal numpy scipy matplotlib --solver classic

COPY . .

RUN python setup.py install

ENTRYPOINT ["arrnorm", "-h"]