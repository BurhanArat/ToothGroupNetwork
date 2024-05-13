# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install necessary tools including wget and git
RUN apt-get update && apt-get install -y wget git zip 


#RUN git clone https://github.com/BurhanArat/ToothGroupNetwork.git


# Install pip requirements
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh


# Set path to conda
ENV PATH /opt/conda/bin:$PATH

COPY environment.yaml .
RUN conda env create -f environment.yaml



# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "/bin/bash", "-c"]

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH

# Assume your application's entry point is setup in the repository
CMD ["conda", "run", "-n", "segmentation", "python", "segm_unittest.py"]
