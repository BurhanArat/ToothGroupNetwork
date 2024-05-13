# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install necessary tools including wget and git
RUN apt-get update && apt-get install -y wget git zip libgl1-mesa-dev libgomp1 && rm -rf /var/lib/apt/lists/*



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
# Activate the Conda environment

# Install the package from the Git repository in editable mode
# Use Conda environment for subsequent commands
SHELL ["conda", "run", "-n", "segmentation", "/bin/bash", "-c"]

# Prepare directory and install the package from the Git repository in editable mode


CMD ["conda", "run", "-n", "segmentation", "pip", "install" ,"-e", "./external_libs/pointops/setup.py"]
# Make port 80 available to the world outside this container
EXPOSE 80


# Assume your application's entry point is setup in the repository
CMD ["conda", "run", "-n", "segmentation", "python", "-m ","unittest","unittest.py"]
