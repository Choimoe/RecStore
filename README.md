## Environment Setup

We provide a Dockerfile to simplify the environment setup. Please install `docker` and `nvidia-docker` first by checking [URL](https://docs.docker.com/engine/install/ubuntu/) and [URL2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

In ubuntu, simply:

	curl -fsSL https://get.docker.com -o get-docker.sh
	sudo sh ./get-docker.sh
	curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  	&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
	sudo apt-get update
	sudo apt-get install -y nvidia-container-toolkit

After installing docker and nvidia-docker, build the Docker image by running the following command in the `dockerfiles` directory:

	cd dockerfiles
	sudo docker build -f Dockerfile.recstore --build-arg uid=$UID  -t recstore .
	cd -

And then start this container, by running the following commands. **Please modify corresponding pathes below**.

	sudo docker run --cap-add=SYS_ADMIN --privileged --security-opt seccomp=unconfined --runtime=nvidia --name recstore --net=host -v /home/xieminhui/RecStore:/home/xieminhui/RecStore  -v /dev/shm:/dev/shm -v /dev/hugepages:/dev/hugepages -v /home/xieminhui/FrugalDataset:/home/xieminhui/FrugalDataset -v /home/xieminhui/dgl-data:/home/xieminhui/dgl-data -v /dev:/dev -w /home/xieminhui/RecStore --rm -it --gpus all -d recStore

or 
	
	cd dockerfiles && bash start_docker.sh && cd -

Enter the container.

	sudo docker exec -it recstore /bin/bash

**We provide a script for one-click environment initialization**. Simply run the following command **in the docker** to set up the environment:

	(inside docker) cd dockerfiles
	(inside docker) bash init_env_inside_docker.sh


## Build RecStore

	(inside docker) mkdir build
	(inside docker) cd build
	(inside docker) cmake .. -DCMAKE_BUILD_TYPE=Release
