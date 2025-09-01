# RecStore

RecStore is a high-performance parameter storage system designed to meet the growing demands of recommendation models in modern AI data centers. Unlike NLP and vision models, where computation dominates, recommendation models are bottlenecked by memory due to massive, trillion-scale sparse embedding parameters. RecStore addresses this challenge by abstracting heterogeneous, networked memory as a unified embedding pool. It provides key functionalities—parameter indexing, memory management, near-memory computation, and communication—as modular components within the system.

RecStore is:
- Universal: Easily integrated into existing DL frameworks via minimal OP implementations.
- Efficient: Optimized for the unique access patterns of recommendation models, leveraging GPU and NIC hardware acceleration.
- Cost-effective: Utilizes low-cost storage (e.g., SSDs, persistent memory) to expand memory capacity and reduce large model serving costs.




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




## Cite
We would appreciate citations to the following papers:

	@inproceedings{xie2025frugal,
		title={Frugal: Efficient and Economic Embedding Model Training with Commodity GPUs},
		author={Xie, Minhui and Zeng, Shaoxun and Guo, Hao and Gao, Shiwei and Lu, Youyou},
		booktitle={Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1},
		pages={509--523},
		year={2025}
	}

	@inproceedings{fan2024maxembed,
		title={MaxEmbed: Maximizing SSD Bandwidth Utilization for Huge Embedding Models Serving},
		author={Fan, Ruwen and Xie, Minhui and Jiang, Haodi and Lu, Youyou},
		booktitle={The 29th Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS'24)},
		year={2024}
	}


	@article{xie2023petps,
		title={PetPS: supporting huge embedding models with persistent memory},
		author={Xie, Minhui and Lu, Youyou and Wang, Qing and Feng, Yangyang and Liu, Jiaqiang and Ren, Kai and Shu, Jiwu},
		journal={Proceedings of the VLDB Endowment},
		volume={16},
		number={5},
		pages={1013--1022},
		year={2023},
		publisher={VLDB Endowment}
	}


	@inproceedings{xie2022fleche,
		title={Fleche: an efficient gpu embedding cache for personalized recommendations},
		author={Xie, Minhui and Lu, Youyou and Lin, Jiazhen and Wang, Qing and Gao, Jian and Ren, Kai and Shu, Jiwu},
		booktitle={Proceedings of the Seventeenth European Conference on Computer Systems},
		pages={402--416},
		year={2022}
	}

	@inproceedings{xie2020kraken,
		title={Kraken: memory-efficient continual learning for large-scale real-time recommendations},
		author={Xie, Minhui and Ren, Kai and Lu, Youyou and Yang, Guangxu and Xu, Qingxing and Wu, Bihai and Lin, Jiazhen and Ao, Hongbo and Xu, Wanhong and Shu, Jiwu},
		booktitle={SC20: International Conference for High Performance Computing, Networking, Storage and Analysis},
		pages={1--17},
		year={2020},
		organization={IEEE}
	}