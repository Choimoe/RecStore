set -x s
cd "$(dirname "$0")"


# sudo docker build -f Dockerfile.recstore --build-arg uid=$UID  -t recstore .


RECSTORE_PATH="$(cd ".." && pwd)"
DATASET_PATH="/home/xieminhui/FrugalDataset"
DGL_DATASET_PATH="/home/xieminhui/dgl-data"

sudo docker run --cap-add=SYS_ADMIN --privileged --security-opt seccomp=unconfined  \
--name recstore --net=host \
-v ${RECSTORE_PATH}:${RECSTORE_PATH} \
-v /dev/shm:/dev/shm \
-v /dev/hugepages:/dev/hugepages \
-v ${DATASET_PATH}:${DATASET_PATH} \
-v ${DGL_DATASET_PATH}:${DGL_DATASET_PATH} \
-v /dev:/dev -v /nas:/nas \
-w ${RECSTORE_PATH} --rm -it --gpus all -d recstore


# sudo docker exec -it recstore /bin/bash
