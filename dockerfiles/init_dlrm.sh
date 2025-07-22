cd "$(dirname "$0")"
set -x
set -e

USER="$(whoami)"
PROJECT_PATH="$(cd .. && pwd)"
DLRM_PATH="${PROJECT_PATH}/model_zoo/torchrec_dlrm"
MIRROR_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/"


sudo apt install python3.10-venv -y

cd ${DLRM_PATH}
python3 -m venv dlrm_venv
source ${DLRM_PATH}/dlrm_venv/bin/activate
pip install --upgrade pip

pip install numpy==1.24.4 -f ${MIRROR_URL}
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install fbgemm-gpu==0.5.0 tqdm torchmetrics scikit-learn pandas matplotlib torchx -f ${MIRROR_URL}

cd ${PROJECT_PATH}/third_party/torchrec
pip install -e .

# source /home/${USER}/.bashrc