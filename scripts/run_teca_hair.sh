source /is/cluster/hzhang/.bashrc
source /is/cluster/hzhang/anaconda3/etc/profile.d/conda.sh

conda activate latent-nerf
cd /home/hzhang/nerf_face/TECA

# TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
/home/hzhang/anaconda3/envs/latent-nerf/bin/python "${1}" "${2}" "${3}" "${4}" "${5}"