ssh-keygen -t ed25519 -f ~/.ssh/vast_ai_key -N ""
python.exe -m pip install --upgrade pip
pip install vastai

vastai search offers 'gpu_name=RTX_4090 reliability>0.98 num_gpus=1 inet_down>400 disk_space>60' --order 'dph_total'

vastai create instance 1234567 \
    --image pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel \
    --disk 60 \
    --env "-e HF_TOKEN=hf_xxxxxx" \
    --onstart-cmd "curl -sL https://gist.githubusercontent.com/LINK_TO_YOUR_RAW_SCRIPT.sh | bash" \
    --ssh --direct