ssh-keygen -t ed25519 -f ~/.ssh/vast_ai_key -N ""
python.exe -m pip install --upgrade pip
pip install vastai

vastai search offers 'gpu_name=RTX_4090 reliability>0.98 num_gpus=1 inet_down>400 disk_space>60' --order 'dph_total'