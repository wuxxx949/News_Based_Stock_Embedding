# prepare env for sentence transformer
conda create --name st python=3.9
conda deactivate
conda activate st
pip install -U pip
pip3 install torch torchvision torchaudio

