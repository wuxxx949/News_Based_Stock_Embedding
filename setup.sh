# prepare env for sentence transformer
conda create --name st python=3.9
conda deactivate
conda activate st
pip install -U pip

# install pytorch for sentence transformers
pip3 install torch torchvision torchaudio
pip3 install click==8.1.6
pip3 install sentence-transformers
pip install fastparquet==2023.4.0
pip install pyarrow==12.0.1

# prepare a separate env for tensorflow as it cannot live with pytorch
conda deactivate
conda create --name tf python=3.9
conda activate tf
conda install -U pip
pip install -r requirements.txt
conda deactivate

