apt install unzip

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/GiaKhangLuu/hope_to_public_net.git

cd hope_to_public_net

python3 -m pip install -e detectron2

python3 -m pip install opencv_python

python3 -m pip install gdown

gdown 18F-uHt0MH5iSy2e7SMKRGvFcCbfuCKxA
gdown 1Mqz_cIt8KCplKA-cYkCXoCEjoYk-MOPw

unzip huflit_net_20k_iters.zip
unzip cityscapes2.zip

mkdir output

# Using `cp` instead of `ln`. The os may not detect symbolic files
cp ./huflit_net_20k_iters/last_checkpoint ./output
cp ./huflit_net_20k_iters/model_final.pth ./output

python3 train_huflitnet.py
