apt install unzip

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

#git clone https://github.com/GiaKhangLuu/hope_to_public_net.git

#cd hope_to_public_net
#git checkout attention

python3 -m pip install -e detectron2

python3 -m pip install opencv_python

python3 -m pip install gdown

gdown 1IpxCZH7gKZpDf48sh_S52-l6Tbl6Hw0-
gdown 1VeqhcRm4aZYiS4bWOqLtmq0pQNAkxZHR

unzip huflit_net_150k_iters.zip
unzip coco2017.zip

mkdir output

# Using `cp` instead of `ln`. The os may not detect symbolic files
cp ./huflit_net_150k_iters/last_checkpoint ./output
cp ./huflit_net_150k_iters/model_final.pth ./output

#screen -S train_first_10k

#python3 train_huflitnet.py
