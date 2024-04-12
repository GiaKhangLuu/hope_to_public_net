pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

python3 -m pip install -e detectron2

scp -r -P 11277 giakhang@0.tcp.ap.ngrok.io:/media/giakhang/146096A860968FDA/autopilot_giakhang/dataset/cityscapes2/cityscapes .

scp -r -P 11277 giakhang@0.tcp.ap.ngrok.io:/media/giakhang/146096A860968FDA/autopilot_giakhang/weights/huflit_net_20k_iters .

mkdir output

ln ./huflit_net_20k_iters/last_checkpoint ./output
ln ./huflit_net_20k_iters/model_final.pth ./output

python3 train_huflitnet.py

scp -r -P 11277 ./output giakhang@0.tcp.ap.ngrok.io:/home/giakhang/Downloads