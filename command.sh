# 权重链接地址:
链接：https://pan.baidu.com/s/1p70DoCV_-QH_GMCTxKvhhg?pwd=6n5y 
提取码：6n5y 


# train
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/swin/yz_fuqi.yaml --pretrained swin_tiny_patch4_window7_224_22k.pth --local_rank 0


# 多卡训练:
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/swin/yz_fuqi.yaml --pretrained swin_tiny_patch4_window7_224_22k.pth  --batch-size 16

# 评估代码
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin/yz_fuqi.yaml --resume='/home/apulis-test/userdata/code/mycode/Swin-Transformer-main/Swin-Transformer-main/output/yz_model/default/best_ckpt.pth'  --local_rank 0

# 推理代码
python infer_pt.py --cfg configs/swin/yz_fuqi.yaml   --pretrained output/yz_fuqi_1108/default/best_ckpt.pth --local_rank 0


# 环境:
Package                Version
---------------------- ---------------
addict                 2.4.0
albumentations         1.1.0
aliyun-python-sdk-core 2.16.0
aliyun-python-sdk-kms  2.16.5
attrs                  24.2.0
certifi                2021.5.30
cffi                   1.17.1
charset-normalizer     3.4.0
click                  8.1.7
codecov                2.1.13
colorama               0.4.6
coloredlogs            15.0.1
coverage               7.6.1
crcmod                 1.7
cryptography           43.0.3
cycler                 0.10.0
Cython                 0.29.24
efficientnet-pytorch   0.7.1
exceptiongroup         1.2.2
filelock               3.14.0
flake8                 7.1.1
flatbuffers            2.0
humanfriendly          10.0
idna                   3.10
imageio                2.13.1
importlib-metadata     8.5.0
iniconfig              2.0.0
instaboostfast         0.1.2
interrogate            1.7.0
isort                  4.3.21
jmespath               0.10.0
joblib                 1.1.0
kiwisolver             1.3.2
Markdown               3.7
markdown-it-py         3.0.0
matplotlib             3.4.3
mccabe                 0.7.0
mdurl                  0.1.2
mkl-fft                1.3.0
mkl-random             1.2.2
mkl-service            2.4.0
mmcv-full              1.4.0
mmdet                  3.3.0
model-index            0.1.11
modelindex             0.0.2
mpmath                 1.3.0
networkx               2.6.3
numpy                  1.22.3
olefile                0.46
onnx                   1.11.0
onnxruntime-gpu        1.16.3
opencv-python          4.4.0.46
opencv-python-headless 4.5.4.60
opendatalab            0.0.10
openmim                0.3.9
openxlab               0.1.2
ordered-set            4.1.0
oss2                   2.17.0
packaging              24.1
pandas                 2.0.3
Pillow                 8.4.0
pip                    21.0.1
pluggy                 1.5.0
protobuf               3.20.1
py                     1.11.0
pycocotools            2.0.3
pycodestyle            2.12.1
pycparser              2.22
pycryptodome           3.21.0
pyflakes               3.2.0
pygments               2.18.0
pyparsing              2.4.7
pytest                 8.3.3
python-dateutil        2.8.2
pytz                   2023.4
PyWavelets             1.2.0
PyYAML                 6.0
qudida                 0.0.4
requests               2.28.2
rich                   13.4.2
scikit-image           0.18.3
scikit-learn           1.0.1
scipy                  1.7.3
setuptools             60.2.0
shapely                2.0.6
six                    1.16.0
some-package           0.1
sympy                  1.13.3
tabulate               0.9.0
termcolor              1.1.0
terminaltables         3.1.0
threadpoolctl          3.0.0
tifffile               2021.11.2
timm                   0.4.12
tomli                  2.0.2
torch                  1.8.0
torchaudio             0.8.0a0+a751e1d
torchvision            0.9.0
tqdm                   4.65.2
typing-extensions      4.12.2
tzdata                 2024.2
urllib3                1.26.20
wheel                  0.37.0
xdoctest               1.2.0
yacs                   0.1.8
yapf                   0.31.0
zipp                   3.20.2


