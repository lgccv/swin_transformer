
import os
import argparse
from torch.autograd import Variable
import cv2

import torch
from torchvision import transforms

from config import get_config
from models import build_model
from PIL import Image
import onnx
import onnxruntime as ort
import numpy as np
import time


from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR
        
    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp

except:
    from timm.data.transforms import _pil_interp


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer inference script',add_help=False)
    parser.add_argument('--cfg',type=str,required=True,metavar='FILE',help='path to config file')
    parser.add_argument("--opts",help = "Modify config options by adding 'KEY VALUE' pairs.",default=None,nargs='+')

    # easy config modification
    parser.add_argument('--batch-size',type=int,help="batch size for single GPU")
    parser.add_argument('--data-path',type=str,help='path to dataset')
    parser.add_argument('--zip',action='store_true',help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode',type=str,default='part',choices=['no','full','part'],help='no:no cache;full:cache all data;part:sharding the dataset into nonoverlapping piecs and only cache one piece')
    parser.add_argument('--pretrained',help='pretrained weight from checkpoint,could be imagenet22k pretrained weight')
    parser.add_argument('--resume',help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

        # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')


    args,unparsed = parser.parse_known_args()
    config = get_config(args)
    return args,config

def pytorch2onnx(model,img_shape=448):
    x = torch.randn(1,3,img_shape,img_shape).to(DEVICE)
    torch.onnx.export(model,x,"swin_transformer.onnx",opset_version =11,input_names=['input'],output_names=['output'])
    onnx_model = onnx.load("swin_transformer.onnx")
    onnx.checker.check_model(onnx_model)
    print("onnx export successfully!")


def infer_by_onnx(image_path =r''):
    # onnx 推理
    session_options = ort.SessionOptions()
    providers = ['CUDAExecutionProvider','CPUExecutionProvider']
    options = [{}]
    sess = ort.InferenceSession("swin_transformer.onnx", providers=providers)
    io_binding = sess.io_binding()
    output_names = [_.name for _ in sess.get_outputs()]
    for name in output_names:
        io_binding.bind_output(name)
    
    testList = os.listdir(image_path)
    for file in testList:
        img = Image.open(os.path.join(image_path,file)).convert('RGB')
        img = transforms_test(img)
        img.unsqueeze_(0)
        img = Variable(img).to(DEVICE)
        io_binding.bind_input(name='input',device_type="cuda",device_id=0,element_type=np.float32,shape=(1,3,448,448),buffer_ptr=img.data_ptr())
        sess.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()


def infer_by_pth(model,image_path =r''):
    testList = os.listdir(image_path)
    ok = 0
    fuqi = 0

    for file in testList:
        img = Image.open(os.path.join(image_path,file)).convert('RGB')
        img = transforms_test(img)
        img.unsqueeze_(0)
        img = Variable(img).to(DEVICE)
        out = model(img)
        print(f'out:{out}')
        probabilities = torch.softmax(out,dim=1)
        print(f'probabilities:{probabilities}')
        _,pred = torch.max(out.data,1)
        ori_img = cv2.imread(os.path.join(image_path,file))
        if classes[pred.data.item()] == 'fuqi':
            fuqi = fuqi + 1
            cv2.imwrite(os.path.join(r'/home/apulis-test/userdata/code/mycode/Swin-Transformer-main/Swin-Transformer-main/result/fuqi/'+file),ori_img)
        if classes[pred.data.item()] == 'ok':
            ok = ok + 1
            cv2.imwrite(os.path.join(r'/home/apulis-test/userdata/code/mycode/Swin-Transformer-main/Swin-Transformer-main/result/ok/'+file),ori_img)
        print(f'fuqi:{fuqi}')
        print(f'ok:{ok}')
        text = 'ImageName:{},predict:{},item:{}'.format(file,classes[pred.data.item()],pred.data.item())
        print(f"result:{text}")

# python infer_pt.py --cfg configs/swin/yz_fuqi.yaml   --pretrained output/yz_fuqi_1108/default/best_ckpt.pth --local_rank 0

if __name__=='__main__':
    args,config = parse_option()

    transforms_test = transforms.Compose([transforms.Resize((config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),interpolation=_pil_interp(config.DATA.INTERPOLATION)),\
                                          transforms.ToTensor(),\
                                          transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)
                                          ])
    classes = config.DATA.NAME_CLASSES
    classes.sort()
    print(f'classes:{classes}')
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    checkpoint = torch.load(config.MODEL.PRETRAINED,map_location='cpu')
    model.load_state_dict(checkpoint['model'],strict=False)
    model.eval()
    model.to(DEVICE)

    path = r'/home/apulis-test/teamdata/yz_dataset/fuqi/val/ok'

    infer_by_pth(model,path)
    # pytorch2onnx(model)
    # infer_by_onnx(path)


