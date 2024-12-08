import argparse
from utils.data_utils import get_loader
from medical.trainer import Trainer, Validator
from monai.inferers import SlidingWindowInferer
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import numpy as np
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from medical.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses.dice import DiceLoss
from medical.model.nested_former import NestedFormer
from collections import OrderedDict
import nibabel as nib
import os

parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline for BRATS Challenge')
parser.add_argument('--model_name', default="swinunetr", help='the model will be trained')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--fold', default=0, type=int, help='data fold')
parser.add_argument('--pretrain_model_path', default='./model.pt', type=str, help='pretrained model name')
parser.add_argument('--load_pretrain', action="store_true", help='pretrained model name')
parser.add_argument('--data_dir', default='/mnt/Nestedformer/dataset/brats2020/MICCAI_BraTS2020_TrainingData', type=str, help='dataset directory')
parser.add_argument('--json_list', default='./brats2020_datajson.json', type=str, help='dataset json file')
parser.add_argument('--max_epochs', default=300, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=2, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=1, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--val_every', default=10, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization name')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--feature_size', default=24, type=int, help='feature size')
parser.add_argument('--in_channels', default=2, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--cache_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.25, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
args = parser.parse_args()

def post_pred(pred):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    return pred

def save_image(output, name, output_dir):
    npx = output.cpu().numpy()
    affine = np.eye(4) 
    nii_image = nib.Nifti1Image(npx, affine)
    path = os.path.join(output_dir, f"{name}_nf_mask.nii.gz")
    nib.save(nii_image, path)
    print(f"{name} 已保存至 {path}")

def gen_result(img, name, output_dir):
    torch.cuda.empty_cache()
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    model = NestedFormer(model_num=2,
                        out_channels=3,
                        image_size=inf_size,
                        window_size=(4, 4, 4),
                        )

    model.to("cuda")
    checkpoint = torch.load(args.checkpoint)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[k.replace('backbone.','')] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    # train_loader, val_loader = get_loader(args)
    # print(val_loader)
    with torch.no_grad():
        window_infer = SlidingWindowInferer(roi_size=inf_size,
                                            sw_batch_size=args.sw_batch_size,
                                            overlap=args.infer_overlap)
        sliding_window_infer = window_infer

        output = post_pred(sliding_window_infer(img, model))
        output = output[0, 0]
        save_image(output, name, output_dir)

def get_images(t1_path, t2_path, output_dir):

    for t1, t2 in zip(os.listdir(t1_path), os.listdir(t2_path)):
        
        t1 = os.path.join(t1_path, t1)
        t2 = os.path.join(t2_path, t2)
        
        nii_t1 = torch.tensor(nib.load(t1).get_fdata(), dtype=torch.float32)
        nii_t2 = torch.tensor(nib.load(t2).get_fdata(), dtype=torch.float32)

        img = torch.stack([nii_t1, nii_t2], dim=0).unsqueeze(0).to("cuda")
        name = os.path.splitext(os.path.splitext(t1)[0])[0].split('/')[-1]

        print(f'{name} has been added!')

        gen_result(img, name, output_dir)

        


if __name__ == '__main__':

    base_fold = "/mnt/brain_tumor"
    t1_path = os.path.join(base_fold, 't1/data')
    t2_path = os.path.join(base_fold, 't2/data')
    output_dir = "/mnt/brain_tumor/results"

    get_images(t1_path, t2_path, output_dir)





'''
python visualization.py --checkpoint=./runs/log_brain_tumor/model.pt --out_channels=3 --batch_size=1 --infer_overlap=0.5
'''