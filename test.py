import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from LDWTUNet import LDWTUNet
from dataset import TestDataset


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, required=True, 
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, required=True,
                    help="path to the mask files for testing")
parser.add_argument("--txt_root", type=str, required=True,   # 新增
                    help="path to the text tag directory for testing")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the predicted masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352, txt_root=args.txt_root)
model = LDWTUNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, label_idx, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        label_idx_tensor = torch.tensor([label_idx], device=device)
        res, _, _ = model(image, label_idx_tensor)
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        # If you want to binarize the prediction results, please uncomment the following three lines. 
        # Note that this action will affect the calculation of evaluation metrics.
        # lambda = 0.5
        # res[res >= int(255 * lambda)] = 255
        # res[res < int(255 * lambda)] = 0
        filename = os.path.basename(name)
        save_name = os.path.splitext(filename)[0] + ".png"
        save_path = os.path.join(args.save_path, save_name)
        imageio.imsave(save_path, res)
        print("Saving", save_path)
