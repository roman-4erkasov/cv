import os
# import numpy as np
# from PIL import Image

# import torch
# import torch.utils.data
# from torch.optim.lr_scheduler import StepLR
# import torchvision as vision
# from dataset import PennFudanDataset, get_transform
# from model import (
#     get_instance_segmentation_model, 
#     validate, 
#     load_checkpoint, 
#     save_checkpoint
# )
# from dataset import PennFudanDataset, get_transform, get_data
# from engine import train_one_epoch, evaluate

# import utils

import argparse
from model import predict_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", help="Path to input file", type=str)
    parser.add_argument("--out_path", help="Path to output file", type=str)
    parser.add_argument("--checkpoint_path", help="Path to checkpoint", default="maskrcnn.ckpt")#, nargs="?")
    parser.add_argument("--threshold", help="Threshold for masks and boxes", type=float, default=0.5)#, nargs="?")
    args = parser.parse_args()
    predict_from_file(
        in_path=args.in_path, 
        out_path=args.out_path, 
        checkpoint_path=args.checkpoint_path, 
        threshold=args.threshold
    )
    
# checkpoint_path: str = "maskrcnn.ckpt", 
    # boxes: bool = True, 
    # masks: bool = True,
    # min_score: float = 0.5

# def train(
#     data_path: str, 
#     threshold: float = 0.5, 
#     checkpoint_path: str = "maskrcnn.ckpt", 
#     num_epochs: int = 1, 
#     debug: bool = True
# ):
#     data_loader, data_loader_test = get_data(path=data_path, debug=debug)
#     model = get_instance_segmentation_model(2)
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(
#         params=params,
#         lr=0.005, 
#         momentum=0.9, 
#         weight_decay=0.0005
#     )
#     if os.path.exists(checkpoint_path):
#         (
#             model_state_dict, optimizer_state_dict, epoch_start, loss_min
#         ) = load_checkpoint(checkpoint_path)
#         model.load_state_dict(model_state_dict)
#         optimizer.load_state_dict(optimizer_state_dict)
#     else:
#         epoch_start = 0
#         loss_min = None
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
#     for epoch in range(epoch_start, num_epochs):
#         metric_logger = train_one_epoch(
#             model, optimizer, data_loader, DEVICE, epoch,
#             print_freq=2 if DEBUG else 10,
#         )
#         lr_scheduler.step()
#         loss = validate(model, optimizer, data_loader_test, DEVICE)
#         print(f"loss_val={loss}")
#         if loss_min is None or loss < loss_min:
#             save_checkpoint(model, optimizer, checkpoint_path, epoch, loss)
#             loss_min = loss
#         evaluator = evaluate(
#             model,
#             data_loader_test,
#             device=DEVICE
#         )


# def predict_from_file(
#     in_path: str, 
#     out_path=None, 
#     checkpoint_path: str = "maskrcnn.ckpt", 
#     boxes: bool = True, 
#     masks: bool = True,
#     min_score: float = 0.5
# ):
#     if not os.path.exists(in_path):
#         raise Exception(
#             f"file path does not exist: in_path={in_path}"
#         )
#     if not os.path.exists(checkpoint_path):
#         raise Exception(
#             f"file path does not exist: checkpoint_path={checkpoint_path}"
#         )
#     model = get_instance_segmentation_model(2)
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(params=params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#     (
#         model_state_dict, 
#         optimizer_state_dict, 
#         epoch_start, 
#         loss_min
#     ) = load_checkpoint(checkpoint_path)
#     model.load_state_dict(model_state_dict)
#     optimizer.load_state_dict(optimizer_state_dict)
#     img = Image.open(in_path).convert("RGB")
#     to_tensor = vision.transforms.PILToTensor()
#     tensor = to_tensor(img).to(torch.float)/255
#     model.eval()
#     with torch.no_grad():
#         preds = model(tensor.unsqueeze(0))
#     if out_path is None:
#         return preds
#     else:
#         Vizualizer.save(
#             img=tensor,
#             filepath=out_path,
#             boxes=pred[0]["boxes"][pred[0]["scores"]>threshold,...], 
#             masks=pred[0]["masks"][pred[0]["scores"]>threshold,...],
#         )


        

# min_score = 0.5
# fpath = "/home/ubuntu/maskrcnn/PennFudanPed/PNGImages/FudanPed00019.png"

# img = Image.open(fpath).convert("RGB")
# pred = predict_from_file(fpath)
# to_tensor = vision.transforms.PILToTensor()
# tensor = to_tensor(img).to(torch.float)/255
# Vizualizer.show(
#     img=tensor, 
#     boxes=pred[0]["boxes"][pred[0]["scores"]>.5,...], 
#     masks=pred[0]["masks"][pred[0]["scores"]>.5,...],
# )
