import os
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from visualizer import Visualizer

def save_checkpoint(model, optimizer, save_path, epoch, loss):
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }, 
        save_path
    )

def load_checkpoint(load_path):
    checkpoint = torch.load(load_path)
    return (
        checkpoint['model_state_dict'], 
        checkpoint['optimizer_state_dict'], 
        checkpoint['epoch'],
        checkpoint['loss']
    )

def validate(model, optimizer, val_dataloader, device):
    running_loss = 0
    for data in val_dataloader:
        optimizer.zero_grad()
        images, targets = data[0], data[1]
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        running_loss += loss.item()
    val_loss = running_loss/len(val_dataloader.dataset)
    return val_loss 
      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True
    )

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def train(
    data_path: str, 
    threshold: float = 0.5, 
    checkpoint_path: str = "maskrcnn.ckpt", 
    num_epochs: int = 1, 
    debug: bool = True
):
    data_loader, data_loader_test = get_data(path=data_path, debug=debug)
    model = get_instance_segmentation_model(2)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params=params,lr=0.005, momentum=0.9, weight_decay=0.0005)
    if os.path.exists(checkpoint_path):
        (
            model_state_dict, optimizer_state_dict, epoch_start, loss_min
        ) = load_checkpoint(checkpoint_path)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        epoch_start = 0
        loss_min = None
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for epoch in range(epoch_start, num_epochs):
        metric_logger = train_one_epoch(
            model, optimizer, data_loader, DEVICE, epoch,
            print_freq=2 if DEBUG else 10,
        )
        lr_scheduler.step()
        loss = validate(model, optimizer, data_loader_test, DEVICE)
        print(f"loss_val={loss}")
        if loss_min is None or loss < loss_min:
            save_checkpoint(model, optimizer, checkpoint_path, epoch, loss)
            loss_min = loss
        evaluator = evaluate(
            model,
            data_loader_test,
            device=DEVICE
        )


def predict_from_file(
    in_path: str, 
    out_path=None, 
    checkpoint_path: str = "maskrcnn.ckpt", 
    boxes: bool = True, 
    masks: bool = True,
    threshold: float = 0.5
):
    if not os.path.exists(in_path):
        raise Exception(
            f"file path does not exist: in_path={in_path}"
        )
    if not os.path.exists(checkpoint_path):
        raise Exception(
            f"file path does not exist: checkpoint_path={checkpoint_path}"
        )
    model = get_instance_segmentation_model(2)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params=params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    (
        model_state_dict, 
        optimizer_state_dict, 
        epoch_start, 
        loss_min
    ) = load_checkpoint(checkpoint_path)
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    img = Image.open(in_path).convert("RGB")
    to_tensor = torchvision.transforms.PILToTensor()
    tensor = to_tensor(img).to(torch.float)/255
    model.eval()
    with torch.no_grad():
        preds = model(tensor.unsqueeze(0))
    if out_path is None:
        return preds
    else:
        Visualizer.save(
            img=tensor,
            filepath=out_path,
            boxes=preds[0]["boxes"][preds[0]["scores"] > threshold, ...], 
            masks=preds[0]["masks"][preds[0]["scores"] > threshold, ...],
        )