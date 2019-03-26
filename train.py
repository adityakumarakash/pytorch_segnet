import argparse
import collections
import datetime
import functools
import google.protobuf.text_format as txtf
import logging
import numpy as np
import os
import random
import shutil
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


from datasets.camvid_dataset import CamvidDataset
from models.segnet import Segnet
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm
from utils import config_pb2
from utils.average_meter import AverageMeter
from utils.running_score import RunningScore


def get_loss_fn(loss_args, logger):
    # Returns the appropriate loss function depenending on the arguments.
    def cross_entropy(input, target, size_average=True):
        return F.cross_entropy(input=input, target=target,
                               size_average=size_average, ignore_index=250)

    logger.info("Loading cross entropy")
    loss_fn = functools.partial(cross_entropy,
                                size_average=loss_args.size_average)
    return loss_fn


def get_optimizer(model, optimizer_args, logger):
    # Returns the appropriate optimizer based on the config.
    logger.info("Loaded SGD optimizer")
    optimizer_class = optim.SGD
    specific_args = {arg.split(':')[0]:float(arg.split(':')[1])
                     for arg in optimizer_args.args.split(',')}
    optimizer = optimizer_class(model.parameters(), lr=optimizer_args.lr,
                                **specific_args)
    return optimizer


def get_data_loaders(config, logger):
    # Returns the appropriate data loaders for training and validation.
    logger.info("Creating Camvid dataset loaders")
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    # Training dataset.
    train_dataset = CamvidDataset(config.data.path, config.data.train_split,
                                  transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=config.training.batch_size,
                                   num_workers=config.training.num_workers, shuffle=True,)

    # Validation dataset.
    val_dataset = CamvidDataset(config.data.path, config.data.val_split,
                                       transform=transform)
    val_loader = data.DataLoader(val_dataset, batch_size=config.training.batch_size,
                                        num_workers=config.training.num_workers)

    num_classes = train_dataset.num_classes
    return train_loader, val_loader, num_classes


def train_epoch(module, config, writer, logger):
    # Trains the model for a single epoch.
    batch_time = AverageMeter()
    train_loss = AverageMeter()

    # Unpacks the module
    model = module.model
    device = module.device
    train_loader = module.train_loader
    loss_fn = module.loss_fn
    optimizer = module.optimizer

    model.train()
    idx = 0
    for images, labels in train_loader:
        idx += 1
        start_tic = time.time()
        
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(input=outputs, target=labels)
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - start_tic)
        train_loss.update(loss.data.item())

        if idx % config.training.disp_iter == 0:
            # This is the iteration to display the information.
            print_str = "Iter {:d} Loss: {:.4f} Time/Batch: {:.4f}".format(idx,
                                                                           train_loss.average(),
                                                                           batch_time.average())
            print(print_str)
            logger.info(print_str)


def save_model(model, optimizer, epoch, best_iou, path):
    # Saves the arguments in state dict.
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_iou": best_iou,
    }
    torch.save(state, path)
    
    
def validate(module, epoch, best_iou, num_classes, writer, logger):
    # Runs validation for the model on the appropriate split and returns best iou.
    # Unpack the module
    model = module.model
    device = module.device
    val_loader = module.val_loader
    loss_fn = module.loss_fn

    avg_loss = AverageMeter()
    running_score = RunningScore(num_classes)

    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in tqdm(enumerate(val_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(input=outputs, target=labels)
            
            avg_loss.update(loss.data.item())

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            running_score.update(gt, pred)

    writer.add_scalar("Val Loss", avg_loss.average(), epoch)
    logger.info("Epoch: {} Loss: {:.4f}".format(epoch, avg_loss.average()))

    mean_iou, disp_score = running_score.get_scores()
    logger.info(disp_score)
    if mean_iou >= best_iou:
        # Saves the model if the current mean_iou is better.
        best_iou = mean_iou
        path = os.path.join(writer.file_writer.get_logdir(),
                            "best_model.pkl")
        save_model(model=model, optimizer=module.optimizer, epoch=epoch,
                   best_iou=best_iou, path=path)
    return best_iou


def checkpoint(module, epoch, best_iou, writer):
    path = os.path.join(writer.file_writer.get_logdir(),
                        "model_epoch_{}.pkl".format(epoch))
    save_model(model=module.model, optimizer=module.optimizer,
               epoch=epoch, best_iou=best_iou, path=path)


def train(config, writer, logger):
    # Trains the model end to end.

    # Sets up random seeds.
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Sets up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sets up dataloader.
    train_loader, val_loader, num_classes = get_data_loaders(config, logger)
    
    # Sets up model.
    model = Segnet(num_classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Sets up optimizer.
    optimizer = get_optimizer(model, config.training.optimizer, logger)

    # Sets up loss function.
    loss_fn = get_loss_fn(config.training.loss, logger)

    epoch = 0
    best_iou = -100
    # Handles model resume point.
    if config.training.HasField('resume_path'):
        resume_path = config.training.resume_path
        if os.path.isfile(resume_path):
            logger.info("Resuming from {}".format(resume_path))
            resume_point = torch.load(resume_path)
            model.load_state_dict(resume_point["model_state"])
            optimizer.load_state_dict(resume_point["optimizer_state"])
            epoch = resume_point["epoch"]
            best_iou = resume_point["best_iou"]
            logger.info("Resume point loaded")
        else:
            logger.info("No resume file found {}".format(resume_path))

    # The module encapsulates the variables relevant for the model training.
    Module = collections.namedtuple('Module', ['model', 'optimizer', 'loss_fn',
                                               'train_loader', 'val_loader', 'device'])
    module = Module(model, optimizer, loss_fn, train_loader, val_loader, device)

    while epoch < config.training.epoch:
        # Trains the model for specified number of epochs.
        epoch += 1
        train_epoch(module, config, writer, logger)
        # Creating checkpoints every epoch might be costly. Best iou model is saved in
        # validation step.
        #checkpoint(module, epoch, best_iou, writer) 
        if epoch % config.training.val_epoch == 0:
            best_iou = validate(module, epoch, best_iou, num_classes, writer, logger)

    logger.info("Training finished!")
    print('Training finished!')


def get_logger(log_dir):
    # Returns the logger.
    timestamp = str(datetime.datetime.now())
    for ch in ": -.":
        timestamp = timestamp.replace(ch, "_")

    log_file = os.path.join(log_dir, "run_{}.log".format(timestamp))
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    logger = logging.getLogger("Pytorch_Segnet")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    # Parses the command line arguments
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file for model",
    )
    args = parser.parse_args()

    # Creates config using the arguments.
    config = config_pb2.Config()
    with open(args.config) as f:
        txtf.Merge(f.read(), config)

    # Initializes run_id and log directory.
    run_id = random.randint(1, 100000)
    log_dir = os.path.join("runs", str(run_id))
    writer = SummaryWriter(log_dir=log_dir)

    print("Run dir: {}".format(run_id))
    shutil.copy(args.config, log_dir)
    logger = get_logger(log_dir)
    train(config, writer, logger)
    

