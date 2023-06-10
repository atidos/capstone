"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Training & Validation
"""
import numpy as np 
import argparse, cv2
import logging
import time
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
import torch.utils.tensorboard as tensorboard
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

import utils
from utils import visualize_confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import SeResNeXt as SeResNeXt

from torch.utils.data import DataLoader, WeightedRandomSampler

cudnn.benchmark = True
cudnn.enabled = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help="training batch size")
    parser.add_argument('--tensorboard', type=str, default='checkpoint/tensorboard', help='path log dir of tensorboard')
    parser.add_argument('--logging', type=str, default='checkpoint', help='path of logging')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
    parser.add_argument('--datapath', type=str, default='data/dataset_age_hybrid2_aligned', help='root path of dataset')
    parser.add_argument('--test_datapath', type=str, default='data/dataset_age_hybrid2_aligned', help='root path of test dataset')
    parser.add_argument('--pretrained', type=str,default='models age/resnext_37_dataset_age_UTK_custom_64_0.005_40_1e-06.pth.tar',help='load checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from pretrained path specified in prev arg')
    parser.add_argument('--savepath', type=str, default='checkpoint', help='save checkpoint path')
    parser.add_argument('--savefreq', type=int, default=1, help="save weights each freq num of epochs")
    parser.add_argument('--logdir', type=str, default='logs', help='logging')
    parser.add_argument("--lr_patience", default=40, type=int)
    parser.add_argument('--evaluate', action='store_true', help='evaluation only')
    parser.add_argument('--mode', type=str, default='val', choices=['val','test', 'train'], help='dataset type for evaluation only')
    args = parser.parse_args()

    return args
# ======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
args = parse_args()
# logging
logging.basicConfig(
format='[%(message)s',
level=logging.INFO,
handlers=[logging.FileHandler(args.logdir + "/resnext50_" +
                              args.datapath.split("/")[-1] + "_" +
                              str(args.batch_size) + "_" +
                              str(args.lr) + "_" +
                              str(args.lr_patience) + "_" +
                              str(args.weight_decay), mode='w'), logging.StreamHandler()
                              ])
# tensorboard
writer = tensorboard.SummaryWriter(args.tensorboard)

transform = transforms.Compose([# transforms.Grayscale(num_output_channels=1),
                                # transforms.ToPILImage(),
                                transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                                transforms.RandomHorizontalFlip(),
                                # transforms.RandomRotation(degrees=10),
                                # transforms.RandomEqualize(p=1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])

def main():

    # ========= dataloaders ===========
    trainDataset = datasets.ImageFolder(args.datapath + "/Train", transform=transform)

    class_weights = []

    for root, subdir, files in os.walk(args.datapath + "/Train"):
        if len(files) > 0:
            class_weights.append(1/len(files))

    #sample_weights = [0] * len(trainDataset)

    #for idx, (data,label) in enumerate(trainDataset):
    #    class_weight = class_weights[label]
    #    sample_weights[idx] = class_weight

    #sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)

    testDataset = datasets.ImageFolder(args.test_datapath + "/Test", transform=transform)
    test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=args.batch_size)

    print(trainDataset.class_to_idx)

    start_epoch = 0

    # ======== models & loss ==========
    resnext = SeResNeXt.se_resnext50(num_classes=5)

    loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

    # ========= load weights ===========
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.pretrained, map_location=device)
        resnext.load_state_dict(checkpoint['resnext'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        print(f'\tLoaded checkpoint from {args.pretrained}\n')
        time.sleep(1)
    else:
        print("******************* Start training from scratch *******************\n")
        time.sleep(2)

    if args.evaluate:
        if args.mode == 'val':
            testDataset = datasets.ImageFolder(args.test_datapath + "/Test", transform=transform)
            test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=args.batch_size, shuffle=True)
        elif args.mode == 'train':
            testDataset = datasets.ImageFolder(args.test_datapath + "/Train", transform=transform)
            test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=args.batch_size, shuffle=True)

        validate(resnext, loss, test_dataloader, 0)
        return

    # =========== optimizer ===========
    # parameters = resnext.named_parameters()
    # for name, p in parameters:
    #     print(p.requires_grad, name)
    # return
    optimizer = torch.optim.Adam(resnext.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)
    # ========================================================================
    for epoch in range(start_epoch, args.epochs):
        # =========== train / validate ===========
        train_loss = train_one_epoch(resnext, loss, optimizer, train_dataloader, epoch)
        val_loss, accuracy, percision, recall = validate(resnext, loss, test_dataloader, epoch)
        scheduler.step(val_loss)
        val_loss, accuracy, percision, recall = round(val_loss,3), round(accuracy,3), round(percision,3), round(recall,3)
        logging.info(f"\ttraining epoch={epoch} .. train_loss={train_loss}")
        logging.info(f"\tvalidation epoch={epoch} .. val_loss={val_loss}")
        logging.info(f'\tAccuracy = {accuracy*100} % .. Percision = {percision*100} % .. Recall = {recall*100} % \n')
        time.sleep(2)
        # ============= tensorboard =============
        writer.add_scalar('train_loss',train_loss, epoch)
        writer.add_scalar('val_loss',val_loss, epoch)
        writer.add_scalar('percision',percision, epoch)
        writer.add_scalar('recall',recall, epoch)
        writer.add_scalar('accuracy',accuracy, epoch)
        # ============== save model =============
        if epoch % args.savefreq == 0:
            checkpoint_state = {
                'resnext': resnext.state_dict(),
                "epoch": epoch
            }
            savepath = os.path.join(args.savepath, "resnext_"+f'{epoch}' + "_" +
                                                   args.datapath.split("/")[-1] + "_" +
                                                   str(args.batch_size) + "_" +
                                                   str(args.lr) + "_" +
                                                   str(args.lr_patience) + "_" +
                                                   str(args.weight_decay)
                                                   + '.pth.tar')
            torch.save(checkpoint_state, savepath)
            print(f'\n\t*** Saved checkpoint in {savepath} ***\n')
            time.sleep(2)
    writer.close()

def train_one_epoch(model, criterion, optimizer, dataloader, epoch):
    model.train()
    model.to(device)
    losses = []

    for images, labels in tqdm(dataloader):
        images = images.to(device) # (batch, 1, 48, 48)
        labels = labels.to(device) # (batch,)
        emotions = model(images)
        # from (batch, 7, 1, 1) to (batch, 7)
        emotions = torch.squeeze(emotions)
        # print(emotions)
        # print(labels,'\n')

        if len(labels) == 1:
            labels = labels[0]

        loss = criterion(emotions, labels)
        losses.append(loss.cpu().item())
        print(f'training @ epoch {epoch} .. loss = {round(loss.item(),3)}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # images = images.squeeze().cpu().detach().numpy()
        # cv2.imshow('f', images[0])
        # cv2.waitKey(0)

    return round(np.mean(losses).item(),3)


def validate(model, criterion, dataloader, epoch):
    model.eval()
    model.to(device)
    losses = []

    total_pred = []
    total_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            mini_batch = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)

            emotions = model(images)
            emotions = torch.squeeze(emotions)
            emotions = emotions.reshape(mini_batch, -1)

            loss = criterion(emotions, labels)

            losses.append(loss.cpu().item())

            # # ============== Evaluation ===============
            # index of the max value of each sample (shape = (batch,))
            _, indexes = torch.max(emotions, axis=1)
            # print(indexes.shape, labels.shape)
            total_pred.extend(indexes.cpu().detach().numpy())
            total_labels.extend(labels.cpu().detach().numpy())

            print(f'validation loss = {round(loss.item(),3)}')

        val_loss = np.mean(losses).item()
        percision = precision_score(total_labels, total_pred, average='macro')
        recall = recall_score(total_labels, total_pred, average='macro')
        accuracy = accuracy_score(total_labels, total_pred)

        val_loss, accuracy, percision, recall = round(val_loss,3), round(accuracy,3), round(percision,3), round(recall,3)
        print(f'Val loss = {val_loss} .. Accuracy = {accuracy} .. Percision = {percision} .. Recall = {recall}')

        if args.evaluate:
            conf_matrix = confusion_matrix(total_labels, total_pred, normalize='true')
            print('Confusion Matrix\n', conf_matrix)
            visualize_confusion_matrix(conf_matrix, size=5)

        return val_loss, accuracy, percision, recall

if __name__ == "__main__":
    main()
