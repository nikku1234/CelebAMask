# from .utils import load_state_dict_from_url
import os
import glob
from posixpath import join
from re import split
from cv2 import phase
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import io
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.io import read_image
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import argparse
from torchvision.transforms.transforms import RandomAffine, RandomVerticalFlip
from dataLoader.celebDataset import celebDatasetTrain,celebDatasetVal,celebDatasetTest
# from model.unet import unet
from matplotlib import pyplot as plt
from torchsummary import summary
from collections import defaultdict
import torch.nn.functional as F
from loss.loss import dice_loss
from model.new_unet import ResNetUNet
from model.unet import unet
from utils import *

train_losses = []
train_iou = []
eval_losses = []
eval_iou = []


def reset_grad(self):
    self.g_optimizer.zero_grad()



def save_checkpoint(model, epoch, optimizer, scheduler, loss, checkpoint_dir, max_checkpoints=5):
    checkpoints_list = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    if len(checkpoints_list) == max_checkpoints:
        outdated = 10000000
        ckpt_to_replace = None
        for ckpt in checkpoints_list:
            temp = os.path.split(ckpt)[-1]
            saved_epoch = int(temp.split('.')[0].split('_')[-1])
            if saved_epoch < outdated:
                outdated = saved_epoch
                ckpt_to_replace = ckpt
        os.remove(ckpt_to_replace)
    path = os.path.join(checkpoint_dir, 'model_' + str(epoch) + '.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # for state in scheduler.state.values():
    #     for k, v in state.items():
    #         if torch.is_tensor(v):
    #             state[k] = v.cuda()

    return model, optimizer, scheduler, checkpoint['epoch']


def save_statistics(epoch, training_loss, train_accuracy, validation_loss, validation_accuracy, checkpoint_dir):
    loc = os.path.join(checkpoint_dir, 'history.txt')
    data = 'epoch: '+str(epoch) + '\ntraining_loss: ' + str(training_loss) + ',' + '   Train_accuracy:' + str(
        train_accuracy) + '\nvalidation_loss: ' + str(validation_loss) + '   Validation_accuracy:' + str(validation_accuracy)

    with open(loc, 'a') as file:
        file.write('\n\n')
        file.write(data)


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = nn.CrossEntropyLoss()

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

# PyTroch version


SMOOTH = 1e-6



# def _fast_hist(true, pred, num_classes):
#     mask = (true >= 0) & (true < num_classes)
#     hist = torch.bincount(
#         num_classes * true[mask] + pred[mask],
#         minlength=num_classes ** 2,
#     ).reshape(num_classes, num_classes).float()
#     return hist

# # computes IoU based on confusion matrix


# def jaccard_index(hist):
#     """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
#     Args:
#         hist: confusion matrix.
#     Returns:
#         avg_jacc: the average per-class jaccard index.
#     """
#     EPS = 1e-9
#     A_inter_B = torch.diag(hist)
#     A = hist.sum(dim=1)
#     B = hist.sum(dim=0)
#     jaccard = A_inter_B / (A + B - A_inter_B + EPS)
#     # avg_jacc = torch.nanmean(jaccard)  # the mean of jaccard without NaNs
#     avg_jacc = torch.mean(jaccard[~jaccard.isnan()])
#     return avg_jacc, jaccard

def mIOU(label, pred, num_classes=19):
    pred = F.softmax(pred, dim=1)              
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)



def train(args, model, device, train_loader, optimizer, scheduler, epoch, criterion):
    model.train()
    running_loss = 0.0
    sub_loss = 0.0
    total_iou = 0
    total = 0
    metrics = defaultdict(float)
    epoch_samples = 0
    phase = 'train'
    for batch_idx, data in enumerate(train_loader):
        inputs = data['image'].to(device)
        labels = data['label'].to(device)
        size = labels.size()

        # labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
        # labels_real_plain = labels[:, 0, :, :].cuda()
        # labels = labels[:, 0, :, :].view(size[0], 1, size[2], size[3])
        # oneHot_size = (size[0], 19, size[2], size[3])
        # labels_real = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        # labels_real = labels_real.scatter_(1, labels.data.long().cuda(), 1.0)

        # labels = labels.squeeze()
        # print(labels.size())
        optimizer.zero_grad()
        outputs = model(inputs)
        # mask = torch.zeros((labels.size(0), 512, 512)).to(device)


        # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        # 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        # for l, att in enumerate(atts, 1):
        #     total += 1
        #     # file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
        #     # path = osp.join(face_sep_mask, str(i), file_name)

        #     # if os.path.exists(path):
        #         # counter += 1
        #     # sep_mask = np.array(Image.open(path).convert('P'))
        #     for channel in range(int(outputs.size()[1])):
        #         sep_mask = outputs[:, channel, :, :].squeeze()
        #         # print(np.unique(sep_mask))
        #         mask[torch.round(sep_mask) == 225] = l

        # # criterion = nn.NLLLoss()
        
        # # c_loss = cross_entropy2d(outputs, labels_real_plain.long())
        # # print(c_loss.item())
        # # reset_grad()



        # # loss = dice_loss(outputs,labels)
        # # loss = calc_loss(outputs, labels, metrics)
        # # mask = mask.flatten()

        labels = labels.reshape((labels.size(0), 512, 512))
        loss = criterion(outputs, labels.long())
        print("loss", loss.item())
        # iou = IoU_score(outputs, labels)
        # iou = jaccard_index(_fast_hist(labels.long(), mask.long(), 19))
        iou = mIOU(labels, outputs)
        total_iou += iou
        # _fast_hist(true, pred, num_classes=2)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # sub_loss += loss.item()

        # _, predicted = torch.max(outputs, 1)
        # total += labels.size(0)
        # correct += predicted.eq(labels).sum().item()

        # if (batch_idx + 1) % 3 == 0:
        #     print("loss:", sub_loss/3, "Accuracy:", correct/total)
        #     sub_loss = 0.0
        # statistics
        epoch_samples += inputs.size(0)

        # print("iou",iou)
    # scheduler.step()
    train_loss = running_loss/len(train_loader)
    # accu = 100.*correct/total
    mean_iou = total_iou / len(train_loader)
    train_losses.append(train_loss)
    train_iou.append(mean_iou)
    # mean_iou = total_iou/len(train_loader)
    print('Train Loss: %.3f | IoU: %.3f' % (train_loss, mean_iou))
    return train_loss, mean_iou


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    running_loss = 0.0
    total_iou = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # mask = torch.zeros((512, 512))
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #         'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

            # for l, att in enumerate(atts, 1):
            #     total += 1
            #     # file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            #     # path = osp.join(face_sep_mask, str(i), file_name)

            #     # if os.path.exists(path):
            #     # counter += 1
            #     # sep_mask = np.array(Image.open(path).convert('P'))
            #     for channel in range(int(outputs.size()[1])):
            #         sep_mask = outputs[:, channel, :, :].squeeze()
            #         # print(np.unique(sep_mask))
            #         mask[sep_mask == 225] = l


            labels = labels.reshape((labels.size(0), 512, 512))
            loss = criterion(outputs, labels.long())
            # iou = IoU_score(outputs, labels)
            total_iou += mIOU(labels, outputs)
            # _fast_hist(true, pred, num_classes=2)
            # print("test loss", loss.item())
            # print("test iou", iou[0])
            running_loss += loss.item()

            # _, predicted = outputs.max(1)
            # total += labels.size(0)
            # correct += predicted.eq(labels).sum().item()
        test_loss = running_loss/len(test_loader)
        # accu = 100.*correct/total

        eval_losses.append(test_loss)
        # eval_accu.append(accu)
        mean_iou = total_iou / len(test_loader)
        eval_iou.append(mean_iou)

        print('Test Loss: %.3f | IoU: %.3f' % (test_loss, mean_iou))
        return test_loss, eval_iou


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CelebHQ')
    parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=12, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: .03)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    train_kwargs = {'batch_size': args.batch_size}
    print("train_batch size", train_kwargs)
    test_kwargs = {'batch_size': args.test_batch_size}
    print("test_batch size", test_kwargs)

    transformation = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((512,512)),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    transformation_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((512, 512)),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    transformation_target = transforms.Compose([transforms.ToTensor()])

    root_dir = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data'

    checkpoint_dir = './checkpoints/model1/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    load_checkpoint_path = None

    celebDataset_train = celebDatasetTrain(
        root_dir, transformation, transformation_target)

    
    
    celebDataset_val = celebDatasetVal(
        root_dir, transformation_val, transformation_target)

    celabDataset_test = celebDatasetTest(root_dir, transformation_val, transformation_target)

    train_loader = DataLoader(celebDataset_train, **train_kwargs, shuffle=True, num_workers=6)

    val_loader = DataLoader(celebDataset_val, **test_kwargs, shuffle=True, num_workers=6)
        
    test_loader = DataLoader(celabDataset_test, **test_kwargs, shuffle=True, num_workers=6)


    model = unet()
    model.to(device)
    print(model)

    # model = ResNetUNet(n_class=19)
    # model = torch.hub.load('pytorch/vision:v0.10.0',
    #                        'deeplabv3_resnet50', pretrained=False)


    model = model.to(device)

    # check keras-like model summary using torchsummary
    summary(model, input_size=(3, 512, 512))


    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=2e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.NLLLoss().to(device)
    # criterion = nn.BCEWithLogitsLoss()

    last_epoch = 0
    exp_lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    # exp_lr_scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5)

    if load_checkpoint_path is not None:
        print('loading')
        model, optimizer, exp_lr_scheduler, last_epoch = load_checkpoint(
            model, optimizer, exp_lr_scheduler, load_checkpoint_path, device)
    model.to(device)
    # optimizer.cuda()
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(last_epoch + 1, args.epochs + 1):
        train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, exp_lr_scheduler, epoch, criterion)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        exp_lr_scheduler.step()
        if epoch % 10 == 0:
            save_checkpoint(model, epoch, optimizer, exp_lr_scheduler,
                            train_loss, checkpoint_dir, max_checkpoints=100)
        save_statistics(epoch, train_loss, train_accuracy,
                        test_loss, test_accuracy, checkpoint_dir)

    plt.plot(train_accu, '-o')
    plt.plot(eval_accu, '-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.savefig('accuracy.png')

    plt.plot(train_losses, '-o')
    plt.plot(eval_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig('Losses.png')

    # plt.show()

    if args.save_model:
        torch.save(model.state_dict(), "new_cnn.pt")


if __name__ == '__main__':
    main()
