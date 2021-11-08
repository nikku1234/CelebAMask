# from .utils import load_state_dict_from_url
import os
import glob
from posixpath import join
from re import split
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.io import read_image
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import argparse
from torchvision.transforms.transforms import RandomAffine, RandomVerticalFlip
from dataLoader.celebDataset import celebDatasetTrain,celebDatasetVal,celebDatasetTest
from model.unet import unet
from matplotlib import pyplot as plt
from torchsummary import summary

train_losses = []
train_accu = []
eval_losses = []
eval_accu = []


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


def train(args, model, device, train_loader, optimizer, scheduler, epoch, criterion):
    model.train()
    running_loss = 0.0
    sub_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(train_loader):
        inputs = data['image'].to(device)
        labels = data['class_id'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        sub_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 3 == 0:
            print("loss:", sub_loss/3, "Accuracy:", correct/total)
            sub_loss = 0.0

    # scheduler.step()
    train_loss = running_loss/len(train_loader)
    accu = 100.*correct/total

    train_accu.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f' % (train_loss, accu))
    return train_loss, accu


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    running_loss = 0.0
    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs = data['image'].to(device)
            labels = data['class_id'].to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        test_loss = running_loss/len(test_loader)
        accu = 100.*correct/total

        eval_losses.append(test_loss)
        eval_accu.append(accu)

        print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))
        return test_loss, accu


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CelebHQ')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
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
                                         transforms.RandomAffine(
                                             degrees=(-30, 30), translate=(0.1, 0.1), shear=(0.2)),
                                         transforms.RandomHorizontalFlip(0.3),
                                         transforms.RandomGrayscale(0.3),
                                         transforms.RandomVerticalFlip(0.3),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    root_dir = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data'

    checkpoint_dir = './checkpoints/model1/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    # load_checkpoint_path = '/home/nikhil/673/ckpts/model_75.pt'
    # load_checkpoint_path = './checkpoints/ckpts_resnet18_cubs_seq12_f1/model_60.pt'
    load_checkpoint_path = None

    celebDataset_train = celebDatasetTrain(root_dir,transformation)

    transformation_val = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    celebDataset_val = celebDatasetVal(root_dir, transformation_val)
    celabDataset_test = celebDatasetTest(root_dir, transformation_val)

    train_loader = DataLoader(celebDataset_train, **train_kwargs, shuffle=True, num_workers=6)

    val_loader = DataLoader(celebDataset_val, **test_kwargs, shuffle=True, num_workers=6)
        
    test_loader = DataLoader(celabDataset_test, **test_kwargs, shuffle=True, num_workers=6)


    model = Net()
    model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=2e-5)
    criterion = nn.CrossEntropyLoss().to(device)
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
        train_loss, train_accuracy = train(
            args, model, device, train_loader, optimizer, exp_lr_scheduler, epoch, criterion)
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
