import argparse
import json
import multiprocessing

import pytorch_lightning as pl
from lightning_lite import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from OriginalDivideMix.PreResNet import ResNet18
from OriginalDivideMix.train_cifar import SemiLoss, NegEntropy, CEloss
from configuration import Configuration
from data_parser.data_splitter import DataSplitter
from data_parser.dia_to_metadata_parser import DiaToMetadata
from torchvision import transforms

from data_parser.tiles_dataset import TilesDataset

transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.], std=[255.])])


def create_model():
    model = ResNet18(num_classes=2)
    model = model.cuda()
    return model


def init_argparse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--noise_mode', default='sym')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
    parser.add_argument("-device", type=str)
    parser.add_argument("--gene", type=str)

    return parser


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        # TODO- assume noise_mode is sym, check if it is true
        # if args.noise_mode == 'asym':  # penalize confident prediction for asymmetric noise
        #     penalty = conf_penalty(outputs)
        #     L = loss + penalty
        # elif args.noise_mode == 'sym':
        L = loss
        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()


def start_train(args):
    gene = args.gene
    device = args.device
    if device is None:
        print("Device must be provided")
        exit(1)

    tiles_directory_path = Configuration.TILES_DIRECTORY.format(zoom_level=Configuration.ZOOM_LEVEL,
                                                                patch_size=Configuration.PATCH_SIZE)

    dia_metadata = DiaToMetadata(Configuration.DIA_GENES_FILE_PATH, Configuration.RNR_METADATA_FILE_PATH,
                                 tiles_directory_path)

    data_splitter = DataSplitter(dia_metadata)
    num_workers = int(multiprocessing.cpu_count())

    extreme, ood = dia_metadata.split_by_expression_level(gene)

    with open(Configuration.OOD_FILE_PATH.format(gene=gene), "w") as f:
        json.dump(ood, f)

    train_instances, valid_instances = data_splitter.split_train_val(extreme,
                                                                     seed=Configuration.SEED,
                                                                     val_proportion=0.35)
    train_dataset = TilesDataset(tiles_directory_path, transform_compose, train_instances, "Train-dataset")
    validation_dataset = TilesDataset(tiles_directory_path, transform_compose, valid_instances, "Val-dataset")
    train_loader = DataLoader(train_dataset, batch_size=Configuration.BATCH_SIZE, num_workers=num_workers,
                              persistent_workers=True, pin_memory=True, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=Configuration.BATCH_SIZE, num_workers=num_workers,
                                   persistent_workers=True, pin_memory=True)

    net1 = create_model()
    net2 = create_model()
    warm_up = 5

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode == 'asym':
        conf_penalty = NegEntropy()

    all_loss = [[], []]  # save the history of losses from two networks

    for epoch in range(args.num_epochs + 1):
        lr = args.lr
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr
        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train')

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader)

        else:
            prob1, all_loss[0] = eval_train(net1, all_loss[0])
            prob2, all_loss[1] = eval_train(net2, all_loss[1])

            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

        test(epoch, net1, net2)


def main():
    parser = init_argparse()
    args = parser.parse_args()

    start_train(args)


if __name__ == '__main__':
    main()
