import argparse
import glob
import json
import multiprocessing

import pytorch_lightning as pl
from lightning_lite import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
# from pytorch_lightning.loggers import WandbLogger
import wandb
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

transform_compose = transforms.Compose([transforms.Resize(size=(32, 32)),
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
    parser.add_argument("--gene", default="MKI67", type=str)

    return parser


def warmup(args, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        if args.noise_mode == 'asym':  # penalize confident prediction for asymmetric noise
            # penalty = conf_penalty(outputs)
            # L = loss + penalty
            raise NotImplementedError()
        elif args.noise_mode == 'sym':
            L = loss
        L.backward()
        optimizer.step()

        wandb.log({"warmup_loss": loss.item()})
        # sys.stdout.write('\r')
        # sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
        #                  % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
        #                     loss.item()))
        # sys.stdout.flush()


def start_train(args):
    print("Starting training")
    gene = args.gene
    device = args.device
    if device is None:
        print("Device must be provided")
        exit(1)

    tiles_directory_path = Configuration.TILES_DIRECTORY.format(zoom_level=Configuration.ZOOM_LEVEL,
                                                                patch_size=Configuration.PATCH_SIZE)

    dia_metadata = DiaToMetadata(Configuration.DIA_GENES_FILE_PATH, Configuration.RNR_METADATA_FILE_PATH,
                                 tiles_directory_path)
    print("Got all metadata")
    data_splitter = DataSplitter(dia_metadata)
    num_workers = int(multiprocessing.cpu_count())

    extreme, ood = dia_metadata.split_by_expression_level(gene)

    with open(Configuration.OOD_FILE_PATH.format(gene=gene), "w") as f:
        json.dump(ood, f)

    project_name = "proteomics-project"

    # versions = []
    # for path in glob.glob(Configuration.CHECKPOINTS_PATH.format(gene=gene) + "/" + project_name + "/*--v_*"):
    #     name = path.split("/")[-1]
    #     version = int(name.split("_")[-1])
    #     versions.append(version)
    #
    # if len(versions) == 0:
    #     version = 0
    # else:
    #     version = max(versions) + 1
    #
    # run_version = "{gene}--v_{version}".format(gene=gene, version=str(version))
    wandb.init(project=project_name, name="MyFirstAttempt")
    # wandb_logger = WandbLogger(project=project_name, log_model=True,
    #                            save_dir=Configuration.CHECKPOINTS_PATH.format(gene=gene),
    #                            version=run_version)
    net1 = create_model()
    net2 = create_model()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = SemiLoss()
    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()

    all_loss = [[], []]  # save the history of losses from two networks
    warm_up_iterations = 3

    for epoch in range(args.num_epochs + 1):
        print('\nEpoch: %d' % epoch)
        lr = args.lr
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        if epoch < warm_up_iterations:
            tiles_dataset = TilesDataset(tiles_directory_path, ids=extreme, mode="all", noise_ratio=0.2)
            warmup_loader = DataLoader(tiles_dataset, batch_size=Configuration.BATCH_SIZE * 2,
                                       shuffle=True, num_workers=num_workers)

            print('Warmup Net1')
            warmup(args, net1, optimizer1, warmup_loader)
            print('\nWarmup Net2')
            warmup(args, net2, optimizer2, warmup_loader)

    wandb.finish()


def main():
    parser = init_argparse()
    args = parser.parse_args()

    start_train(args)


if __name__ == '__main__':
    main()
