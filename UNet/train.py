import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset import RandomCrop, RandomRotFlip, ToTensor, base_dataset
from networks.net_factory import net_factory
from PIL import Image
from val_2D import val_binary_2d


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes, onehot=True):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.onehot = onehot

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if self.onehot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), "predict & target shape do not match"
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default="../data/Promise12/Foreslices/splits/train.csv",
        help="base_dir of data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../outputs/Promise12/train",
    )
    parser.add_argument(
        "--patch_size",
        nargs=2,
        type=int,
        default=[224, 224],
        help="patch size of network input",
    )

    parser.add_argument("--batch_size", type=int, default=16, help="batch_size per gpu")

    parser.add_argument(
        "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
    )

    parser.add_argument(
        "--max_epoch", type=int, default=400, help="maximum epoch number to train"
    )

    parser.add_argument(
        "--class_num", type=int, default=2, help="number of classes for segmentation"
    )

    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )

    parser.add_argument("--seed", type=int, default=2025, help="random seed")

    return parser.parse_args()


def train(args):
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.basicConfig(
        filename=output_path + "/training_log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = net_factory(net_type="unet", in_chns=1, class_num=args.class_num)

    dataset = base_dataset(
        csv_path=args.csv_path,
        transform=transforms.Compose(
            [
                RandomCrop(args.patch_size),
                RandomRotFlip(),
                ToTensor(),
            ]
        ),
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    model.train()

    optimizer = optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001
    )
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=args.class_num)

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice = 0.0
    max_iterations = args.max_epoch * len(trainloader)
    for epoch_num in range(args.max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())

            outputs_soft = torch.softmax(outputs, dim=1)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1

            logging.info(
                "iteration %d : loss : %f, loss_ce: %f, loss_dice: %f"
                % (iter_num, loss.item(), loss_ce.item(), loss_dice.item())
            )

            if iter_num > 0 and iter_num % 200 == 0:  ## 200
                model.eval()
                dice_mean, dice_std, hd95_mean, hd95_std = val_binary_2d(
                    csv_path=os.path.join(os.path.dirname(args.csv_path), "valid.csv"),
                    net=model,
                    patch_size=args.patch_size,
                    output_path=None,
                )

                if dice_mean > best_dice:
                    best_dice = dice_mean
                    save_mode_path = os.path.join(
                        args.output_path,
                        "iter_{}_dice_{}.pth".format(iter_num, round(best_dice, 4)),
                    )
                    save_best = os.path.join(args.output_path, "best_model.pth")
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    f"iteration {iter_num} : dice : {dice_mean:.4f} std: {dice_std:.4f}, hd95: {hd95_mean:.4f} std: {hd95_std:.4f}"
                )
                model.train()

    return "Training Finished!"


if __name__ == "__main__":
    args = config()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train(args=args)
