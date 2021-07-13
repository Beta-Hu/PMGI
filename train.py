import torch
from torch.utils.data import DataLoader
from model import PMGI
from dataset import MFI_WHU
from loss import grad_loss, ints_loss
from metric import SSIM
from time import time

DEVICE = 'cuda:0'


def train():
    net = PMGI().to(DEVICE)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4)

    datas_train = DataLoader(MFI_WHU('./MFI-WHU/train'),
                             batch_size=32,
                             shuffle=True,
                             drop_last=True,
                             num_workers=2,
                             pin_memory=True)
    datas_test = DataLoader(MFI_WHU('./MFI-WHU/test'),
                            batch_size=1,
                            shuffle=True,
                            drop_last=True)

    max_loss = 1e6
    itera = 50

    for epoch in range(1, 100 + 1):
        epoch_loss = 0
        epoch_ssim = 0
        iter_loss = 0
        st_time = time()
        for idx, data in enumerate(datas_train):
            focus_1, focus_2, _ = data
            focus_1 = focus_1.to(DEVICE)
            focus_2 = focus_2.to(DEVICE)

            output = net(focus_1, focus_2).to(DEVICE)

            loss = grad_loss(focus_1, output) + grad_loss(focus_2, output) + \
                   ints_loss(focus_1, output) + ints_loss(focus_2, output)

            epoch_loss += loss.item()
            iter_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (idx + 1) % 50:
                print('Iteration: %08d itera loss: %.4f' % (itera, iter_loss))
                itera += 50
                iter_loss = 0

        with torch.no_grad():
            for idx, data in enumerate(datas_test):
                focus_1, focus_2, _ = data
                focus_1 = focus_1.to(DEVICE)
                focus_2 = focus_2.to(DEVICE)

                output = net(focus_1, focus_2).to(DEVICE)

                epoch_ssim += SSIM(output, focus_1).item() + SSIM(output, focus_2).item()

        print('Epoch: %03d, epoch loss: %.4f, mean SSIM: %.4f, time: %6.2fsec' %
              (epoch, epoch_loss, epoch_ssim / len(datas_test), time() - st_time))

        if epoch_loss < max_loss:
            max_loss = epoch_loss
            torch.save(net, './pth/model.pth')


if __name__ == '__main__':
    net = PMGI()
    import torchstat
    torchstat.stat(net, (1, 256, 256), arg_num=2)
    # train()
