# -*- ecoding: utf-8 -*-
# @ModuleName: eval
# @Author: BetaHu
# @Time: 2021/7/4 21:37
import torch
from torch.utils.data import DataLoader
from dataset import MFI_WHU, Lytro
import matplotlib.pyplot as plt
from metric import Q_ABF, ssim

DEVICE = 'cpu'

net = torch.load('./pth/PMGI.pth').to(DEVICE)

dataset = Lytro('E:/Database/LytroDataset')  # MFI_WHU('./MFI-WHU/test')
datas = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

mean_ssim = []
mean_qabf = []

for idx, data in enumerate(datas):
    focus_1, focus_2 = data

    output = net(focus_1.to(DEVICE), focus_2.to(DEVICE)).to(DEVICE)
    mean_ssim.append((ssim(focus_1, output).item() + ssim(focus_2, output).item()) / 2)
    mean_qabf.append(Q_ABF(focus_1, focus_2, output).item())

    print('%.4f, %.4f' % (mean_ssim[-1], mean_qabf[-1]))

    # plt.subplot(131)
    # plt.imshow(focus_1.squeeze().detach().numpy(), cmap='gray')
    # plt.axis('off')
    # plt.subplot(132)
    # plt.imshow(focus_2.squeeze().detach().numpy(), cmap='gray')
    # plt.axis('off')
    # plt.subplot(133)
    # plt.imshow(output.squeeze().detach().numpy(), cmap='gray')
    # plt.axis('off')
    # plt.show()

print('average ssim: %.4f, average Q_ABF: %.4f' % (sum(mean_ssim) / len(mean_ssim), sum(mean_qabf) / len(mean_qabf)))
plt.subplot(121)
plt.plot(list(range(len(mean_ssim))), mean_ssim, '->b')
plt.title('SSIM')
plt.subplot(122)
plt.plot(list(range(len(mean_qabf))), mean_qabf, '-<r')
plt.title('Q_ABF')
plt.show()
