import torch
import torch.nn as nn

class PMGI(nn.Module):
    def __init__(self):
        super(PMGI, self).__init__()
        # Gradient Path
        self.conv_G_1 = nn.Sequential(
            nn.Conv2d(kernel_size=5, stride=1, in_channels=3, out_channels=16, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_G_2 = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1, in_channels=16, out_channels=16, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_G_3 = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1, in_channels=48, out_channels=16, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_G_4 = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1, in_channels=64, out_channels=16, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        # Intensity Path
        self.conv_I_1 = nn.Sequential(
            nn.Conv2d(kernel_size=5, stride=1, in_channels=3, out_channels=16, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_I_2 = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1, in_channels=16, out_channels=16, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_I_3 = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1, in_channels=48, out_channels=16, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_I_4 = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1, in_channels=64, out_channels=16, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        # Pathwise Transfer Block 1
        self.transBlock_1_1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, in_channels=32, out_channels=16),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.transBlock_1_2 = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, in_channels=32, out_channels=16),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        # Pathwise Transfer Block 2
        self.transBlock_2_1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, in_channels=32, out_channels=16),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.transBlock_2_2 = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, in_channels=32, out_channels=16),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        # Output Block
        self.outputBlock = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, in_channels=128, out_channels=1),
            nn.Tanh()
        )

    def forward(self, x_1, x_2):
        grad_input = torch.cat((x_1, x_1, x_2), dim=1)
        grad_1 = self.conv_G_1(grad_input)
        grad_2 = self.conv_G_2(grad_1)
        ints_input = torch.cat((x_1, x_2, x_2), dim=1)
        ints_1 = self.conv_I_1(ints_input)
        ints_2 = self.conv_I_2(ints_1)

        trans_1_1 = self.transBlock_1_1(torch.cat((grad_2, ints_2), dim=1))
        trans_1_2 = self.transBlock_1_2(torch.cat((grad_2, ints_2), dim=1))

        grad_3 = self.conv_G_3(torch.cat((grad_1, grad_2, trans_1_1), dim=1))
        ints_3 = self.conv_I_3(torch.cat((ints_1, ints_2, trans_1_2), dim=1))

        trans_2_1 = self.transBlock_2_1(torch.cat((grad_3, ints_3), dim=1))
        trans_2_2 = self.transBlock_2_2(torch.cat((grad_3, ints_3), dim=1))

        grad_4 = self.conv_G_4(torch.cat((grad_1, grad_2, grad_3, trans_2_1), dim=1))
        ints_4 = self.conv_I_4(torch.cat((ints_1, ints_2, ints_3, trans_2_2), dim=1))

        fusion_feature = torch.cat((grad_1, grad_2, grad_3, grad_4, ints_1, ints_2, ints_3, ints_4), dim=1)

        fused_image = self.outputBlock(fusion_feature)

        return fused_image
