import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torchvision.models.vgg import vgg19

from model import basic_model as model
from model.networks import Discriminator_model
# from fvcore.nn import FlopCountAnalysis, flop_count_table

class SRModel(nn.Module):
    def __init__(self, args, train=True):
        super(SRModel, self).__init__()

        self.args = args
        self.scale = args.scale
        self.gpu = args.gpu
        self.error_last = 1e8

        self.phase = args.phase
        self.training_type = args.training_type
        print('sr model : ', self.args.sr_model)
        print('training type : ', self.training_type)

        # define model, optimizer, scheduler, loss
        self.gen = model.Model(args)
        # print(self.gen)
        # self.gen.eval()
        # flops = FlopCountAnalysis(self.gen, torch.rand(1, 3, 224, 224).to(self.gpu))
        # print(flop_count_table(flops))

        if args.pretrain_sr is not None:
            checkpoint = torch.load(args.pretrain_sr, map_location=lambda storage, loc: storage)['gen']
            self.gen.load_state_dict(checkpoint)
            print('Load pretrained SR model from {}'.format(args.pretrain_sr))

        if train:
            self.gen_opt = Adam(self.gen.parameters(), lr=args.lr_sr, betas=(0.9, 0.999))
            self.gen_sch = torch.optim.lr_scheduler.StepLR(self.gen_opt, args.decay_batch_size_sr, gamma=0.5)  # args.gamma)

            self.content_criterion = nn.L1Loss().to(self.gpu)

            if self.training_type == 'endosr':
                self.dis = Discriminator_model(scale=int(args.scale)).to(self.gpu)
                self.dis_opt = Adam(self.dis.parameters(), lr=args.lr_sr, betas=(0.9, 0.999))
                self.dis_sch = torch.optim.lr_scheduler.StepLR(self.dis_opt, args.decay_batch_size_sr, gamma=0.1)

                self.GAN_Loss = GANLoss().to(self.gpu)
                self.Wavelet_Loss = DWT_LOSS().to(self.gpu)
                self.dis.train()

            self.gen.train()

        self.gen_loss = 0
        self.recon_loss = 0

    ## update images to the model
    def update_img(self, lr, hr=None, lr_t=None):
        self.img_lr = lr
        self.img_hr = hr
        if lr_t is not None:
            self.img_t = lr_t.to(self.gpu).detach()

        self.gen_loss = 0
        self.recon_loss = 0

    def generate_HR(self):
        # self.img_lr *= 255
        if self.training_type == 'endosr' and self.phase == 'train':
            self.feature_s, self.img_gen = self.gen(self.img_lr, 0)
            self.feature_t, _ = self.gen(self.img_t, 0)
        elif self.training_type == 'endosr' and self.phase == 'test':
            self.feature_s, self.img_gen = self.gen(self.img_lr, 0)
        else:
            self.img_gen = self.gen(self.img_lr, 0)
        # self.img_gen /= 255

    def update_G(self):
        # EDSR style
        if self.training_type == 'edsr':
            self.gen_opt.zero_grad()

            self.recon_loss = self.content_criterion(self.img_gen, self.img_hr) * 255.0  # compensate range of 0 to 1

            self.recon_loss.backward()
            self.gen_opt.step()
            self.gen_loss = self.recon_loss.item()
            self.gen_sch.step()

        elif self.training_type == 'endosr':
            if self.args.cycle_recon:
                raise NotImplementedError('Do not support using cycle reconstruction loss in EndoSR training')

            # training generator
            self.gen_opt.zero_grad()

            score_fake = self.dis(self.feature_s)
            score_real = self.dis(self.feature_t)

            adversarial_loss_rf = self.GAN_Loss(score_fake, 'mean')
            adversarial_loss_fr = self.GAN_Loss(score_real, 'mean')
            adversarial_loss = adversarial_loss_fr + adversarial_loss_rf

            content_loss = self.content_criterion(self.img_gen, self.img_hr)  # compensate range of 0 to 1
            wavelet_loss = self.Wavelet_Loss(self.img_gen, self.img_hr)

            gen_loss = adversarial_loss * self.args.adv_w + wavelet_loss * self.args.wav_w + content_loss * self.args.con_w

            gen_loss.backward()
            self.gen_loss = gen_loss.item()
            self.gen_opt.step()

            # training discriminator
            self.dis_opt.zero_grad()

            score_real = self.dis(self.feature_s.detach())
            score_fake = self.dis(self.feature_t.detach())

            adversarial_loss_rf = self.GAN_Loss(score_real, 'real')
            adversarial_loss_fr = self.GAN_Loss(score_fake, 'fake')
            discriminator_loss = adversarial_loss_fr + adversarial_loss_rf

            discriminator_loss.backward()
            self.dis_opt.step()

            self.gen_sch.step()
            self.dis_sch.step()
        else:
            raise NotImplementedError('training type is not possible')

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)
        self.gen.load_state_dict(checkpoint['gen'])
        if train:
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
            if self.training_type == 'endosr':
                self.dis.load_state_dict(checkpoint['dis'])
                self.dis_opt.load_state_dict(checkpoint['dis_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def state_save(self, filename, ep, total_it):
        state = {'gen': self.gen.state_dict(),
                 'gen_opt': self.gen_opt.state_dict(),
                 'ep': ep,
                 'total_it': total_it
                 }
        if self.training_type == 'endosr':
            state['dis'] = self.dis.state_dict()
            state['dis_opt'] = self.dis_opt.state_dict()
        time.sleep(5)
        torch.save(state, filename)
        return

    def model_save(self, filename, ep, total_it):
        state = {'gen': self.gen.state_dict()}
        if self.training_type == 'endosr':
            state['dis'] = self.dis.state_dict()
        time.sleep(5)
        torch.save(state, filename)
        return


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.L1Loss()  # nn.L1Loss
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real == 'real':
            target_tensor = self.real_label
        elif target_is_real == 'fake':
            target_tensor = self.fake_label
        else:
            target_tensor = (self.real_label + self.fake_label) / 2
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DWT_LOSS(nn.Module):
    def __init__(self):
        super(DWT_LOSS, self).__init__()
        self.requires_grad = False

    def dwt_init(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, x_HL, x_LH, x_HH

    def forward(self, x, target):
        x_LL1, x_HL1, x_LH1, x_HH1 = self.dwt_init(x)
        y_LL1, y_HL1, y_LH1, y_HH1 = self.dwt_init(target)
        HL1_loss = F.l1_loss(x_HL1, y_HL1)
        LH1_loss = F.l1_loss(x_LH1, y_LH1)
        HH1_loss = F.l1_loss(x_HH1, y_HH1)
        x_LL2, x_HL2, x_LH2, x_HH2 = self.dwt_init(x_LL1)
        y_LL2, y_HL2, y_LH2, y_HH2 = self.dwt_init(y_LL1)
        HL2_loss = F.l1_loss(x_HL2, y_HL2)
        LH2_loss = F.l1_loss(x_LH2, y_LH2)
        HH2_loss = F.l1_loss(x_HH2, y_HH2)
        x_LL3, x_HL3, x_LH3, x_HH3 = self.dwt_init(x_LL2)
        y_LL3, y_HL3, y_LH3, y_HH3 = self.dwt_init(y_LL2)
        HL3_loss = F.l1_loss(x_HL3, y_HL3)
        LH3_loss = F.l1_loss(x_LH3, y_LH3)
        HH3_loss = F.l1_loss(x_HH3, y_HH3)
        return 0.12 * HL2_loss + 0.12 * LH2_loss + 0.12 * HH2_loss + 0.05 * HL3_loss + 0.05 * LH3_loss + 0.05 * HH3_loss + 0.16 * LH1_loss + 0.16 * HL1_loss + 0.16 * HH1_loss