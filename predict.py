import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import math
import numpy as np
from PIL import Image

from utils.saver import Saver
from model.trainer_sr import SRModel
from options import TestOptions
from utils.utility import timer, calc_psnr, quantize, load_file, get_gaussian
from data.dataset import unpaired_dataset, paired_dataset
from model.trainer_down import AdaptiveDownsamplingModel
# from fvcore.nn import FlopCountAnalysis, flop_count_table

"""
test_sr: 
--test_mode sr_patch --name scale_4x --scale 4 --crop 480 --pretrain_sr ./experiment/ckpt/scale_4x/model_sr_0110.pth --test_lr Capsule_Data/TestSet/side_480 --gpu cuda:6 --sr_model endosr --training_type endosr --save_results --realsr
--test_mode sr --name scale_2x --scale 2 --crop 480 --pretrain_sr ./experiment/ckpt/scale_2x/model_sr_0120.pth --test_lr Capsule_Data/Si_De/TestSet --gpu cuda:3 --sr_model endosr --training_type endosr --save_results --realsr
test_down: 
--test_mode down --name down_x4 --scale 4 --resume_down ./experiment/ckpt/scale_4x/model_down_0110.pth --patch_size_down 512 --test_range 1-2000 --gpu cuda:6
"""


def test_sr(args):
    # test mode
    saver = Saver(args, test=True)

    # daita loader
    dataset = paired_dataset(args)
    test_loader_sr = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.nThreads)

    # model
    SRM = SRModel(args, train=False)
    if (args.resume_sr is None) and (args.pretrain_sr is None):
        raise NotImplementedError('put pretrained model for test')
    elif args.resume_sr is not None:
        _, _ = SRM.resume(args.resume_sr, train=False)
    print('load model successfully!')

    # print(SRM)
    # SRM.eval()
    # flops = FlopCountAnalysis(SRM.update_img, torch.rand(1, 3, args.crop, args.crop))
    # print(flop_count_table(flops))

    eval_timer_sr = timer()

    eval_timer_sr.tic()
    ep0 = 100
    psnr_sum = 0
    cnt = 0
    with torch.no_grad():
        for img_hr, img_lr, fn in tqdm(test_loader_sr, ncols=80):
            img_hr, img_lr = img_hr.to(args.gpu), img_lr.to(args.gpu)
            if args.precision == 'half':
                img_lr = img_lr.half()
            # b, c, w, h = img_lr.shape
            # input_lr = torch.zeros((b, c, 366, 366)).to(args.gpu)
            # input_lr[:b, :c, 15:w+15, 15:h+15] = img_lr
            # img_lr = input_lr

            SRM.update_img(img_lr)
            SRM.generate_HR()

            # img_sr = quantize(SRM.img_gen[:b, :c, 15*args.scale:w*args.scale+15*args.scale, 15*args.scale:h*args.scale+15*args.scale]) # origin
            # img_sr = SRM.img_gen[:b, :c, :w * args.scale, :h * args.scale].clamp(0, 1)
            img_sr = quantize(SRM.img_gen)

            # if args.realsr:
            #     lr_resize = F.interpolate(img_lr[:1], scale_factor=int(args.scale), mode='bicubic', align_corners=True)
            #     img_sr = torch.cat((lr_resize, img_sr[:1]), dim=0)

            if args.save_results:
                saver.write_img_SR(ep0, img_sr, fn)

            if not args.realsr:
                psnr_sum += calc_psnr(img_sr, img_hr, args.scale, rgb_range=1)
                cnt += 1
    eval_timer_sr.hold()
    if not args.realsr:
        print('PSNR on test set: %.04f, %.01fs' % (psnr_sum/(cnt), eval_timer_sr.release()))


def test_sr_patch(args):
    # test mode
    saver = Saver(args, test=True)
    result_path = os.path.join(saver.image_sr_dir, 'sr_results')
    os.makedirs(result_path, exist_ok=True)

    # daita loader
    gaussian_importance_map = get_gaussian([args.crop * args.scale, args.crop * args.scale], sigma_scale=1. / 8)
    filenames = os.listdir(os.path.join(args.test_dataroot, args.test_lr))
    filenames.sort()

    # model
    SRM = SRModel(args, train=False)
    if (args.resume_sr is None) and (args.pretrain_sr is None):
        raise NotImplementedError('put pretrained model for test')
    elif args.resume_sr is not None:
        _, _ = SRM.resume(args.resume_sr, train=False)
    print('load model successfully!')

    SRM.eval()
    with torch.no_grad():
        for filename in tqdm(filenames):
            img_name = os.path.join(args.test_dataroot, args.test_lr, filename)
            image, name, point = load_file(img_name, args.crop, stride=args.crop//2)
            image = np.transpose(image, (2, 0, 1))
            C, H, W = image.shape
            predict_array = np.zeros((C, H * args.scale, W * args.scale))
            repeat_array = np.zeros((C, H * args.scale, W * args.scale))
            num = math.floor(len(point) / args.batch_size)
            num_remain = len(point) % args.batch_size
            if num_remain > 0:
                all_num = num + 1
            else:
                all_num = num
            # print(num, num_remain)
            for p in range(all_num):
                if p < num:
                    for b in range(args.batch_size):
                        pos = p * args.batch_size + b
                        x_left, x_right, y_up, y_down = point[pos][0], point[pos][1], point[pos][2], point[pos][3]
                        lr = np.copy(image[:, x_left:x_right, y_up:y_down])[np.newaxis, :, :, :]
                        if b == 0:
                            img = lr
                        else:
                            img = np.concatenate([img, lr], axis=0)
                else:
                    for b in range(num_remain):
                        idx = b + (num - 1) * args.batch_size
                        x_left, x_right, y_up, y_down = point[idx][0], point[idx][1], point[idx][2], point[idx][3]
                        lr = np.copy(image[:, x_left:x_right, y_up:y_down])[np.newaxis, :, :, :]
                        if b == 0:
                            img = lr
                        else:
                            img = np.concatenate([img, lr], axis=0)

                img = torch.from_numpy(img).float().to(args.gpu)
                batch, _, _, _ = img.shape

                SRM.update_img(img)
                SRM.generate_HR()
                recon_hr = SRM.img_gen.cpu().detach().numpy()
                batch, _, _, _ = recon_hr.shape
                for _batch in range(batch):
                    pre_x = recon_hr[_batch]
                    pos = p * args.batch_size + _batch
                    x_left, x_right, y_up, y_down = point[pos][0], point[pos][1], point[pos][2], point[pos][3]
                    predict_array[:, x_left * args.scale:x_left * args.scale + args.crop * args.scale,
                    y_up * args.scale:y_up * args.scale + args.crop * args.scale] += pre_x * gaussian_importance_map
                    repeat_array[:, x_left * args.scale:x_left * args.scale + args.crop * args.scale, y_up * args.scale:y_up * args.scale + args.crop * args.scale] += gaussian_importance_map
            repeat_array = repeat_array.astype(np.float)
            predict_array = predict_array / repeat_array

            recon_hr = np.transpose(predict_array, (1, 2, 0))
            recon_hr = (recon_hr * 255).clip(0, 255).astype(np.uint8)
            recon_hr = Image.fromarray(recon_hr)
            recon_hr.save(f"{result_path}/{name}.jpg")

            # if args.realsr:
            #     lr_resize = F.interpolate(img_lr[:1], scale_factor=int(args.scale), mode='bicubic', align_corners=True)
            #     img_sr = torch.cat((lr_resize, img_sr[:1]), dim=0)


def test_down(args):
    # daita loader
    print('\nmaking dataset ...')
    dataset = unpaired_dataset(args, phase='test')  #
    test_loader_down = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.nThreads)

    # model
    print('\nmaking model ...')
    ADM = AdaptiveDownsamplingModel(args)

    if args.resume_down is None:
        raise NotImplementedError('put trained downsampling model for testing')
    else:
        ep0, total_it = ADM.resume(args.resume_down, train=False)
    ep0 += 1
    print('load model successfully!')

    saver = Saver(args, test=True)
    print('\ntest start ...')
    ADM.eval()
    with torch.no_grad():
        for number, (img_s, _, fn) in enumerate(test_loader_down):
            # if (number + 1) % (len(test_loader_down) // 10) == 0:
            #     print('[{:05d} / {:05d}] ...'.format(number + 1, len(test_loader_down)))
            ADM.update_img(img_s)
            ADM.generate_LR()
            saver.write_img_LR(8, (number + 1), ADM, args, fn)
    print('\ntest done!')


if __name__ == '__main__':
    # parse options
    parser = TestOptions()
    args = parser.parse()
    args.batch_size = 1

    if args.test_mode == 'sr':
        test_sr(args)
    elif args.test_mode == 'sr_patch':
        test_sr_patch(args)
    else:
        test_down(args)
