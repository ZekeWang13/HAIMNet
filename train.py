
import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from data.options import option
# from measure import metrics
metrics = None
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 引入

opt = option()
from measure import metrics

# TensorBoard writer 初始化
os.makedirs(opt.checkpoint_dir, exist_ok=True)
writer = SummaryWriter(os.path.join(opt.checkpoint_dir, 'runs'))

def group_params_for_saffm(model):
    base, saffm = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = n.lower()
        if 'saffm' in lname:   # 命中 saffm_i3 / saffm_h3 等
            saffm.append(p)
        else:
            base.append(p)
    return base, saffm

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_init():
    seed_torch();
    cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
      
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        if iter == 0 and opt.lol_v1:
            print(f"[DEBUG] opt.gamma={opt.gamma}")
            print(f"[DEBUG] first pair: {path1[0]}  <->  {path2[0]}")

            import cv2, numpy as np
            from skimage.metrics import structural_similarity as sk_ssim
            a = (im1[0].detach().cpu().clamp(0,1).numpy().transpose(1,2,0)*255).astype(np.uint8)
            b = (im2[0].detach().cpu().clamp(0,1).numpy().transpose(1,2,0)*255).astype(np.uint8)
            y1 = cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)[...,0]
            y2 = cv2.cvtColor(b, cv2.COLOR_RGB2YCrCb)[...,0]
            ssim_y = sk_ssim(y1, y2, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            print(f"[DEBUG] input-pair SSIM(Y, σ=1.5) ≈ {ssim_y:.3f}")
        if iter == 0 and opt.SICE_mix:
            print(f"[DEBUG] opt.gamma={opt.gamma}")

            in_name = str(path1[0])  # 例如 '5.JPG'
            gt_name = str(path2[0])  # 例如 '123.JPG'

            # SICE 目录
            train_root = opt.data_train_SICE
            label_root = os.path.join(os.path.dirname(opt.data_train_SICE), 'label')

            scene_gt = os.path.splitext(gt_name)[0]  # '123'
            expected_in = os.path.join(train_root, scene_gt, in_name)

            if os.path.exists(expected_in):
                in_full = expected_in
                scene_in = scene_gt
                note = "OK: expected input exists under the same scene"
            else:
                import glob
                candidates = glob.glob(os.path.join(train_root, '*', in_name))
                in_full = f"(NOT under scene {scene_gt})"
                scene_in = "?"
                note = f"NOT FOUND under scene {scene_gt}; candidates={len(candidates)}"
                show = candidates[:5]
                if show:
                    print("[DEBUG] candidate inputs (first few):")
                    for p in show:
                        print("        ", p)

            gt_full = os.path.join(label_root, gt_name)

            print(f"[DEBUG] scene id (input_gt): {scene_in}  _  {scene_gt}   [{note}]")
            print(f"[DEBUG] first pair: {in_full}  <->  {gt_full}")

        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)
        else:
            output_rgb = model(im1)

        gt_rgb = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)

        p_loss_hvi = P_loss(output_hvi, gt_hvi)
        p_loss_rgb = P_loss(output_rgb, gt_rgb)

        loss_hvi = (L1_loss(output_hvi, gt_hvi)
                    + D_loss(output_hvi, gt_hvi)
                    + E_loss(output_hvi, gt_hvi)
                    + (opt.P_weight * p_loss_hvi[0] if p_loss_hvi[0] is not None else 0))
        loss_rgb = (L1_loss(output_rgb, gt_rgb)
                    + D_loss(output_rgb, gt_rgb)
                    + E_loss(output_rgb, gt_rgb)
                    + (opt.P_weight * p_loss_rgb[0] if p_loss_rgb[0] is not None else 0))

        loss = loss_rgb + opt.HVI_weight * loss_hvi
        iter += 1

        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_print += loss.item()
        loss_last_10 += loss.item()
        pic_cnt += 1
        pic_last_10 += 1
        if iter == train_len:
            print(
                f"===> Epoch[{epoch}]: Loss: {loss_last_10 / pic_last_10:.4f} || Learning rate: lr={optimizer.param_groups[0]['lr']}.")
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = transforms.ToPILImage()(output_rgb[0].squeeze(0))
            gt_img = transforms.ToPILImage()(gt_rgb[0].squeeze(0))
            if not os.path.exists(opt.val_folder + 'training'):
                os.makedirs(opt.val_folder + 'training', exist_ok=True)
            output_img.save(opt.val_folder + 'training/test.png')
            gt_img.save(opt.val_folder + 'training/gt.png')
    return loss_print, pic_cnt


def checkpoint(epoch):
    # 用 opt.checkpoint_dir
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    model_out_path = os.path.join(opt.checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_out_path)
    print(f"Checkpoint saved to {model_out_path}")
    return model_out_path


def load_datasets():
    print('===> Loading datasets')
    if not any([opt.lol_v1, opt.lol_blur, opt.lolv2_real, opt.lolv2_syn, opt.SID, opt.SICE_mix, opt.SICE_grad]):
        raise Exception("should choose a dataset")
    if opt.lol_v1:
        train_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize)
        training_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_lol_v1)
        testing_loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif opt.lol_blur:
        train_set = get_training_set_blur(opt.data_train_lol_blur, size=opt.cropSize)
        training_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_lol_blur)
        testing_loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif opt.lolv2_real:
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real, size=opt.cropSize)
        training_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_lolv2_real)
        testing_loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif opt.lolv2_syn:
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, size=opt.cropSize)
        training_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_lolv2_syn)
        testing_loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif opt.SID:
        train_set = get_SID_training_set(opt.data_train_SID, size=opt.cropSize)
        training_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_SID)
        testing_loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif opt.SICE_mix:
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        training_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        testing_loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:  # opt.SICE_grad
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        training_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        testing_loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return training_loader, testing_loader


def build_model():
    print('===> Building model ')
    model = CIDNet().cuda()
    if opt.start_epoch > 0:
        pth = os.path.join(opt.checkpoint_dir, f"epoch_{opt.start_epoch}.pth")
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model


def make_scheduler():
    base_params, saffm_params = group_params_for_saffm(model)

    optimizer = optim.Adam([
        {'params': base_params, 'lr': opt.lr,       'name': 'base'},
        {'params': saffm_params,'lr': opt.saffm_lr, 'name': 'saffm'},
    ])

    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[(opt.nEpochs // 4) - opt.warmup_epochs, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001]
            )
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs,
                                               after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[opt.nEpochs // 4, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001]
            )
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch],
                restart_weights=[1],
                eta_min=1e-7
            )
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs,
                                               after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.start_epoch],
                restart_weights=[1],
                eta_min=1e-7
            )
    else:
        raise Exception("should choose a scheduler")
    return optimizer, scheduler


def init_loss():
    L1_loss = L1Loss(loss_weight=opt.L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=opt.D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=opt.E_weight).cuda()
    P_loss = PerceptualLoss(
        {'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
        perceptual_weight=opt.P_weight, criterion='mse'
    ).cuda()
    return L1_loss, P_loss, E_loss, D_loss



from datetime import datetime

if __name__ == '__main__':
    # preparation
    train_init()

    if opt.lol_v1:
        dataset_name = "LOLv1"
    elif opt.lolv2_real:
        dataset_name = "LOLv2Real"
    elif opt.lolv2_syn:
        dataset_name = "LOLv2Syn"
    elif opt.lol_blur:
        dataset_name = "LOLBlur"
    elif opt.SID:
        dataset_name = "SID"
    elif opt.SICE_mix:
        dataset_name = "SICEMix"
    elif opt.SICE_grad:
        dataset_name = "SICEGrad"
    else:
        dataset_name = "UnknownDataset"
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    opt_file = os.path.join(opt.checkpoint_dir, f"train_options_{dataset_name}_{now}.txt")
    with open(opt_file, "w") as f:
        f.write(f"====== Training Options ({dataset_name}) ======\n")
        for k, v in vars(opt).items():
            f.write(f"{k}: {v}\n")

    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer, scheduler = make_scheduler()
    L1_loss, P_loss, E_loss, D_loss = init_loss()

    start_epoch = opt.start_epoch if opt.start_epoch > 0 else 0
    if not os.path.exists(opt.val_folder):
        os.makedirs(opt.val_folder, exist_ok=True)

    psnr, ssim, lpips, psnr_gt, ssim_gt, lpips_gt= [], [], [], [], [], []
    for epoch in range(start_epoch + 1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        avg_train_loss = epoch_loss / pic_num
        writer.add_scalar('train/loss', avg_train_loss, epoch)

        scheduler.step()

        base_lr = next(g['lr'] for g in optimizer.param_groups if g.get('name') == 'base')
        saffm_lr = next(g['lr'] for g in optimizer.param_groups if g.get('name') == 'saffm')
      
        writer.add_scalar('train/lr_base', base_lr, epoch)
        writer.add_scalar('train/lr_saffm', saffm_lr, epoch)

        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch)
            norm_size = True

            if opt.lol_v1:
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            elif opt.lolv2_real:
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            elif opt.lolv2_syn:
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            elif opt.lol_blur:
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
            elif opt.SID:
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            elif opt.SICE_mix:
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            else:  # opt.SICE_grad
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False

            im_dir = os.path.join(opt.val_folder, output_folder + '*.png')
            eval(
                model, testing_data_loader, model_out_path,
                opt.val_folder + output_folder,
                norm_size=norm_size, LOL=opt.lol_v1,
                v2=opt.lolv2_real, alpha=0.8
            )

            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            avg_psnr_gt, avg_ssim_gt, avg_lpips_gt = metrics(im_dir, label_dir, use_GT_mean=True)
            print(f"===> Avg.PSNR: {avg_psnr:.4f} dB ")
            print(f"===> Avg.SSIM: {avg_ssim:.4f} ")
            print(f"===> Avg.LPIPS: {avg_lpips:.4f} ")
            print(f"===> Avg.PSNR (GT mean): {avg_psnr_gt:.4f} dB ")
            print(f"===> Avg.SSIM  (GT mean): {avg_ssim_gt:.4f} ")
            print(f"===> Avg.LPIPS (GT mean): {avg_lpips_gt:.4f} ")
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            psnr_gt.append(avg_psnr_gt)
            ssim_gt.append(avg_ssim_gt) 
            lpips_gt.append(avg_lpips_gt)

            writer.add_scalar('val/PSNR', avg_psnr, epoch)
            writer.add_scalar('val/SSIM', avg_ssim, epoch)
            writer.add_scalar('val/LPIPS', avg_lpips, epoch)
            writer.add_scalar('val/PSNR_GTmean', avg_psnr_gt, epoch)
            writer.add_scalar('val/SSIM_GTmean', avg_ssim_gt, epoch)
            writer.add_scalar('val/LPIPS_GTmean', avg_lpips_gt, epoch)
            torch.cuda.empty_cache()

    writer.close()

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.makedirs("./results/training", exist_ok=True)
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("dataset: " + output_folder + "\n")
        f.write(f"lr: {opt.lr}\n")
        f.write(f"batch size: {opt.batchSize}\n")
        f.write(f"crop size: {opt.cropSize}\n")
        f.write(f"HVI_weight: {opt.HVI_weight}\n")
        f.write(f"L1_weight: {opt.L1_weight}\n")
        f.write(f"D_weight: {opt.D_weight}\n")
        f.write(f"E_weight: {opt.E_weight}\n")
        f.write(f"P_weight: {opt.P_weight}\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS | PSNR_GT | SSIM_GT | LPIPS_GT |\n")
        f.write("|--------|------|------|--------|----------|----------|----------|\n")

        for i in range(len(psnr)):
            epoch_id = opt.start_epoch + (i + 1) * opt.snapshots
            f.write(f"| {epoch_id} | {psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} | {psnr_gt[i]:.4f} | {ssim_gt[i]:.4f} | {lpips_gt[i]:.4f} |\n")
