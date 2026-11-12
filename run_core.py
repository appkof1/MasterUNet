"""Core experiments for the R3 revision (retraining with the fixed code).

Configs
  masterunet         full two-stage MasterUNet with preprocessing
  masterunet_noprep  MasterUNet without the preprocessing pipeline (raw 1-channel input)
  unet               baseline U-Net (Ronneberger et al., with batch norm), raw input, Dice+BCE loss
  prep_*             Table 7: MasterUNet with a partial preprocessing pipeline (ad, clahe, edge, ad_clahe)
  loss_*             Table 8: MasterUNet with a simpler loss (dice, dice_bce, dice_bce_ft = fixed weights)
  no_*               Table 10: MasterUNet without augmentation / dropout / attention gates / self-attention
  unetpp, aaunet, mffaunet   Tables 9 and 11: re-implemented baselines, raw input, Dice+BCE loss

Protocol (identical for every config and seed)
  * BUSI, benign capped at 300 (fixed subset) -> 643 images.
  * Fixed stratified 80/20 train/test split (random_state=0): 514 train / 129 test.
  * 10% of the training split held out (stratified, random_state=0) for validation:
    used ONLY for selecting the best checkpoint and choosing the decision threshold.
  * Full schedule (100 / 75 epochs); the best-validation-Dice checkpoint is kept.
  * Seeds change weight initialisation, augmentation and batch order.
  * Batch 8 with 2-step gradient accumulation (effective batch 16).
  * Dice / IoU / precision / recall / pixel accuracy are computed per image on the
    benign + malignant test images (normal images have empty masks and are excluded),
    then averaged. Normal-image false-positive rate is reported separately.
  * Unseen evaluation: fixed 20% split of BrEaST (random_state=0), never trained on.

Usage:  python run_core.py --config masterunet --seed 0
"""
import argparse, copy, json, os, random, sys, time
import multiprocessing as mp
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from models.classicUnetBN import UNetBN  # plain classicUnet collapses (val Dice 0), see models/classicUnetBN.py
from models.two_stage_unet_aux import MasterUNetStage1, MasterUNetStage2
from utils.data_loader import BUSIDataset, BrEaSTLesionsDataset
from utils.losses import CombinedLossStage1, CombinedLossStage2, DiceLoss, FocalTverskyLoss, BoundaryLoss
from utils.preprocessing import preprocess, preprocess_variant, PREPROC_VARIANTS
from models.baselines import UNetPP, AAUNet, MFFAUNet
import torch.nn.functional as F

CONFIGS = {
    'masterunet': dict(prep='full'),
    'masterunet_noprep': dict(prep=None),
    'unet': dict(baseline=UNetBN),
    'prep_ad': dict(prep='ad'), 'prep_clahe': dict(prep='clahe'), 'prep_edge': dict(prep='edge'),
    'prep_ad_clahe': dict(prep='ad_clahe'),
    'loss_dice': dict(prep='full', loss='dice'), 'loss_dice_bce': dict(prep='full', loss='dice_bce'),
    'loss_dice_bce_ft': dict(prep='full', loss='dice_bce_ft'),
    'no_aug': dict(prep='full', aug=False), 'no_dropout': dict(prep='full', model=dict(dropout=False)),
    'no_attention': dict(prep='full', model=dict(attention_gates=False)),
    'no_bottleneck': dict(prep='full', model=dict(self_attention=False)),
    # batch 8 exceeds the 4 GB GPU for these three; batch 4 x 4 accumulation keeps the effective batch at 16
    'unetpp': dict(baseline=UNetPP, batch=(4, 4)), 'aaunet': dict(baseline=AAUNet, batch=(4, 4)),
    'mffaunet': dict(baseline=MFFAUNet, batch=(4, 4)),
}

# ---------------------------------------------------------------- settings
STAGE1_EPOCHS, STAGE2_EPOCHS = 100, 75
BATCH, ACCUM = 8, 2
LR, WEIGHT_DECAY = 1e-4, 1e-4
# Train the full cosine schedule and keep the best-validation checkpoint. Early termination
# (patience) is disabled: stopping mid-schedule leaves the learning rate un-annealed, and the
# small validation set (~41 tumour images) makes patience-based stopping noise-driven.
PATIENCE = None
BENIGN_CAP = 300
SPLIT_SEED = 0
NUM_WORKERS = 2
THRESHOLDS = np.round(np.linspace(0.1, 0.9, 17), 2)

# ---------------------------------------------------------------- augmentation (as train.py)
complete_aug = [
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=15, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
    A.ElasticTransform(p=0.3), A.GridDistortion(p=0.3), A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)]
weak_aug = complete_aug[:5]
strong_aug = [A.ElasticTransform(p=0.1), A.GridDistortion(p=0.1),
              A.GaussNoise(var_limit=(10.0, 25.0), p=0.2)] + complete_aug[:5]
COMPLETE = A.Compose(complete_aug, additional_targets={'mask': 'mask'})
WEAK = A.Compose(weak_aug, additional_targets={'mask': 'mask'})
STRONG = A.Compose(strong_aug, additional_targets={'mask': 'mask'})
VAL = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)], additional_targets={'mask': 'mask'})


def stage1_aug(epoch):
    return WEAK if epoch < 20 else STRONG if epoch < 50 else COMPLETE


def stage2_aug(epoch):
    return STRONG if epoch < 15 else COMPLETE


def no_aug(epoch):
    return VAL


class AblationLoss(nn.Module):
    """Simpler losses for Table 8; same main/auxiliary structure as the proposed loss.
    dice: Dice only. dice_bce: 0.5 Dice + 0.5 BCE. dice_bce_ft: the final-phase weights of the
    proposed schedule applied from the first epoch (Stage 2 keeps the 0.1 boundary term)."""

    def __init__(self, kind, stage):
        super().__init__()
        self.kind, self.stage = kind, stage
        self.dice, self.bce = DiceLoss(), nn.BCEWithLogitsLoss()
        self.ft = FocalTverskyLoss(alpha=0.2, beta=0.8, gamma=1.0) if stage == 1 else             FocalTverskyLoss(alpha=0.4, beta=0.6, gamma=1.0)
        self.boundary = BoundaryLoss()
        self.w_aux = 0.6 if stage == 1 else 0.4

    def term(self, logits, masks):
        d = self.dice(torch.sigmoid(logits), masks)
        if self.kind == 'dice':
            return d
        if self.kind == 'dice_bce':
            return 0.5 * d + 0.5 * self.bce(logits, masks)
        w_ft = 0.5 if self.stage == 1 else 0.4
        return 0.3 * d + w_ft * self.ft(logits, masks) + 0.2 * self.bce(logits, masks)

    def forward(self, out, aux_out, masks, epoch):
        aux_up = F.interpolate(aux_out, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        loss = self.term(out, masks) + self.w_aux * self.term(aux_up, masks)
        if self.kind == 'dice_bce_ft' and self.stage == 2:
            loss = loss + 0.1 * self.boundary(out, masks)
        return loss


class ScheduledAug:
    """Augmentation that follows an epoch schedule via a shared counter, so persistent
    DataLoader workers pick up schedule changes without being re-spawned each epoch."""

    def __init__(self, schedule):
        self.schedule = schedule
        self.epoch = mp.Value('i', 0)

    def __call__(self, **kw):
        return self.schedule(self.epoch.value)(**kw)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------- data
def make_transforms(prep):
    """prep: None (raw image), 'full' (proposed pipeline) or a PREPROC_VARIANTS key."""
    if not prep:
        return None, None
    tm = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor()])
    if prep is True or prep == 'full':
        return partial(preprocess), tm
    return partial(preprocess_variant, variant=prep), tm


def build_busi(prep):
    ti, tm = make_transforms(prep)
    mk = lambda aug: BUSIDataset(os.path.join(ROOT, 'datasets/BUSI'), transform_image=ti, transform_mask=tm,
                                 augmentation=aug, limit={'benign': BENIGN_CAP})
    train_ds, eval_ds = mk(WEAK), mk(VAL)
    labels = np.array([s[0] for s in eval_ds.samples])
    assert [s[1] for s in train_ds.samples] == [s[1] for s in eval_ds.samples]

    idx = np.arange(len(labels))
    tr, te = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=SPLIT_SEED).split(idx, labels))
    sub_tr, sub_va = next(StratifiedShuffleSplit(1, test_size=0.1, random_state=SPLIT_SEED).split(tr, labels[tr]))
    tr, va = tr[sub_tr], tr[sub_va]
    return train_ds, eval_ds, labels, tr, va, te


def build_breast(prep):
    ti, tm = make_transforms(prep)
    ds = BrEaSTLesionsDataset(os.path.join(ROOT, 'datasets/BrEaST-Lesions_USG-images_and_masks'),
                              transform_image=ti, transform_mask=tm, augmentation=VAL)
    _, te = next(ShuffleSplit(1, test_size=0.2, random_state=SPLIT_SEED).split(np.arange(len(ds))))
    return Subset(ds, te)


def eval_loader(ds, idx):
    # small sets: main process, avoids re-spawning workers at every validation
    return DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)


def train_loader(train_ds, idx, schedule):
    """Fresh dataset copy per stage carrying its own scheduled augmentation."""
    ds = copy.copy(train_ds)
    ds.augmentation = ScheduledAug(schedule)
    dl = DataLoader(Subset(ds, idx), batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                    pin_memory=True, persistent_workers=True)
    return dl, ds.augmentation


# ---------------------------------------------------------------- inference + metrics
@torch.no_grad()
def predict(fn, dl, dev):
    """Returns (probabilities, binary ground truths) as uint8/float32 numpy arrays."""
    probs, gts = [], []
    for x, y in dl:
        p = torch.sigmoid(fn(x.to(dev, non_blocking=True)))
        probs.append(p.squeeze(1).float().cpu().numpy())
        gts.append((y.squeeze(1).numpy() > 0.5).astype(np.uint8))
    return np.concatenate(probs), np.concatenate(gts)


def per_image(prob, gt, thr):
    pred = (prob > thr)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum((1, 2)); fp = np.logical_and(pred, ~gt).sum((1, 2))
    fn = np.logical_and(~pred, gt).sum((1, 2)); tn = np.logical_and(~pred, ~gt).sum((1, 2))
    e = 1e-8
    return dict(dice=2 * tp / (2 * tp + fp + fn + e), iou=tp / (tp + fp + fn + e),
                precision=tp / (tp + fp + e), recall=tp / (tp + fn + e),
                accuracy=(tp + tn) / (tp + tn + fp + fn), pred_area=pred.sum((1, 2)))


def summarise(prob, gt, thr):
    """Metrics on tumour images (non-empty GT); FP rate on empty-GT images."""
    m = per_image(prob, gt, thr)
    tumour = gt.reshape(len(gt), -1).any(1)
    out = {k: float(np.mean(v[tumour])) for k, v in m.items() if k != 'pred_area'}
    out['n_tumour'] = int(tumour.sum())
    if (~tumour).any():
        out['normal_fp_rate'] = float(np.mean(m['pred_area'][~tumour] > 0))
        out['n_normal'] = int((~tumour).sum())
    out['per_image_dice'] = [float(v) for v in m['dice'][tumour]]
    return out


def best_threshold(prob, gt):
    tumour = gt.reshape(len(gt), -1).any(1)
    scores = [per_image(prob[tumour], gt[tumour], t)['dice'].mean() for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def val_dice(prob, gt):
    tumour = gt.reshape(len(gt), -1).any(1)
    return float(per_image(prob[tumour], gt[tumour], 0.5)['dice'].mean())


# ---------------------------------------------------------------- training loop
def train_model(model, params, step_fn, schedule, epochs, train_ds, tr_idx, dl_va, pred_fn, dev, log, tag):
    dl_tr, aug = train_loader(train_ds, tr_idx, schedule)
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best, best_state, best_ep, since = -1.0, None, -1, 0
    for ep in range(epochs):
        aug.epoch.value = ep  # shared with the persistent workers
        model.train(); t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = step_fn(x, y, ep) / ACCUM
            loss.backward()
            if (i + 1) % ACCUM == 0 or (i + 1) == len(dl_tr):
                opt.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * ACCUM; n += 1
        sched.step()
        model.eval()
        vd = val_dice(*predict(pred_fn, dl_va, dev))
        improved = vd > best
        if improved:
            best, best_ep, since = vd, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
        log(f'{tag} ep {ep + 1}/{epochs} loss {tot / n:.4f} val_dice {vd:.4f}{" *" if improved else ""} ({time.time() - t0:.0f}s)')
        if PATIENCE is not None and since >= PATIENCE:
            log(f'{tag} early stop at epoch {ep + 1}; best epoch {best_ep + 1} val_dice {best:.4f}')
            break
    model.load_state_dict(best_state)
    model.eval()
    return dict(best_epoch=best_ep + 1, best_val_dice=best, epochs_run=ep + 1)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(CONFIGS))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results_r3'))
    ap.add_argument('--max-epochs', type=int, default=None, help='smoke test only')
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, 'ckpt'), exist_ok=True)
    name = f'{args.config}_seed{args.seed}'
    logf = open(os.path.join(args.out, name + '.log'), 'a', encoding='utf-8')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True); logf.write(line + '\n'); logf.flush()

    set_seed(args.seed)
    dev = torch.device('cuda')
    cfg = CONFIGS[args.config]
    global BATCH, ACCUM
    BATCH, ACCUM = cfg.get('batch', (BATCH, ACCUM))
    prep = cfg.get('prep')
    in_ch = PREPROC_VARIANTS[prep] if prep else 1
    sched1, sched2 = (stage1_aug, stage2_aug) if cfg.get('aug', True) else (no_aug, no_aug)
    e1 = args.max_epochs or STAGE1_EPOCHS
    e2 = args.max_epochs or STAGE2_EPOCHS

    train_ds, eval_ds, labels, tr, va, te = build_busi(prep)
    log(f'{name}: train {len(tr)} val {len(va)} test {len(te)} | test classes '
        f'{dict(zip(*np.unique(labels[te], return_counts=True)))}')
    dl_va, dl_te = eval_loader(eval_ds, va), eval_loader(eval_ds, te)
    dl_br = DataLoader(build_breast(prep), batch_size=BATCH, num_workers=0)
    result = dict(config=args.config, seed=args.seed, batch=BATCH, accum=ACCUM, n_train=len(tr),
                  n_val=len(va), n_test=len(te))
    t_start = time.time()

    if 'baseline' in cfg:
        model = cfg['baseline'](1, 1).to(dev)
        dice, bce = DiceLoss(), nn.BCEWithLogitsLoss()

        def step(x, y, ep):
            out = model(x)
            return 0.5 * dice(torch.sigmoid(out), y) + 0.5 * bce(out, y)
        fn = lambda x: model(x)
        result['train'] = train_model(model, model.parameters(), step, stage1_aug, e1, train_ds, tr, dl_va,
                                      fn, dev, log, args.config)
        result['params_M'] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        evals = {'full': fn}
        torch.save(model.state_dict(), os.path.join(args.out, 'ckpt', name + '.pth'))
    else:
        mkw = cfg.get('model', {})
        s1 = MasterUNetStage1(in_ch=in_ch, **mkw).to(dev)
        c1 = AblationLoss(cfg['loss'], 1) if 'loss' in cfg else CombinedLossStage1()

        def step1(x, y, ep):
            aux, out = s1(x)
            return c1(out, aux, y, ep)
        fn1 = lambda x: s1(x)[1]
        result['train_stage1'] = train_model(s1, s1.parameters(), step1, sched1, e1, train_ds, tr, dl_va,
                                             fn1, dev, log, 'stage1')
        for p in s1.parameters():
            p.requires_grad = False

        s2 = MasterUNetStage2(in_ch=in_ch, **mkw).to(dev)
        c2 = AblationLoss(cfg['loss'], 2) if 'loss' in cfg else CombinedLossStage2()

        def step2(x, y, ep):
            with torch.no_grad():
                roi = s1(x)[1]
            aux, out = s2(x, roi)
            return c2(out, aux, y, ep)
        fn2 = lambda x: s2(x, s1(x)[1])[1]
        result['train_stage2'] = train_model(s2, s2.parameters(), step2, sched2, e2, train_ds, tr, dl_va,
                                             fn2, dev, log, 'stage2')
        # null ROI: Stage 2 applies sigmoid to its ROI input, so a large negative constant gives an all-zero prior
        null = lambda x: s2(x, torch.full((x.shape[0], 1, *x.shape[-2:]), -20.0, device=x.device))[1]
        evals = {'full': fn2, 'stage1_only': fn1, 'stage2_null_roi': null}
        torch.save({'stage1': s1.state_dict(), 'stage2': s2.state_dict()},
                   os.path.join(args.out, 'ckpt', name + '.pth'))

    result['train_minutes'] = round((time.time() - t_start) / 60, 1)
    result['eval'] = {}
    for key, fn in evals.items():
        thr = best_threshold(*predict(fn, dl_va, dev))
        res = {'threshold': thr, 'busi_test': summarise(*predict(fn, dl_te, dev), thr)}
        if key == 'full':
            res['breast_unseen'] = summarise(*predict(fn, dl_br, dev), thr)
        result['eval'][key] = res
        b = res['busi_test']
        log(f'{key}: thr {thr} | BUSI test Dice {b["dice"]:.4f} IoU {b["iou"]:.4f} Acc {b["accuracy"]:.4f}'
            + (f' | BrEaST Dice {res["breast_unseen"]["dice"]:.4f}' if 'breast_unseen' in res else ''))

    # inference time for the full pipeline (single 256x256 image)
    x = torch.randn(1, in_ch, 256, 256, device=dev)
    with torch.no_grad():
        for _ in range(5):
            evals['full'](x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(50):
            evals['full'](x)
        torch.cuda.synchronize()
    result['inference_ms'] = round((time.time() - t) / 50 * 1000, 1)

    with open(os.path.join(args.out, name + '.json'), 'w') as f:
        json.dump(result, f, indent=1)
    log(f'DONE {name} in {result["train_minutes"]} min; inference {result["inference_ms"]} ms/image')


if __name__ == '__main__':
    main()
