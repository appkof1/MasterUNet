"""Boundary metrics (Table 9) and noise robustness (Table 11) from the results_r3 checkpoints.

For every finished run of the listed configs:
  * clean BUSI test set: Dice, IoU, HD95 and ASSD (pixels on the 256x256 grid; BUSI provides no pixel
    spacing, so millimetres cannot be computed). For a tumour image with an empty prediction the surface
    distances are undefined; such images are assigned the image diagonal (362 px) as a penalty and counted.
  * noisy BUSI test sets: additive Gaussian noise with sigma = 5%, 10% and 20% of the 8-bit range, added to
    the resized raw image before any preprocessing, clipped to [0, 255]. The noise for each image is seeded
    by the image content, so every model sees identical noisy images.
  * HD95/ASSD are also reported over the images with a non-empty prediction only, with the count of empty ones.
The decision threshold is the one chosen on the validation set during training (stored in the run JSON).


Usage: python eval_extra.py [--configs unet masterunet ...] [--aggregate]
"""
import argparse, glob, json, os, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scipy import ndimage
import run_core as rc
from models.two_stage_unet_aux import MasterUNetStage1, MasterUNetStage2

SIGMAS = [0.05, 0.10, 0.20]  # fractions of the 8-bit range; 0.20 is the level used in the manuscript
DIAG = float(np.hypot(256, 256))
OUT = os.path.join(rc.ROOT, 'results_r3', 'extra')


class NoisyVal:
    """VAL resize followed by seeded additive Gaussian noise (picklable, deterministic per image)."""

    def __init__(self, sigma=0.20):
        self.sigma = sigma * 255

    def __call__(self, image, mask):
        out = rc.VAL(image=image, mask=mask)
        img = out['image']
        rng = np.random.default_rng(zlib.crc32(img.tobytes()))
        out['image'] = np.clip(img.astype(np.float32) + rng.normal(0, self.sigma, img.shape), 0, 255).astype(np.uint8)
        return out


def surface(m):
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)


def surface_distances(pred, gt):
    if not pred.any():
        return None
    sp, sg = surface(pred), surface(gt)
    d_to_g = ndimage.distance_transform_edt(~sg)[sp]
    d_to_p = ndimage.distance_transform_edt(~sp)[sg]
    return np.concatenate([d_to_g, d_to_p])


def boundary_metrics(prob, gt, thr):
    hd, asd, empty, hd_ne, asd_ne = [], [], 0, [], []
    for p, g in zip(prob, gt):
        g = g.astype(bool)
        if not g.any():
            continue
        d = surface_distances(p > thr, g)
        if d is None:
            empty += 1; hd.append(DIAG); asd.append(DIAG)
        else:
            hd.append(np.percentile(d, 95)); asd.append(d.mean()); hd_ne.append(hd[-1]); asd_ne.append(asd[-1])
    return dict(hd95=float(np.mean(hd)), assd=float(np.mean(asd)), n_empty_pred=empty,
                hd95_nonempty=float(np.mean(hd_ne)) if hd_ne else float('nan'),
                assd_nonempty=float(np.mean(asd_ne)) if asd_ne else float('nan'))


def load_fn(cfg_name, run, dev):
    cfg = rc.CONFIGS[cfg_name]
    ck = torch.load(os.path.join(rc.ROOT, 'results_r3', 'ckpt', run + '.pth'), map_location=dev, weights_only=True)
    if 'baseline' in cfg:
        m = cfg['baseline'](1, 1).to(dev); m.load_state_dict(ck); m.eval()
        return lambda x: m(x)
    in_ch = rc.PREPROC_VARIANTS[cfg['prep']] if cfg.get('prep') else 1
    s1 = MasterUNetStage1(in_ch=in_ch, **cfg.get('model', {})).to(dev)
    s2 = MasterUNetStage2(in_ch=in_ch, **cfg.get('model', {})).to(dev)
    s1.load_state_dict(ck['stage1']); s2.load_state_dict(ck['stage2']); s1.eval(); s2.eval()
    return lambda x: s2(x, s1(x)[1])[1]


def evaluate(cfg_name, dev):
    os.makedirs(OUT, exist_ok=True)
    prep = rc.CONFIGS[cfg_name].get('prep')
    _, ev, labels, tr, va, te = rc.build_busi(prep)
    noisy = {}
    for sg in SIGMAS:
        noisy[sg] = rc.copy.copy(ev); noisy[sg].augmentation = NoisyVal(sg)
    for jf in sorted(glob.glob(os.path.join(rc.ROOT, 'results_r3', f'{cfg_name}_seed*.json'))):
        run = os.path.basename(jf)[:-5]
        r = json.load(open(jf)); thr = r['eval']['full']['threshold']
        fn = load_fn(cfg_name, run, dev)
        res = {'config': cfg_name, 'seed': r['seed'], 'threshold': thr}
        for split, ds in [('clean', ev)] + [(f'noisy{int(round(sg * 100))}', noisy[sg]) for sg in SIGMAS]:
            prob, gt = rc.predict(fn, rc.eval_loader(ds, te), dev)
            s = rc.summarise(prob, gt, thr); s.pop('per_image_dice')
            s.update(boundary_metrics(prob, gt, thr))
            res[split] = s
        json.dump(res, open(os.path.join(OUT, run + '.json'), 'w'), indent=1)
        print(f"{run}: clean Dice {res['clean']['dice']:.3f} HD95 {res['clean']['hd95']:.1f} "
              f"ASSD {res['clean']['assd']:.1f} | noisy Dice " + ' '.join(f"{k} {v['dice']:.3f}" for k, v in res.items() if k.startswith('noisy')), flush=True)


def aggregate():
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, '*.json'))):
        r = json.load(open(f)); rows.setdefault(r['config'], []).append(r)
    ms = lambda v: f'{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}'
    ms1 = lambda v: f'{np.mean(v):.1f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.1f}'
    for c, rs in rows.items():
        g = lambda s, m: [r[s][m] for r in rs]
        print(f"{c:18s} n={len(rs)} clean Dice {ms(g('clean', 'dice'))} IoU {ms(g('clean', 'iou'))} | "
              f"HD95 {ms1(g('clean', 'hd95'))} ASSD {ms1(g('clean', 'assd'))} (penalised) | "
              f"non-empty HD95 {ms1(g('clean', 'hd95_nonempty'))} ASSD {ms1(g('clean', 'assd_nonempty'))} "
              f"empty/run {np.mean(g('clean', 'n_empty_pred')):.1f}")
        for k in [k for k in rs[0] if k.startswith('noisy')]:
            print(f"{'':18s} {k}: Dice {ms(g(k, 'dice'))} IoU {ms(g(k, 'iou'))} "
                  f"drop Dice {np.mean(g('clean', 'dice')) - np.mean(g(k, 'dice')):.3f} "
                  f"IoU {np.mean(g('clean', 'iou')) - np.mean(g(k, 'iou')):.3f} empty/run {np.mean(g(k, 'n_empty_pred')):.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='*', default=['unet', 'masterunet', 'masterunet_noprep', 'unetpp', 'aaunet', 'mffaunet'])
    ap.add_argument('--aggregate', action='store_true')
    a = ap.parse_args()
    if not a.aggregate:
        dev = torch.device('cuda')
        for c in a.configs:
            evaluate(c, dev)
    aggregate()
