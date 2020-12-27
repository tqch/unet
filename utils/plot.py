from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import math


def plot_masks(preds, masks, path, nrow=8, title=""):
    size = min(nrow, preds.size(0)), math.ceil(preds.size(0) / nrow)
    fig, ax = plt.subplots(2, 1, figsize=(6 * size[0], 12 * size[1]))
    preds_cat = make_grid(preds, nrow=nrow).numpy().transpose(1, 2, 0)
    masks_cat = make_grid(masks.unsqueeze(1), nrow=nrow).numpy().transpose(1, 2, 0)
    ax[0].imshow(preds_cat.astype("float"))
    ax[1].imshow(masks_cat.astype("float"))
    fig.suptitle(title, fontsize=16)
    fig.savefig(path)
