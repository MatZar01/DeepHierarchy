#%%
import os
import numpy as np
import torch
from imutils import paths
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from src import Augmenter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

aug_dict = {
        'CUTMIX': False,
        'CUTMIX_P': 0.5,
        'MEAN': [0.40371441, 0.44959548, 0.47393997],
        'STD': [0.29335567, 0.27962715, 0.28810233],
        'FLIP': 0.5,
        'ROTATION': [-20, 20],
        'GAUSS': [0, 0.08]
    }


def get_class_sorted(path='./Models_CSM/sample/sample_DS'):
    names = [f.path for f in os.scandir(path) if f.is_dir()]
    names = [x.split('/')[-1] for x in names]
    names.sort()
    return names


def get_image_tensor(path, augmenter=None):
    im_0 = cv2.imread(path)
    im_0 = cv2.normalize(im_0, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if augmenter is not None:
        im = augmenter.apply_augs(im_0)
    else:
        im = cv2.resize(im_0, (224, 224))
        im = torch.Tensor(im)
        im = torch.permute(im, [-1, 0, 1])
    return im.unsqueeze(0)


def get_scalped_model(model):
    model.clf = torch.nn.Identity()
    return model


def get_top_bottom(data, top, num):
    paths = np.array(data['paths'])
    sims = data['sims']
    if top:
        selected = np.argpartition(sims, -1*num)[-1*num:]
    else:
        selected = np.argpartition(sims, num)[:num]
    return paths[[selected]], np.array(sims)[[selected]]


def get_sims(model, augmenter, target_name, source_name, sub_source=False, ds_path='./Models_CSM/sample/sample_DS'):
    model_templates = model.clf.weight.detach().cpu().numpy()
    model_scalped = get_scalped_model(model).to('cuda')
    model_scalped.eval()
    class_names = get_class_sorted()

    target_id = class_names.index(target_name)
    source_id = class_names.index(source_name)
    source_paths = list(paths.list_images(f'{ds_path}/{source_name}'))

    target_template = model_templates[target_id, :]
    source_template = model_templates[source_id, :]

    out_dict = {'paths': [], 'sims': []}
    for p in tqdm(source_paths):
        im = get_image_tensor(p, augmenter).to('cuda')
        out = model_scalped(im).squeeze().detach().cpu().numpy()

        sim = cosine_similarity(out.reshape(1, -1), target_template.reshape(1, -1)).item()
        if sub_source:
            sim -= cosine_similarity(out.reshape(1, -1), source_template.reshape(1, -1)).item()
        out_dict['paths'].append(p)
        out_dict['sims'].append(sim)
    print('DONE')
    return out_dict


def get_sample_tgt(id):
    pts = list(paths.list_images(f'/home/mateusz/Desktop/DeepHierarchy/Models_CSM/sample/sample_DS/{id}'))
    pt = np.random.choice(pts, 1)
    im = cv2.imread(pt[0])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (300, 300))
    return im


def show_im_split(best_pt, best_sim, worst_pt, worst_sim, sample_target):
    best_ims = []
    worst_ims = []
    for p in best_pt[0]:
        im = cv2.imread(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (300, 300))
        best_ims.append(im)
    for p in worst_pt[0]:
        im = cv2.imread(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (300, 300))
        worst_ims.append(im)

    captions = ["Main caption", "Caption 2", "Caption 3", "Caption 4", "Caption 5"]

    # Set up figure with GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

    # --- Row 1 (centered image spans 2 columns) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(sample_target)
    ax1.set_title("Target Image", fontsize=14, weight="bold")
    ax1.axis("off")
    ax1.set_xlabel(captions[0], fontsize=10)

    # --- Row 2 (two images) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(best_ims[0])
    ax2.set_title(f'SIM: {best_sim[0, 0]}')
    ax2.axis("off")
    ax2.set_xlabel(captions[1], fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(best_ims[1])
    ax3.set_title(f'SIM: {best_sim[0, 1]}')
    ax3.axis("off")
    ax3.set_xlabel(captions[2], fontsize=9)

    # --- Row 3 (two images) ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.imshow(worst_ims[0])
    ax4.set_title(f'SIM: {worst_sim[0, 0]}')
    ax4.axis("off")
    ax4.set_xlabel(captions[3], fontsize=9)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.imshow(worst_ims[1])
    ax5.set_title(f'SIM: {worst_sim[0, 1]}')
    ax5.axis("off")
    ax5.set_xlabel(captions[4], fontsize=9)

    # First tighten layout to prevent overlaps
    plt.tight_layout()

    # Then increase space between rows (without breaking tightness)
    plt.subplots_adjust(hspace=0.6, top=0.9)

    # --- Auto place row titles ---
    def add_row_title(fig, axes, text, **kwargs):
        """Place a row title centered above a list of axes."""
        boxes = [ax.get_position() for ax in axes]
        xmid = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
        ytop = max(b.y1 for b in boxes)
        fig.text(xmid, ytop + 0.02, text, ha="center", va="bottom", **kwargs)

    add_row_title(fig, [ax2, ax3], "Best SIM matches", fontsize=16, weight="bold")
    add_row_title(fig, [ax4, ax5], "Worst SIM matches", fontsize=16, weight="bold")

    plt.savefig('./Models_CSM/sample/sample_output.png')

model = torch.load('./Models_CSM/sample/sample_MobilenetV2.pth')
model.eval()

augmenter = Augmenter(aug_dict=aug_dict, shape=[224, 224], train=False)

TARGET_ID = input('Set TARGET ID (eg. n01855672): ')
SOURCE_ID = input('Set SOURCE ID (eg. n01532829): ')

# images source for pics ids
#SOURCE_ID = 'n01532829'
# template target to get weights
#TARGET_ID = "n01855672"

try:
    sample_target_im = get_sample_tgt(TARGET_ID)

    out_sims = get_sims(model, augmenter, target_name=TARGET_ID, source_name=SOURCE_ID, sub_source=True)
    bests_pts, best_sims = get_top_bottom(out_sims, True, 2)
    worsts_pts, worsts_sims = get_top_bottom(out_sims, False, 2)

    show_im_split(bests_pts, best_sims, worsts_pts, worsts_sims, sample_target_im)
except:
    print('Check your IDS, might be wrong')
