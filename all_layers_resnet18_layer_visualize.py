import random
import numpy as np
import torch
import os
from torchvision import transforms
import argparse
from torch import nn
from utils import supervisor, tools, default_args
import config
from matplotlib import pyplot as plt
from sklearn import svm
from umap import UMAP
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from tqdm import tqdm

# Function to combine multiple colormaps
def get_combined_colormap(num_classes, colormaps):
    colors = []
    for cmap_name in colormaps:
        cmap = cm.get_cmap(cmap_name)
        num_colors = cmap.N
        for i in range(num_colors):
            colors.append(cmap(i / num_colors))
            if len(colors) >= num_classes:
                return colors
    # If not enough colors, repeat
    while len(colors) < num_classes:
        colors.extend(colors[:num_classes - len(colors)])
    return colors

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-method', type=str, required=False, default='umap',
                    choices=['umap'])  # Only UMAP is supported
parser.add_argument('-n_neighbors', type=int, default=10, help='Number of neighbors for UMAP')
parser.add_argument('-min_dist', type=float, default=0.1, help='Minimum distance for UMAP')
parser.add_argument('-n_components', type=int, default=2, help='Number of components for UMAP')
parser.add_argument('-metric', type=str, default='euclidean', help='Metric for UMAP')
parser.add_argument('-dataset', type=str, required=False, default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=True,
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float, required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float, required=False, default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False, default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-model', type=str, required=False, default=None)
parser.add_argument('-model_path', required=False, default=None)
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-target_class', type=int, default=-1)
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
# New argument for controlling the fraction of data to visualize
parser.add_argument('-data_ratio', type=float, default=1.0,
                    help='Ratio of the dataset to use for visualization (0 < data_ratio <= 1.0)')

args = parser.parse_args()

# Set CUDA devices and seed
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
tools.setup_seed(args.seed)

# Determine target class
if args.target_class == -1:
    target_class = config.target_class[args.dataset]
else:
    target_class = args.target_class

# Set trigger if not provided
if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]

# Batch size and DataLoader settings
batch_size = 64
kwargs = {'num_workers': 4, 'pin_memory': True}

# Determine number of classes based on dataset
if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'gtsrb':
    num_classes = 43
elif args.dataset == 'imagenette':
    num_classes = 10
else:
    raise NotImplementedError('<Unimplemented Dataset> %s' % args.dataset)

# Get data transforms
data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

# Get model architecture
arch = supervisor.get_arch(args)

# Set up poisoned dataset
poison_set_dir = supervisor.get_poison_set_dir(args)
if os.path.exists(os.path.join(poison_set_dir, 'data')):
    # old version directory
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
elif os.path.exists(os.path.join(poison_set_dir, 'imgs')):
    # new version directory
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'imgs')
else:
    raise FileNotFoundError("Poisoned data directory not found.")

poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                 label_path=poisoned_set_label_path, transforms=data_transform)

# Use a fraction of the dataset based on data_ratio
if 0 < args.data_ratio < 1.0:
    total_samples = len(poisoned_set)
    selected_count = int(total_samples * args.data_ratio)
    subset_indices = torch.arange(0, selected_count)
    poisoned_set = torch.utils.data.Subset(poisoned_set, subset_indices)

poisoned_set_loader = torch.utils.data.DataLoader(
    poisoned_set,
    batch_size=batch_size, shuffle=False, **kwargs)

# Load poison indices and adjust if subset is used
if os.path.exists(poison_indices_path):
    poison_indices_full = torch.tensor(torch.load(poison_indices_path))
    # If we took a subset, we only keep indices that are within the subset range
    if 0 < args.data_ratio < 1.0:
        selected_count = len(poisoned_set)  # after subset
        poison_indices = poison_indices_full[poison_indices_full < selected_count]
    else:
        poison_indices = poison_indices_full
else:
    poison_indices = torch.tensor([])  # No poisoned samples

# Set up test dataset
test_set_dir = f'clean_set/{args.dataset}/test_split/'
test_set_img_dir = os.path.join(test_set_dir, 'data')
test_set_label_path = os.path.join(test_set_dir, 'labels')
test_set = tools.IMG_Dataset(data_dir=test_set_img_dir, label_path=test_set_label_path,
                             transforms=data_transform)
test_set_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size, shuffle=False, **kwargs
)

# Prepare model list
model_list = []
alias_list = []

if (hasattr(args, 'model_path') and args.model_path is not None) or (hasattr(args, 'model') and args.model is not None):
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append('assigned')
else:
    args.no_aug = False
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append(supervisor.get_model_name(args))

# Define poison transform
poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                   target_class=target_class,
                                                   trigger_transform=data_transform,
                                                   is_normalized_input=True,
                                                   alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                   trigger_name=args.trigger, args=args)

# Define source classes if needed
if args.poison_type == 'TaCT':
    source_classes = [config.source_class]
else:
    source_classes = None

for vid, path in enumerate(model_list):
    # Load the model checkpoint. If using PyTorch < 2.0, remove 'weights_only' argument.
    ckpt = torch.load(path, weights_only=True)

    model = arch(num_classes=num_classes)
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    # Begin visualization
    print(f"Visualizing model '{path}' on {args.dataset}...")

    print('[test]')
    tools.test(model, test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=num_classes,
               source_classes=source_classes)

    targets = []
    ids = torch.arange(len(poisoned_set))

    # Dictionary to store outputs for each layer
    layer_outputs = {}

    # Hook function to capture layer outputs
    def get_activation(name):
        def hook(model, input, output):
            if name not in layer_outputs:
                layer_outputs[name] = []
            layer_outputs[name].append(output.detach().cpu())
        return hook

    # Register hooks recursively for all child layers
    def register_hooks(module, prefix=""):
        for name, layer in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            # Register hook for feature layers
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.ReLU, nn.AdaptiveAvgPool2d)):
                layer.register_forward_hook(get_activation(layer_name))
            # Continue recursively
            register_hooks(layer, prefix=layer_name)

    try:
        register_hooks(model.module)
    except AttributeError as e:
        raise AttributeError(f"Error registering hooks: {e}. Ensure the model architecture matches the expected layers.")

    # Processing poisoned data to capture features
    for batch_idx, (data, target) in enumerate(tqdm(poisoned_set_loader, desc="Processing Poisoned Data")):
        data, target = data.cuda(), target.cuda()
        targets.append(target.cpu())
        with torch.no_grad():
            _ = model(data)

    targets = torch.cat(targets, dim=0)
    ids = torch.arange(len(poisoned_set))

    # Create directory to save figures
    save_dir = os.path.join('assets', 'resnet18_layer_visualization', args.dataset, args.poison_type)
    os.makedirs(save_dir, exist_ok=True)

    for layer_name in tqdm(layer_outputs.keys(), desc="Layers"):
        print(f"Processing layer {layer_name}...")

        layer_features = torch.cat(layer_outputs[layer_name], dim=0)

        # Flatten features if needed
        if len(layer_features.size()) > 2:
            layer_features = layer_features.view(layer_features.size(0), -1)

        # Only UMAP is used here
        visualizer = UMAP(
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed
        )

        # Prepare labels
        all_labels = targets.clone()
        all_features = layer_features
        ids = torch.arange(len(all_features))

        if len(poison_indices) == 0:
            # No poisoned samples
            print("No poisoned samples found.")
            labels = all_labels.numpy()
            feats_np = all_features.numpy()
        else:
            # Mark poisoned samples with a new class index
            poisoned_label = num_classes
            all_labels[poison_indices] = poisoned_label
            labels = all_labels.numpy()
            feats_np = all_features.numpy()

        # Prepare colormap
        num_total_classes = num_classes + (1 if len(poison_indices) > 0 else 0)
        colormap_list = ['Set2', 'Set3', 'Accent', 'tab20b']
        colors = get_combined_colormap(num_total_classes, colormap_list)

        # Update color for the target class (black)
        target_color = (0.0, 0.0, 0.0, 1.0)
        colors[target_class] = target_color

        # Add red for poisoned samples if any
        if len(poison_indices) > 0:
            poisoned_color = (1.0, 0.0, 0.0, 1.0)
            colors[num_classes] = poisoned_color

        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=num_total_classes - 1)

        # Apply UMAP for dimensionality reduction
        reduced_features = visualizer.fit_transform(feats_np)

        # Assign markers and colors per class
        class_markers = {}
        class_colors = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == target_class:
                class_markers[label] = '*'
                class_colors[label] = target_color
            elif len(poison_indices) > 0 and label == poisoned_label:
                # Poisoned samples
                class_markers[label] = 'X'
                class_colors[label] = poisoned_color
            else:
                # Other classes
                class_markers[label] = 'o'
                class_colors[label] = colors[int(label)]

        plt.figure(figsize=(10, 8))
        for label in unique_labels:
            idx = labels == label
            plt.scatter(
                reduced_features[idx, 0],
                reduced_features[idx, 1],
                c=[class_colors[label]],
                marker=class_markers[label],
                s=10,
                alpha=0.4,
                label=f'Class {int(label)}' if label not in [target_class, poisoned_label] else ''
            )

        # Create custom legend handles
        handles = [
            Line2D([], [], marker='*', color=target_color, linestyle='None', markersize=6, label='Target Class'),
            Line2D([], [], marker='X', color=(1.0, 0.0, 0.0, 1.0), linestyle='None', markersize=6, label='Poisoned Samples'),
            Line2D([], [], marker='o', color='grey', linestyle='None', markersize=6, label='Other Classes')
        ]

        # plt.legend(handles=handles, title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'UMAP Visualization for Layer {layer_name}')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.tight_layout()

        # Construct the filename
        core_dir = supervisor.get_dir_core(args, include_poison_seed=True)
        alias = alias_list[vid]
        safe_layer_name = layer_name.replace('.', '_')
        filename = f"{safe_layer_name}_{args.method}_{core_dir}_{alias}_class={target_class}.png"
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi=300)
        print(f"Saved figure at {save_path}")
        plt.clf()

print("Visualization completed.")

