import os
import pdb
import torch
import config
import torchvision
from sklearn import metrics
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn as nn
import numpy as np

from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms
from utils import supervisor, tools

class IBD_PSC(BackdoorDefense):
    """Identify and filter malicious testing samples (IBD-PSC)."""

    name: str = 'IBD_PSC'

    def __init__(self, args, n=5, xi=0.6, T=0.9, scale=1.5):
        super().__init__(args)
        self.model.eval()
        self.args = args
        self.n = n
        self.xi = xi
        self.T = T
        self.scale = scale

        self.test_loader = generate_dataloader(
            dataset=self.dataset,
            dataset_path=config.data_dir,
            batch_size=64,  # smaller batch to mitigate OOM
            split='test',
            data_transform=self.data_transform,
            shuffle=False,
            drop_last=False,
            noisy_test=False
        )

        self.val_loader = generate_dataloader(
            dataset=self.dataset,
            dataset_path=config.data_dir,
            batch_size=64,  # smaller batch to mitigate OOM
            split='full_test',
            data_transform=self.data_transform,
            shuffle=True,
            drop_last=False,
            noisy_test=False
        )

        # Count BN layers
        layer_num = self.count_BN_layers()
        sorted_indices = list(range(layer_num))
        sorted_indices = list(reversed(sorted_indices))
        self.sorted_indices = sorted_indices

        # Find earliest index that gives > xi error on clean
        self.start_index = self.prob_start(self.scale, self.sorted_indices)

        # Quick check of clean accuracy and ASR
        self.print_metrics()

    def count_BN_layers(self):
        layer_num = 0
        for _, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                layer_num += 1
        return layer_num

    def get_BN_params(self):
        """
        Collect references to all BN weight/bias.
        """
        bn_params = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_params.append((module.weight, module.bias))
        return bn_params

    def scale_BN_inplace(self, index_bn_list, scale_factor):
        bn_params = self.get_BN_params()
        old_params = {}
        for idx in index_bn_list:
            w, b = bn_params[idx]
            old_params[idx] = (w.clone(), b.clone())
            w.data *= scale_factor
            b.data *= scale_factor
        return old_params

    def revert_BN_inplace(self, old_params):
        bn_params = self.get_BN_params()
        for idx, (old_w, old_b) in old_params.items():
            w, b = bn_params[idx]
            w.data = old_w
            b.data = old_b

    def prob_start(self, scale, sorted_indices):
        layer_num = len(sorted_indices)
        for layer_index in range(1, layer_num):
            layers_to_scale = sorted_indices[:layer_index]

            with torch.no_grad():
                old_params = self.scale_BN_inplace(layers_to_scale, scale)

                total_num = 0
                clean_wrong = 0
                for batch_idx, batch in enumerate(self.val_loader):
                    clean_img, labels = batch
                    clean_img = clean_img.cuda()

                    # forward
                    logits = self.model(clean_img).cpu()
                    clean_pred = torch.argmax(logits, dim=1)  # CPU
                    clean_wrong += torch.sum(labels != clean_pred).item()
                    total_num += labels.shape[0]

                # revert
                self.revert_BN_inplace(old_params)

            wrong_acc = clean_wrong / total_num
            if wrong_acc > self.xi:
                return layer_index
        return layer_num

    def print_metrics(self):
        """
        Print clean accuracy and ASR on entire test set.
        Fix: match devices for poison_labels and poison_pred.
        """
        print('Checking clean accuracy and ASR ...')
        total_num = 0
        clean_correct = 0
        bd_correct = 0
        bd_all = 0

        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                clean_img, labels = batch
                batch_size = labels.shape[0]
                total_num += batch_size

                # move to GPU
                clean_img = clean_img.cuda()
                labels = labels.cuda()

                # create poison
                target_flag = labels != 0  # (just as your example logic)
                poison_imgs, poison_labels = self.poison_transform.transform(
                    clean_img[target_flag],
                    labels[target_flag]
                )
                # also on GPU
                poison_imgs = poison_imgs.cuda()
                poison_labels = poison_labels.cuda()

                # forward
                clean_logits = self.model(clean_img)  # still on GPU
                bd_logits = self.model(poison_imgs)   # GPU

                # move predictions to CPU for comparison
                clean_pred = torch.argmax(clean_logits, dim=1).cpu()
                poison_pred = torch.argmax(bd_logits, dim=1).cpu()

                # also move labels to CPU
                labels_cpu = labels.cpu()
                poison_labels_cpu = poison_labels.cpu()

                # Count correct
                clean_correct += torch.sum(labels_cpu == clean_pred).item()

                # Count backdoor success
                if self.args.poison_type == 'TaCT':
                    mask = (labels_cpu == config.source_class)
                    plabels = poison_labels_cpu[mask]
                    ppred = poison_pred[mask]
                    bd_correct += torch.sum(plabels == ppred).item()
                    bd_all += plabels.size(0)
                else:
                    bd_correct += torch.sum(poison_labels_cpu == poison_pred).item()
                    bd_all += poison_labels_cpu.shape[0]

        acc = clean_correct * 100.0 / total_num
        asr = bd_correct * 100.0 / bd_all if bd_all > 0 else 0
        print(f'Clean Accuracy: {acc:.2f}%')
        print(f'ASR: {asr:.2f}%')
        print(f'Start index is {self.start_index}')

    ##########################
    #   The rest is the PSC  #
    ##########################

    def test(self, inspect_correct_predition_only=False):
        """
        Evaluate detection performance on test set.
        """
        print(f"start_index: {self.start_index}")
        y_score_clean = []
        y_score_poison = []
        total_num = 0

        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                clean_img, labels = batch
                batch_size = labels.size(0)
                total_num += batch_size

                clean_img = clean_img.cuda()
                labels = labels.cuda()

                poison_imgs, _ = self.poison_transform.transform(clean_img, labels)
                poison_imgs = poison_imgs.cuda()

                # original predictions
                poison_logits = self.model(poison_imgs).cpu()
                clean_logits = self.model(clean_img).cpu()
                poison_pred = torch.argmax(poison_logits, dim=1)
                clean_pred = torch.argmax(clean_logits, dim=1)

                # PSC accumulators
                spc_poison = torch.zeros(batch_size)
                spc_clean = torch.zeros(batch_size)

                # repeated scaling
                for layer_index in range(self.start_index, self.start_index + self.n):
                    layers_to_scale = self.sorted_indices[:layer_index + 1]
                    old_params = self.scale_BN_inplace(layers_to_scale, self.scale)

                    c_logits = self.model(clean_img).cpu()
                    p_logits = self.model(poison_imgs).cpu()

                    c_softmax = F.softmax(c_logits, dim=1)
                    p_softmax = F.softmax(p_logits, dim=1)

                    spc_clean += c_softmax[torch.arange(batch_size), clean_pred]
                    spc_poison += p_softmax[torch.arange(batch_size), poison_pred]

                    self.revert_BN_inplace(old_params)

                spc_poison /= float(self.n)
                spc_clean /= float(self.n)

                y_score_clean.append(spc_clean)
                y_score_poison.append(spc_poison)

        y_score_clean = torch.cat(y_score_clean, dim=0)
        y_score_poison = torch.cat(y_score_poison, dim=0)

        # Build labels for detection: 0 = clean, 1 = poison
        y_true = torch.cat((torch.zeros_like(y_score_clean),
                            torch.ones_like(y_score_poison)), dim=0)
        y_score = torch.cat((y_score_clean, y_score_poison), dim=0)
        y_pred = (y_score >= self.T).long()

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        myf1 = metrics.f1_score(y_true, y_pred)

        print(f"TPR: {tp / (tp + fn) * 100:.2f}%")
        print(f"FPR: {fp / (tn + fp) * 100:.2f}%")
        print(f"AUC: {auc:.4f}")
        print(f"F1-score: {myf1:.4f}")

        if inspect_correct_predition_only:
            self.inspect_partial_testset(y_true, y_pred, y_score)

    def inspect_partial_testset(self, y_true, y_pred, y_score):
        """
        Only evaluate detection for:
          - clean samples that are predicted correctly,
          - poison samples that actually trigger the backdoor.
        """
        args = self.args
        clean_pred_correct_mask = []
        poison_source_mask = []
        poison_attack_success_mask = []

        # mask building
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                data, label = batch
                data, label = data.cuda(), label.cuda()

                # clean
                clean_output = self.model(data)
                clean_pred = clean_output.argmax(dim=1)
                clean_mask = (clean_pred == label)
                clean_pred_correct_mask.append(clean_mask)

                # poison
                poison_data, poison_target = self.poison_transform.transform(data, label)
                if args.poison_type == 'TaCT':
                    mask1 = (label == config.source_class)
                else:
                    # remove backdoor data whose original class == target class
                    mask1 = (label != poison_target)

                poison_source_mask.append(mask1.clone())

                poison_output = self.model(poison_data)
                poison_pred = poison_output.argmax(dim=1)

                # success if poison_pred == poison_target (and from the source class)
                mask2 = (poison_pred == poison_target) & mask1
                poison_attack_success_mask.append(mask2)

        clean_pred_correct_mask = torch.cat(clean_pred_correct_mask, dim=0)
        poison_source_mask = torch.cat(poison_source_mask, dim=0)
        poison_attack_success_mask = torch.cat(poison_attack_success_mask, dim=0)

        # If user wants partial test
        # y_true = [clean(0) + poison(1)] in that order
        # So we need to stack the same ordering as in test() method above
        # We won't re-run forward; we'll simply filter existing y_true/y_pred/y_score.

        # For TaCT special case, we ensure that the # of backdoor samples is from source_class only.
        if args.poison_type == 'TaCT':
            # In some code, they do a slicing to match
            # but here let's just do a simpler approach:
            pass

        # Combine clean + poison mask
        partial_mask = torch.cat((clean_pred_correct_mask, poison_attack_success_mask), dim=0)
        # Ensure the shape matches y_true in test(). They should be the same length if test_loader
        # is used fully once for clean and once for poison.
        # In practice, you might need extra index bookkeeping. This is just a demo.

        # Filter
        y_true_partial = y_true[partial_mask]
        y_pred_partial = y_pred[partial_mask]
        y_score_partial = y_score[partial_mask]

        # Recompute detection metrics
        fpr, tpr, thresholds = metrics.roc_curve(y_true_partial, y_score_partial)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true_partial, y_pred_partial).ravel()
        myf1 = metrics.f1_score(y_true_partial, y_pred_partial)

        print("\n[IBD_PSC: Partial Test Set]")
        print(f"  TPR: {tp / (tp + fn) * 100:.2f}%")
        print(f"  FPR: {fp / (tn + fp) * 100:.2f}%")
        print(f"  AUC: {auc:.4f}")
        print(f"  F1: {myf1:.4f}\n")

    ########################
    # 5) ONLINE DETECTION  #
    ########################

    def _detect(self, inputs: torch.Tensor):
        """
        Online detection for a batch of inputs.
        """
        inputs = inputs.cuda()
        self.model.eval()

        with torch.no_grad():
            original_logits = self.model(inputs).cpu()
            original_pred = torch.argmax(original_logits, dim=1)

            psc_score = torch.zeros(inputs.size(0))

            for layer_index in range(self.start_index, self.start_index + self.n):
                layers_to_scale = self.sorted_indices[:layer_index + 1]
                old_params = self.scale_BN_inplace(layers_to_scale, self.scale)

                logits = self.model(inputs).cpu()
                softmax_logits = F.softmax(logits, dim=1)
                psc_score += softmax_logits[torch.arange(inputs.size(0)), original_pred]

                self.revert_BN_inplace(old_params)

            psc_score /= float(self.n)
            return (psc_score >= self.T)  # True if backdoor

    def detect(self):
        """
        Demo usage of _detect() on first batch only.
        """
        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                imgs, _ = batch
                detection = self._detect(imgs)
                print(f"Single-batch detection: {detection}")
                break