"""
Modified cgr.py with diagnostic logging for the rebuttal experiments.

When --cgr_diag_log is set, this version:
  * During task 1 ONLY, runs CGR's existing eval-mode forward pass on
    not_aug_inputs for ALL epochs of the task (instead of just the first E).
  * Records, from that same eval-mode pass: per-sample target confidence,
    margin (target prob - max other prob), correctness (argmax == label),
    and per-sample cross-entropy loss.
  * After the last epoch of task 1, saves all of the above plus
    CGR's confidence trajectory to disk as cgr_diag_logs/cgr_diag_seed<S>.pt
    (one file per run).

Run separately for each seed (--seed 0, --seed 1, ...). Each run produces
one .pt file. Combine across seeds by hand.

When --cgr_diag_log is NOT set, behaviour is identical to your original cgr.py.

All additions are marked with `# === DIAG: ... === / === END DIAG ===` blocks.
"""

import os
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel

import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='CGR: Confidence-Guided Reply for Buffer-Based Continual Learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--E', type=int, default=4,
                        help='Epoch for selecting samples')
    # === DIAG: new CLI args ===
    parser.add_argument('--cgr_diag_log', action='store_true',
                        help='If set, record per-sample diagnostics during task 1 (from CGR\'s eval forward pass) and save to disk.')
    parser.add_argument('--cgr_diag_dir', type=str, default='cgr_diag_logs',
                        help='Directory to save per-seed diagnostic logs.')
    # === END DIAG ===
    return parser


def distribute_samples(probabilities, M):
    total_probability = sum(probabilities.values())
    normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}
    samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
    discrepancy = M - sum(samples.values())
    for key in samples:
        if discrepancy == 0:
            break
        if discrepancy > 0:
            samples[key] += 1
            discrepancy -= 1
        elif discrepancy < 0 and samples[key] > 0:
            samples[key] -= 1
            discrepancy += 1
    return samples


def distribute_excess(lst, check_bound):
    total_excess = sum(val - check_bound for val in lst if val > check_bound)
    recipients = [i for i, val in enumerate(lst) if val < check_bound]
    num_recipients = len(recipients)
    avg_share, remainder = divmod(total_excess, num_recipients)
    lst = [val if val <= check_bound else check_bound for val in lst]
    for idx in recipients:
        lst[idx] += avg_share
    for idx in recipients[:remainder]:
        lst[idx] += 1
    for i, val in enumerate(lst):
        if val > check_bound:
            return distribute_excess(lst, check_bound)
    return lst


def adjust_values_integer_include_all(a, b):
    excess = {}
    shortage = {}
    total_excess = 0
    for k in a:
        if k in b:
            if a[k] > b[k]:
                excess[k] = a[k] - b[k]
                total_excess += a[k] - b[k]
                a[k] = b[k]
            elif a[k] < b[k]:
                shortage[k] = b[k] - a[k]
        else:
            shortage[k] = float('inf')
    while total_excess > 0 and shortage:
        per_key_excess = max(total_excess // len(shortage), 1)
        for k in list(shortage):
            if total_excess == 0:
                break
            if shortage[k] == float('inf'):
                increment = per_key_excess
            else:
                increment = min(shortage[k], per_key_excess)
            a[k] += increment
            total_excess -= increment
            if shortage[k] != float('inf'):
                shortage[k] -= increment
                if shortage[k] == 0:
                    del shortage[k]
    for key in a:
        a[key] = int(a[key])
    return a


class Cgr(ContinualModel):
    NAME = 'cgr'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Cgr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.epoch = 0
        self.unique_classes = set()
        self.mapping = {}
        self.reverse_mapping = {}
        self.confidence_by_sample = None
        self.n_sample_per_task = None
        self.class_portion = []
        self.dist_task_prev = None
        self.dist_class_prev = None
        # === DIAG: per-sample diagnostic tensors (allocated in begin_task for task 1 only) ===
        self.diag_margin = None       # (n_epochs, n_sample_per_task)
        self.diag_correct = None      # (n_epochs, n_sample_per_task) bool
        self.diag_loss = None         # (n_epochs, n_sample_per_task) per-sample CE
        self.diag_labels = None       # (n_sample_per_task,) global class id
        # === END DIAG ===

    def _diag_active(self):
        """True iff diagnostic logging is enabled AND we're in task 1."""
        return getattr(self.args, 'cgr_diag_log', False) and self.task == 1

    def begin_train(self, dataset):
        self.n_sample_per_task = dataset.get_examples_number() // dataset.N_TASKS

    def begin_task(self, dataset, train_loader):
        self.epoch = 0
        self.task += 1
        self.unique_classes = set()
        for _, labels, _, _ in train_loader:
            self.unique_classes.update(labels.numpy())
            if len(self.unique_classes) == dataset.N_CLASSES_PER_TASK:
                break
        self.mapping = {value: index for index, value in enumerate(self.unique_classes)}
        self.reverse_mapping = {index: value for value, index in self.mapping.items()}
        self.confidence_by_sample = torch.zeros((self.args.n_epochs, self.n_sample_per_task))

        # === DIAG: allocate task-1 diagnostic tensors ===
        if self._diag_active():
            n_e = self.args.n_epochs
            n_s = self.n_sample_per_task
            self.diag_margin = torch.zeros((n_e, n_s))
            self.diag_correct = torch.zeros((n_e, n_s), dtype=torch.bool)
            self.diag_loss = torch.zeros((n_e, n_s))
            self.diag_labels = torch.full((n_s,), -1, dtype=torch.long)
        # === END DIAG ===

    def _save_diag(self):
        """Dump task-1 diagnostics to disk at the end of task 1's last epoch."""
        if not self._diag_active():
            return
        save_dir = getattr(self.args, 'cgr_diag_dir', 'cgr_diag_logs')
        os.makedirs(save_dir, exist_ok=True)
        seed = getattr(self.args, 'seed', 'unknown')
        save_path = os.path.join(save_dir, f'cgr_diag_seed{seed}.pt')
        torch.save({
            'seed': seed,
            'E': self.args.E,
            'n_epochs': self.args.n_epochs,
            'n_sample_per_task': self.n_sample_per_task,
            'buffer_size': self.args.buffer_size,
            # CGR's eval-mode target confidence. With --cgr_diag_log this is
            # filled for ALL epochs of task 1; use [:E] for CGR's variance.
            'cgr_confidence_by_sample': self.confidence_by_sample.clone(),
            # Diagnostics computed from the same eval-mode forward pass on not_aug_inputs
            'diag_margin': self.diag_margin.clone(),
            'diag_correct': self.diag_correct.clone(),
            'diag_loss': self.diag_loss.clone(),
            'diag_labels': self.diag_labels.clone(),
            'class_mapping': dict(self.mapping),
        }, save_path)
        print(f"[CGR-Diag] Saved task-1 diagnostics to {save_path}")

    def end_epoch(self, dataset, train_loader):

        self.epoch += 1

        if self.epoch == self.args.n_epochs:
            # === DIAG: dump task-1 diagnostics before the buffer-update logic ===
            self._save_diag()
            # === END DIAG ===

            # ... rest of the function unchanged from original ...
            std_of_means_by_class = {class_id: 1 for class_id, __ in enumerate(self.unique_classes)}
            std_of_means_by_task = {task_id: 1 for task_id in range(self.task)}

            Confidence_mean = self.confidence_by_sample[:self.args.E].mean(dim=0)
            Variability = self.confidence_by_sample[:self.args.E].var(dim=0)

            sorted_indices_2 = np.argsort(Variability.numpy())
            top_indices_sorted = sorted_indices_2[::-1].copy()

            all_inputs, all_labels, all_not_aug_inputs, all_indices = [], [], [], []
            for data_1 in train_loader:
                inputs_1, labels_1, not_aug_inputs_1, indices_1 = data_1
                all_inputs.append(inputs_1)
                all_labels.append(labels_1)
                all_not_aug_inputs.append(not_aug_inputs_1)
                all_indices.append(indices_1)

            all_inputs = torch.cat(all_inputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_not_aug_inputs = torch.cat(all_not_aug_inputs, dim=0)
            all_indices = torch.cat(all_indices, dim=0)

            top_indices_sorted = torch.tensor(top_indices_sorted, dtype=torch.long)
            positions = torch.hstack([torch.where(all_indices == index)[0] for index in top_indices_sorted])

            all_images = all_not_aug_inputs[positions]
            all_labels = all_labels[positions]

            updated_std_of_means_by_class = {self.reverse_mapping[k]: 1 for k, _ in std_of_means_by_class.items()}
            self.class_portion.append(updated_std_of_means_by_class)
            updated_std_of_means_by_task = {k: 1 for k, v in std_of_means_by_task.items()}
            dist_task_before = distribute_samples(updated_std_of_means_by_task, self.args.buffer_size)

            if self.task > 1:
                dist_task = adjust_values_integer_include_all(dist_task_before.copy(), self.dist_task_prev)
            else:
                dist_task = dist_task_before

            dist_class = [distribute_samples(self.class_portion[i], dist_task[i]) for i in range(self.task)]
            self.dist_task_prev = dist_task

            dist = dist_class.pop()
            dist_last = dist.copy()
            dist = {self.mapping[k]: v for k, v in dist.items()}

            counter_class = [0 for _ in range(len(self.unique_classes))]
            condition = [dist[k] for k in range(len(dist))]

            check_bound = self.n_sample_per_task // len(self.unique_classes)
            for i in range(len(condition)):
                if condition[i] > check_bound:
                    condition = distribute_excess(condition, check_bound)
                    break

            images_list_ = []
            labels_list_ = []
            for i in range(all_labels.shape[0]):
                if counter_class[self.mapping[all_labels[i].item()]] < condition[self.mapping[all_labels[i].item()]]:
                    counter_class[self.mapping[all_labels[i].item()]] += 1
                    labels_list_.append(all_labels[i])
                    images_list_.append(all_images[i])
                if counter_class == condition:
                    break

            all_images_ = torch.stack(images_list_).to(self.device)
            all_labels_ = torch.stack(labels_list_).to(self.device)

            counter_manage = [{k: 0 for k, __ in dist_class[i].items()} for i in range(self.task - 1)]
            dist_class_merged = {}
            counter_manage_merged = {}
            dist_class_merged_prev = {}

            for d in dist_class:
                dist_class_merged.update(d)
            for f in counter_manage:
                counter_manage_merged.update(f)
            if self.task > 1:
                dist_class_merged_prev = self.dist_class_prev
                class_key = list(dist_class_merged.keys())
                temp_key = -1
                for k, value in dist_class_merged.items():
                    temp_key += 1
                    if value > dist_class_merged_prev[k]:
                        temp = value - dist_class_merged_prev[k]
                        dist_class_merged[k] -= temp
                        for hh in range(temp):
                            dist_class_merged[class_key[temp_key + hh + 1]] += 1

            self.dist_class_prev = dist_class_merged.copy()
            self.dist_class_prev.update(dist_last)

            if not self.buffer.is_empty():
                images_store = []
                labels_store = []
                for i in range(len(self.buffer)):
                    if counter_manage_merged[self.buffer.labels[i].item()] < dist_class_merged[self.buffer.labels[i].item()]:
                        counter_manage_merged[self.buffer.labels[i].item()] += 1
                        labels_store.append(self.buffer.labels[i])
                        images_store.append(self.buffer.examples[i])
                    if counter_manage_merged == dist_class_merged:
                        break
                images_store_ = torch.stack(images_store).to(self.device)
                labels_store_ = torch.stack(labels_store).to(self.device)
                all_images_ = torch.cat((images_store_, all_images_))
                all_labels_ = torch.cat((labels_store_, all_labels_))

            if not hasattr(self.buffer, 'examples'):
                self.buffer.init_tensors(all_images_, all_labels_, None, None)

            self.buffer.num_seen_examples += self.n_sample_per_task
            self.buffer.labels = all_labels_
            self.buffer.examples = all_images_

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]

        batch_x, batch_y = inputs, labels
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_x_combine = batch_x
        batch_y_combine = batch_y

        self.opt.zero_grad()

        # === DIAG: decide whether to run the eval forward pass on this step ===
        # Original CGR: only during the first E epochs (for variance computation).
        # With diag logging: for ALL epochs of task 1, so we get target confidence,
        # margin, correctness, and per-sample loss from the SAME eval-mode pass at
        # every epoch (needed for forgetting events and the diagnostic table).
        run_eval_pass = self.epoch < self.args.E
        if self._diag_active() and self.epoch < self.args.n_epochs:
            run_eval_pass = True
        # === END DIAG ===

        if run_eval_pass:
            targets = torch.tensor([self.mapping[val.item()] for val in labels]).to(self.device)
            confidence_batch = []
            self.net.eval()
            with torch.no_grad():
                cgr_logits = self.net(not_aug_inputs)
                soft_ = nn.functional.softmax(cgr_logits, dim=1)
                # Existing: per-sample target confidence into self.confidence_by_sample
                for i in range(targets.shape[0]):
                    confidence_batch.append(soft_[i, labels[i]].item())
                conf_tensor = torch.tensor(confidence_batch)
                self.confidence_by_sample[self.epoch, index_] = conf_tensor

                # === DIAG: record margin / correctness / per-sample loss from same eval pass ===
                if self._diag_active():
                    labels_dev = labels.to(self.device).long()
                    target_prob = soft_.gather(1, labels_dev.unsqueeze(1)).squeeze(1)
                    soft_other = soft_.clone()
                    soft_other.scatter_(1, labels_dev.unsqueeze(1), float('-inf'))
                    max_other = soft_other.max(dim=1)[0]
                    margin = (target_prob - max_other).cpu()

                    pred = cgr_logits.argmax(dim=1)
                    correct = (pred == labels_dev).cpu()

                    per_sample_loss = F.cross_entropy(cgr_logits, labels_dev, reduction='none').cpu()

                    if torch.is_tensor(index_):
                        idx_cpu = index_.detach().cpu().long()
                    else:
                        idx_cpu = torch.as_tensor(index_, dtype=torch.long)
                    self.diag_margin[self.epoch, idx_cpu] = margin
                    self.diag_correct[self.epoch, idx_cpu] = correct
                    self.diag_loss[self.epoch, idx_cpu] = per_sample_loss
                    self.diag_labels[idx_cpu] = labels.detach().cpu().long()
                # === END DIAG ===
            self.net.train()

        # SGD forward + backward (unchanged)
        if self.buffer.is_empty():
            logits = self.net(batch_x_combine)
            novel_loss = self.loss(logits, batch_y_combine)
        else:
            mem_x, mem_y = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            mem_x = mem_x.to(self.device)
            mem_y = mem_y.to(self.device)
            mem_x_combine = mem_x
            mem_y_combine = mem_y
            combined_inputs = torch.cat([mem_x_combine, batch_x_combine])
            combined_labels = torch.cat((mem_y_combine, batch_y_combine))
            combined_logits = self.net(combined_inputs)
            novel_loss = self.loss(combined_logits, combined_labels)

        novel_loss.backward()
        self.opt.step()

        return novel_loss.item()
