import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel

import torch.nn as nn
import numpy as np
##import matplotlib.pyplot as plt
import torchvision

import torch.nn.functional as F
import math

from collections import defaultdict
import random


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Class-Adaptive Sampling Policy.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--E', type=int, default=4,
                        help='Epoch for strategies')
    
    return parser


def distribute_samples(probabilities, M):
    # Normalize the probabilities
    total_probability = sum(probabilities.values())
    normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}

    # Calculate the number of samples for each class
    samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
    
    # Check if there's any discrepancy due to rounding and correct it
    discrepancy = M - sum(samples.values())
    
    # Adjust the number of samples in each class to ensure the total number of samples equals M
    for key in samples:
        if discrepancy == 0:
            break    # Stop adjusting if there's no discrepancy
        if discrepancy > 0:
            # If there are less samples than M, add a sample to the current class and decrease discrepancy
            samples[key] += 1
            discrepancy -= 1
        elif discrepancy < 0 and samples[key] > 0:
            # If there are more samples than M and the current class has samples, remove one and increase discrepancy
            samples[key] -= 1
            discrepancy += 1

    return samples    # Return the final classes distribution

    
def distribute_excess(lst, check_bound):
    # Calculate the total excess value
    total_excess = sum(val - check_bound for val in lst if val > check_bound)

    # Number of elements that are not greater than check_bound
    recipients = [i for i, val in enumerate(lst) if val < check_bound]

    num_recipients = len(recipients)

    # Calculate the average share and remainder
    avg_share, remainder = divmod(total_excess, num_recipients)

    lst = [val if val <= check_bound else check_bound for val in lst]
    
    # Distribute the average share
    for idx in recipients:
        lst[idx] += avg_share
    
    # Distribute the remainder
    for idx in recipients[:remainder]:
        lst[idx] += 1
    
    # Cap values greater than check_bound
    for i, val in enumerate(lst):
        if val > check_bound:
            return distribute_excess(lst, check_bound)
            break

    return lst


def adjust_values_integer_include_all(a, b):
    excess = {}
    shortage = {}
    total_excess = 0

    # Establish initial excess and shortage based on the limits in b
    for k in a:
        if k in b:
            if a[k] > b[k]:
                excess[k] = a[k] - b[k]
                total_excess += a[k] - b[k]
                a[k] = b[k]  # Adjust to the limit of b
            elif a[k] < b[k]:
                shortage[k] = b[k] - a[k]  # Available space to increase
        else:
            # If no corresponding key in b, treat as having no upper limit
            shortage[k] = float('inf')  # Theoretically unlimited capacity

    # Distribute the excess to those under the limit as integers
    while total_excess > 0 and shortage:
        per_key_excess = max(total_excess // len(shortage), 1)  # Ensure minimal distribution
        for k in list(shortage):
            if total_excess == 0:
                break
            if shortage[k] == float('inf'):
                increment = per_key_excess  # No limit, so use per_key_excess
            else:
                increment = min(shortage[k], per_key_excess)

            a[k] += increment
            total_excess -= increment

            if shortage[k] != float('inf'):
                shortage[k] -= increment
                if shortage[k] == 0:
                    del shortage[k]  # Remove key from shortage if fully adjusted

    # Ensure all values are integers
    for key in a:
        a[key] = int(a[key])

    return a


class Cgr(ContinualModel):
    NAME = 'cgr'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Cgr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        ##self.transform = None
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

    def begin_train(self, dataset):
        self.n_sample_per_task = dataset.get_examples_number()//dataset.N_TASKS
    
    def begin_task(self, dataset, train_loader):
        self.epoch = 0
        self.task += 1
        self.unique_classes = set()
        for _, labels, _, _ in train_loader:
            self.unique_classes.update(labels.numpy())
            if len(self.unique_classes)==dataset.N_CLASSES_PER_TASK:
                break
        self.mapping = {value: index for index, value in enumerate(self.unique_classes)}
        self.reverse_mapping = {index: value for value, index in self.mapping.items()}
        self.confidence_by_sample = torch.zeros((self.args.n_epochs, self.n_sample_per_task))
    
    def end_epoch(self, dataset, train_loader):
        
        self.epoch += 1
        
        if self.epoch == self.args.n_epochs:
            
            # Calculate standard deviation of mean confidences by class
            std_of_means_by_class = {class_id: 1 for class_id, __ in enumerate(self.unique_classes)}
            std_of_means_by_task = {task_id: 1 for task_id in range(self.task)}
            
            # Compute mean and variability of confidences for each sample
            Confidence_mean = self.confidence_by_sample[:self.args.E].mean(dim=0)
            Variability = self.confidence_by_sample[:self.args.E].var(dim=0)
            
        
            # Sort indices based on the Confidence
            ##sorted_indices_1 = np.argsort(Confidence_mean.numpy())
            
            # Sort indices based on the variability
            sorted_indices_2 = np.argsort(Variability.numpy())
            
        
            ##top_indices_sorted = sorted_indices_1 #hard
            
            ##top_indices_sorted = sorted_indices_1[::-1].copy() #simple
        
            # Descending order
            top_indices_sorted = sorted_indices_2[::-1].copy() #challenging

            
            # Convert standard deviation of means by class to item form
            updated_std_of_means_by_class = {self.reverse_mapping[k]: 1 for k, _ in std_of_means_by_class.items()}   #uncomment for balance

            self.class_portion.append(updated_std_of_means_by_class)
            
            updated_std_of_means_by_task = {k: 1 for k, v in std_of_means_by_task.items()}    #uncomment for balance
            dist_task_before = distribute_samples(updated_std_of_means_by_task, self.args.buffer_size)
            
            if self.task > 1:
                dist_task = adjust_values_integer_include_all(dist_task_before.copy(), self.dist_task_prev)
            else:
                dist_task = dist_task_before
            
            dist_class = [distribute_samples(self.class_portion[i], dist_task[i]) for i in range(self.task)]
            
            self.dist_task_prev = dist_task
            
            # Distribute samples based on the standard deviation
            dist = dist_class.pop()
            dist_last = dist.copy()
            dist = {self.mapping[k]: v for k, v in dist.items()}

            
            # Initialize a counter for each class
            counter_class = [0 for _ in range(len(self.unique_classes))]
        
            # Distribution based on the class variability
            condition = [dist[k] for k in range(len(dist))]
        
            # Check if any class exceeds its allowed number of samples
            check_bound = self.n_sample_per_task//len(self.unique_classes)
            for i in range(len(condition)):
                if condition[i] > check_bound:
                    # Redistribute the excess samples
                    condition = distribute_excess(condition, check_bound)
                    break
        
            # Assuming train_loader is defined and each batch consists of (inputs, labels)
            class_samples = defaultdict(list)
            
            for inputs_1, labels_1, not_aug_inputs_1, indices_1 in train_loader:
                for input, label in zip(not_aug_inputs_1, labels_1):
                    class_samples[label.item()].append((input, label))

            desired_samples = condition
            selected_data = []
            
            for label, samples in class_samples.items():
                n_samples = desired_samples[self.mapping[label]]
                if len(samples) >= n_samples:
                    selected_data.extend(random.sample(samples, n_samples))
                else:
                    print(f"Not enough samples for class {label}, needed {n_samples}, but got {len(samples)}")


            # Extracting images and labels into separate lists
            images11 = [data[0] for data in selected_data]  # data[0] is the image tensor
            labels11 = [data[1] for data in selected_data]  # data[1] is the label tensor

            # Convert lists of tensors to single tensors
            all_images_ = torch.stack(images11, dim=0).to(self.device)  # Stacks along a new dimension
            all_labels_ = torch.stack(labels11, dim=0).to(self.device)  # Stacks along a new dimension
        
            
            counter_manage = [{k:0 for k, __ in dist_class[i].items()} for i in range(self.task - 1)]

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
                # Assuming train_loader is defined and each batch consists of (inputs, labels)
                class_samples_buffer = defaultdict(list)
                
                for input, label in zip(self.buffer.examples, self.buffer.labels):
                    class_samples_buffer[label.item()].append((input, label))
    
                desired_samples_buffer = dist_class_merged
                selected_data_buffer = []
                
                for label, samples in class_samples_buffer.items():
                    n_samples = desired_samples_buffer[label]
                    if len(samples) >= n_samples:
                        selected_data_buffer.extend(random.sample(samples, n_samples))
                    else:
                        print(f"Not enough samples for class {label}, needed {n_samples}, but got {len(samples)}")
    
    
                # Extracting images and labels into separate lists
                images11_buffer = [data[0] for data in selected_data_buffer]  # data[0] is the image tensor
                labels11_buffer = [data[1] for data in selected_data_buffer]  # data[1] is the label tensor
    
                # Convert lists of tensors to single tensors
                images_store_ = torch.stack(images11_buffer, dim=0).to(self.device)  # Stacks along a new dimension
                labels_store_ = torch.stack(labels11_buffer, dim=0).to(self.device)  # Stacks along a new dimension


                all_images_ = torch.cat((images_store_, all_images_))
                all_labels_ = torch.cat((labels_store_, all_labels_))

            if not hasattr(self.buffer, 'examples'):
                self.buffer.init_tensors(all_images_, all_labels_, None, None)
            
            self.buffer.num_seen_examples += self.n_sample_per_task
            
            # Update the buffer with the shuffled images and labels
            self.buffer.labels = all_labels_
            self.buffer.examples = all_images_
    

    def observe(self, inputs, labels, not_aug_inputs, index_):
        
        real_batch_size = inputs.shape[0]
        
        # batch update
        batch_x, batch_y = inputs, labels
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_x_combine = batch_x
        batch_y_combine = batch_y
            
        self.opt.zero_grad()

        if self.epoch < self.args.E:
            targets = torch.tensor([self.mapping[val.item()] for val in labels]).to(self.device)
            confidence_batch = []
            self.net.eval()
            with torch.no_grad():
                cgr_logits = self.net(not_aug_inputs)
                soft_ = nn.functional.softmax(cgr_logits, dim=1)
                # Accumulate confidences
                for i in range(targets.shape[0]):
                    confidence_batch.append(soft_[i,labels[i]].item())
                
                # Record the confidence scores for samples in the corresponding tensor
                conf_tensor = torch.tensor(confidence_batch)
                self.confidence_by_sample[self.epoch, index_] = conf_tensor
            self.net.train()
    
        
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
