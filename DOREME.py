"""
Domain-Aware Data selection for Speech Classification via Meta-Reweighting

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

-------------------------------------------------------------------------
File: my_model.py
 - train the model with speech data frommulti-source domains 
"""

import torch
import numpy as np
import os
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score as f1
from torch.optim import Adam

class LinearModel(nn.Module):
    def __init__(self, features, num_classes):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(features, num_classes, bias=True)

    def forward(self, inputs):
        output = F.log_softmax(self.linear(inputs), -1)
        return output

class MetaLinearModel(nn.Module):
    def __init__(self, features, num_classes):
        super(MetaLinearModel, self).__init__()

        self.linear = nn.Linear(features, num_classes, bias=True)

    def forward(self, inputs):
        output = F.log_softmax(self.linear(inputs), -1)
        return output

    def parameterized(self, x, weights):
        x = nn.functional.linear(x, weights[0], weights[1])
        return x

def toTorch(data: np.ndarray, target: np.ndarray):
    """ Check if data is of type torch.Tensor, if not convert it to torch.Tensor
    Args:
        data (np.ndarray): data to be converted
        target (np.ndarray): target to be converted

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: converted data and target
    """
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)

    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)
        target = target.type(torch.LongTensor)

    if data.dtype != torch.float32:
        data = data.float()

    return data, target

class doreme:
    def __init__(
        self, 
        gpu,
        seed,
        num_classes,
        ):
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.n_classes = num_classes
        self.seed = seed
        self._device = "cuda:"+str(gpu)
        self.model = LinearModel(1024, num_classes).to(self._device)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.init()
        torch.cuda.manual_seed_all(seed)

    def toDevice(self, data: np.ndarray, target: np.ndarray):
        """ Check if data is on the same device as model, if not move it to the device

        Args:
            data (np.ndarray): data to be moved
            target (np.ndarray): target to be moved

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor]: moved data and target
        """
        if data.device != self._device:
            data = data.to(self._device)

        if target.device != self._device:
            target = target.to(self._device)

        return data, target

    def train_step(
        self, 
        data, 
        target,
        instance_weight,
        domain_weight,
        ) -> torch.Tensor:
        """ Perform one training step

        Args:
            data (torch.Tensor): training data
            target (torch.Tensor): training target
            instance_weight(torch.Tensor): instance weights
            domain_weight(torch.Tensor): domain weights

        Returns:
            torch.Tensor: loss
        """
        self.model.train()
        optim = Adam(params=self.model.parameters())
        optim.zero_grad()
        output = self.model(data)
        loss = self.loss(output, target)
        if instance_weight != None:
            loss *= instance_weight
        loss = torch.mean(loss)
        if domain_weight != None:
            loss *= domain_weight
        loss.backward()
        optim.step()

        return loss

    def meta_gradient(self, source_domain_datas, target_data):
        """ Calculate weights of the instances and domains based on their similarity to target domain

        Args:
            source_domain_datas (numpy.ndarray): every data of source domains
            target_data (numpy.ndarray): data of the target domain 

        Returns:
            [torch.Tensor, torch.Tensor]: calculated weights of the instances and domains
        """

        #Initialize instance weights and domain weights to save the gradient
        instance_weights = [torch.ones_like(torch.Tensor(len(sd_data[0]))).to(self._device) for sd_data in source_domain_datas]
        dels = torch.autograd.Variable(torch.zeros_like(torch.Tensor(len(source_domain_datas))).to(self._device),requires_grad=True)

        val_data, val_target = toTorch(target_data[0], target_data[1].astype(np.int32))
        val_data = torch.autograd.Variable(val_data, requires_grad=False).to(self._device)
        val_target = torch.autograd.Variable(val_target, requires_grad=False).to(self._device)

        #Initialize meta model for the domain weights
        del_model = MetaLinearModel(1024, self.n_classes).to(self._device)
        del_model.load_state_dict(self.model.state_dict())
        for i, sd_data in enumerate(source_domain_datas):
            train_data, train_target = toTorch(sd_data[0], sd_data[1].astype(np.int32))
            train_data = torch.autograd.Variable(train_data, requires_grad=False).to(self._device)
            train_target = torch.autograd.Variable(train_target, requires_grad=False).to(self._device)

            #Initialize meta model for the instance weights
            eps_model = MetaLinearModel(1024, self.n_classes).to(self._device)
            eps_model.load_state_dict(self.model.state_dict())
            loss = self.loss(eps_model(train_data), train_target)
            eps = torch.autograd.Variable(torch.zeros(loss.size()).to(self._device),requires_grad=True)
            eps_loss = torch.mean(loss * eps)
            eps_weights = list(eps_model.parameters())

            #Update the parameters of meta model (instance weights)
            eps_grads = torch.autograd.grad(eps_loss, eps_weights, create_graph=True)
            tmp_eps_weights = [param - 0.01 * grad for param, grad in zip(eps_weights, eps_grads)]

            #Calculate gradient of eps respect to validation (target domain train data)
            eps_val_loss = torch.mean(self.loss(eps_model.parameterized(val_data, tmp_eps_weights), val_target))
            grad_eps = torch.autograd.grad(eps_val_loss, eps, only_inputs=True)[0]
            w_ = torch.clamp(-grad_eps, min=0)
            norm = torch.max(w_)
            if norm != 0:
                w_ /= norm
            with torch.no_grad():
                instance_weights[i] += w_
                instance_weights[i] /= 2.0
        
        #Calculate the loss of every data
        del_loss = 0
        for i, sd_data in enumerate(source_domain_datas):
            train_data, train_target = toTorch(sd_data[0], sd_data[1].astype(np.int32))
            train_data = torch.autograd.Variable(train_data, requires_grad=False).to(self._device)
            train_target = torch.autograd.Variable(train_target, requires_grad=False).to(self._device)
            loss = self.loss(del_model(train_data), train_target)
            del_loss += dels[i] * torch.mean(loss * instance_weights[i].to(self._device))

        #Update the parameters of meta model (domain weights)
        del_loss /= len(source_domain_datas)
        del_weights = list(del_model.parameters())
        del_grads = torch.autograd.grad(del_loss, del_weights, create_graph=True)
        tmp_del_weights = [param - 0.01 * grad for param, grad in zip(del_weights, del_grads)]

        #Calculate gradient of dels respect to validation (target domain train data)
        del_val_loss = torch.mean(self.loss(del_model.parameterized(val_data, tmp_del_weights), val_target))
        grad_del = torch.autograd.grad(del_val_loss, dels, only_inputs=True)[0]
        domain_weights = torch.clamp(-grad_del, min=0)
        return instance_weights, domain_weights


    def train(self, train_data, source_domain_datas):
        """ Perform one training epoch
        
        Args:
            train_data (numpy.ndarray): training data
            source_domain_datas (numpy.ndarray): every data of source domains

        Returns:
            float, float: mean loss of the training data of the target domain, f1 score of the training data
        """
        #calculate meta_reweights 
        instance_weights, domain_weights = self.meta_gradient(source_domain_datas, train_data)

        #Train with domain & instance weights
        for i, sd_data in enumerate(source_domain_datas):
            data = sd_data[0]
            target = sd_data[1]
            target = target.astype(np.int32)
            data, target = self.toDevice(*toTorch(data, target))
            loss = self.train_step(data, target, instance_weights[i], domain_weights[i])

        #Train with target domain train dataset
        data = train_data[0]
        target = train_data[1]
        target = target.astype(np.int32)
        data, target = self.toDevice(*toTorch(data, target))
        loss = self.train_step(data, target, None, None)

        loss_mean = torch.mean(loss).item()
        f1_score = f1(self.model(data), target, num_classes=self.n_classes, average='macro').item()
        return loss_mean, f1_score

    def test(self, model, test_data, device, mode="valid"):
        """ Perform one validation epoch

        Args:
            model (nn.Module): model to test
            train_data (numpy.ndarray): training data
            source_domain_datas (numpy.ndarray): every data of source domains
            device (str): gpu device to run
            mode (str): if valid, return only mean loss, if test, return mean loss and f1 score
        
        Returns:
            float, float: mean loss of the training data of the target domain, f1 score of the training data
        """
        # set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            data = test_data[0]
            target = test_data[1]
            target = target.astype(np.int32)
            data, target = toTorch(data, target)
            data = data.to(device)
            target = target.to(device)
            loss = self.loss(model(data), target)

        loss_mean = torch.mean(loss).item()
        if mode == "test":
            f1_score = f1(model(data), target, num_classes=self.n_classes, average='macro').item()
            acc = torch.argmax(model(data), dim=1) == target
            acc = acc.sum().div(len(acc)).item()
            return loss_mean, f1_score, acc
        return loss_mean

    def fit(
        self, 
        train_data,
        source_domain_datas,
        test_data, 
        epochs: int, 
        ):
        """ Train model for a given number of epochs
        
        Args:
            train_data (numpy.ndarray): training data
            source_domain_datas (numpy.ndarray): every data of source domains
            test_data (numpy.ndarray): test data
            epochs (int): number of epochs

        """
        self._epoch = 1
        for epoch in range(1, epochs+1):
            train_loss, train_f1 = self.train(train_data, source_domain_datas)
            test_loss, test_f1, test_acc = self.test(self.model, test_data, self._device, mode="test")

            statement =f"Epoch {epoch}: "
            statement += "TRAIN: Loss: {:.4f} | F1: {:.4f} | TEST: Loss: {:.4f} | F1: {:.4f} | ACC: {:.4f}".format(round(train_loss, 4), round(train_f1, 4), round(test_loss, 4), round(test_f1, 4), round(test_acc,4))
            if epoch == epochs:
                print(statement)
            else:
                print(statement, end='\r',flush=True)

            self._epoch += 1

