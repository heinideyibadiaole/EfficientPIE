"""

@ Description:
@ Project:APCIL
@ Author:qufang
@ Create:2024/6/8 16:37

"""
import sys

import numpy as np
import sklearn.metrics
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)

    TP = torch.zeros(1).to(device)
    TN = torch.zeros(1).to(device)
    FP = torch.zeros(1).to(device)
    FN = torch.zeros(1).to(device)

    optimizer.zero_grad()
    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(dataloader):
        images, labels = data
        pred = model(images.to(device))
        robust_pred = robust_noisy(pred, epoch)
        pred_classes = torch.max(pred, dim=1)[1]
        labels = labels.to(device)
        # Calculate TP, TN, FP, FN
        TP += ((pred_classes == 1) & (labels == 1)).sum().float()
        TN += ((pred_classes == 0) & (labels == 0)).sum().float()
        FP += ((pred_classes == 1) & (labels == 0)).sum().float()
        FN += ((pred_classes == 0) & (labels == 1)).sum().float()
        # compute the loss
        # loss = loss_function(pred, labels)
        loss = loss_function(robust_pred, labels)
        loss.backward()
        # we should call detach() function to avoid the influence from the computation of accu_loss
        accu_loss += loss.detach()
        # compute the performance
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        dataloader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, precision: {:.3f}, " \
                          "recall: {:.3f}, f1_score: {:.3f}" \
            .format(epoch, accu_loss.item() / (step + 1), accuracy.item(), precision.item(), recall.item(),
                    f1_score.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accuracy.item(), precision.item(), recall.item(), f1_score.item()


@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)

    TP = torch.zeros(1).to(device)
    TN = torch.zeros(1).to(device)
    FP = torch.zeros(1).to(device)
    FN = torch.zeros(1).to(device)
    preds_all = []  # for auc
    labels_all = []  # for auc

    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(dataloader):
        # images, labels = data
        images, labels = data
        pred = model(images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        labels = labels.to(device)
        # Calculate TP, TN, FP, FN
        TP += ((pred_classes == 1) & (labels == 1)).sum().float()
        TN += ((pred_classes == 0) & (labels == 0)).sum().float()
        FP += ((pred_classes == 1) & (labels == 0)).sum().float()
        FN += ((pred_classes == 0) & (labels == 1)).sum().float()

        # compute the loss
        loss = loss_function(pred, labels)
        accu_loss += loss
        # compute the performance
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        # compute AUC by scikit-learn
        preds_all.extend(pred[:, 1].cpu().numpy())
        labels_all.extend(labels.cpu())
        auc_score = roc_auc_score(labels_all, preds_all)
        # the description of console
        dataloader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, precision: {:.3f}, " \
                          "recall: {:.3f}, f1_score: {:.3f}, auc: {:.3f}" \
            .format(epoch, accu_loss.item() / (step + 1), accuracy.item(), precision.item(), recall.item(),
                    f1_score.item(), auc_score)

    return accu_loss.item() / (step + 1), accuracy.item(), precision.item(), recall.item(), f1_score.item()


def pre_train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(dataloader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        dataloader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def pre_evaluate(model, dataloader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(dataloader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        dataloader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def robust_noisy(pred, epoch):
    max_range = 0.5 * (epoch / 30)
    random_num = torch.rand(1).item() * (2 * max_range) - max_range
    noisy = torch.tensor([random_num, -random_num], dtype=pred.dtype, device=pred.device)
    robust_pred = pred + noisy
    return robust_pred


def incremental_learning_train(model, optimizer, dataloader, device, epoch, prev_model):
    model.train()
    loss_new_func = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)

    TP = torch.zeros(1).to(device)
    TN = torch.zeros(1).to(device)
    FP = torch.zeros(1).to(device)
    FN = torch.zeros(1).to(device)

    optimizer.zero_grad()
    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(dataloader):
        images, labels = data
        pred = model(images.to(device))
        # perturbation
        robust_pred = robust_noisy(pred, epoch)
        pred_classes = torch.max(pred, dim=1)[1]
        labels = labels.to(device)
        # Calculate TP, TN, FP, FN
        TP += ((pred_classes == 1) & (labels == 1)).sum().float()
        TN += ((pred_classes == 0) & (labels == 0)).sum().float()
        FP += ((pred_classes == 1) & (labels == 0)).sum().float()
        FN += ((pred_classes == 0) & (labels == 1)).sum().float()
        # compute the L_new loss
        loss_new = loss_new_func(robust_pred, labels)
        # compute the L_old loss
        prev_pred = prev_model(images.to(device))
        loss_old = loss_old_func(prev_pred, labels)

        if loss_new > loss_old:
            loss = loss_new
        else:
            loss = 0.5 * loss_old + loss_new

        loss.backward()
        # call detach() function to avoid the influence from the computation of accu_loss
        accu_loss += loss.detach()
        # compute the performance
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        dataloader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, precision: {:.3f}, " \
                          "recall: {:.3f}, f1_score: {:.3f}" \
            .format(epoch, accu_loss.item() / (step + 1), accuracy.item(), precision.item(), recall.item(),
                    f1_score.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accuracy.item(), precision.item(), recall.item(), f1_score.item()


def loss_old_func(prev_pred, labels, temperature=2):
    labels = F.one_hot(labels, num_classes=prev_pred.size(1)).float()  # one-hot encode labels
    prev_pred = F.softmax(prev_pred, dim=-1)

    prev_pred = torch.pow(prev_pred, 1. / temperature)
    labels = torch.pow(labels, 1. / temperature)
    prev_pred = prev_pred / prev_pred.sum(dim=-1, keepdim=True)
    labels = labels / labels.sum(dim=-1, keepdim=True)

    l_prev_pred = torch.log(prev_pred)

    l_prev_pred[l_prev_pred != l_prev_pred] = 0.  # 去除nan值
    loss = torch.mean(torch.sum(-labels * l_prev_pred, axis=1))

    return loss
