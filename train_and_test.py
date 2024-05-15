import logging
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score


def valids(model, test_loader, device):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        loss = []
        iteration = 0
        for data_sample in test_loader:
            y = data_sample['label'].to(device)
            outputs, _, _ = model(
                data_sample['ligase_ligand'].to(device),
                data_sample['ligase'].to(device),
                data_sample['target_ligand'].to(device),
                data_sample['target'].to(device),
                data_sample['linker'].to(device),
            )
            loss_val = criterion(outputs, y)
            loss.append(loss_val.item())
            y_score = y_score + torch.nn.functional.softmax(outputs, 1)[:, 1].cpu().tolist()
            y_pred = y_pred + torch.max(outputs, 1)[1].cpu().tolist()
            y_true = y_true + y.cpu().tolist()
            iteration += 1
        model.train()
    return sum(loss) / iteration, accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_pred), precision_score(y_true,
                                                                                                                 y_pred), recall_score(
        y_true, y_pred), average_precision_score(y_true, y_pred)


def train(model, lr=0.0001, epoch=30, train_loader=None, valid_loader=None, device=None, writer=None, LOSS_NAME=None,
          batch_size=None):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    _ = valids(model, valid_loader, device)
    running_loss = 0.0
    weight = torch.Tensor([0.8, 0.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    best_model_params = None
    best_val_auc = float('-INF')
    for epo in range(epoch):
        total_num = 0
        for data_sample in train_loader:
            outputs, _, _ = model(
                data_sample['ligase_ligand'].to(device),
                data_sample['ligase'].to(device),
                data_sample['target_ligand'].to(device),
                data_sample['target'].to(device),
                data_sample['linker'].to(device),
            )
            total_num += batch_size
            y = data_sample['label'].to(device)
            loss = criterion(outputs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
        writer.add_scalar(LOSS_NAME + "_train", running_loss / total_num, epo)
        logging.info('Train epoch %d, loss: %.4f' % (epo, running_loss / total_num))
        val_loss, val_acc, auroc, precision, recall, AUPR = valids(model, valid_loader, device)
        writer.add_scalar(LOSS_NAME + "_test_loss", val_loss, epo)
        writer.add_scalar(LOSS_NAME + "_test_acc", val_acc, epo)
        writer.add_scalar(LOSS_NAME + "_test_auroc", auroc, epo)
        writer.add_scalar(LOSS_NAME + "_precision", precision, epo)
        writer.add_scalar(LOSS_NAME + "_recall", recall, epo)
        writer.add_scalar(LOSS_NAME + "_AUPR", AUPR, epo)

        if auroc > best_val_auc:
            best_val_auc = auroc
            best_model_params = model.state_dict()

        logging.info(
            f'Valid epoch {epo} loss:{val_loss}, acc: {val_acc}, auroc: {auroc}, precision: {precision}, recall: {recall}, AUPR: {AUPR}')
        running_loss = 0.0

    torch.save(best_model_params, f"model/{LOSS_NAME}.pt")
    return model
