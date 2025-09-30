from calendar import c
from math import log
from model import BrainCancer
import torch
from torch import nn
import numpy as np
import os
import sys
sys.path.append('/rhome/ssafa013/DGDDPM/DGDDPM/wilds')
from torch import optim
from tensorboardX import SummaryWriter
import math
from DataLoader import OpenBHBDataset
from collections import defaultdict

from wilds.common.data_loaders import get_train_loader, get_eval_loader
from utils import debug_function, log_print, flatten_layer_param_list_for_model, flatten_layer_param_list_for_flower
TOTAL_DATASET_SIZE = 1587
train_batch_size=16
num_domains = 15
NUM_ROUNDS = 15
epochs = 75 + 1
import time
torch.set_num_threads(32)   # or fewer depending on your benchmarking

# @debug_function(context="CLIENT")
# def train_dann3d_model(epoch, dataloader, model, optimizer):
#     _ = model.train()

#     y_preds = []
#     y_trues = []
#     d_losses = []
#     p_losses = []
#     losses = []
#     accs = []
#     loss_metric = nn.MSELoss()
#     acc_metric = nn.L1Loss()
#     domain_metric = nn.CrossEntropyLoss()
#     lambda_d = 0.225 # 2.0 / (1.0 + math.exp(-10 * progress)) - 1.0


#     for i, (image, label, metadata) in enumerate(dataloader):
#         optimizer.zero_grad()
#         domain_real = metadata[:, 0].long().to(image.device)   # shape (B,)
#         # Upper edges of each interval (placed on the same device)
#         boundaries = torch.tensor([3, 6, 9, 12], device=domain_real.device)

#         # For x < 3  → 0
#         # 3 ≤ x < 6  → 1
#         # 6 ≤ x < 9  → 2
#         # 9 ≤ x < 12 → 3
#         # x ≥ 12     → 4
#         domain_label = torch.bucketize(domain_real, boundaries)   # shape (B,), dtype=torch.long

#         # # ---- build the domain-label tensor ----
#         # domain_label = torch.full(                    # shape (B,)
#         #     (image.size(0),),                      # batch size
#         #     client_id,                        # constant value
#         #     dtype=torch.long
#         # )
#         # log_print(f"Epoch {epoch}, Batch {i}: image shape={image.shape}, label shape={label.shape}", context="CLIENT TRAINING")
#         if torch.cuda.is_available():
            
#             image = image.cuda()
#             label = label.cuda()
#             domain_label = domain_label.cuda()
#             # weight = weight.cuda()

#         prediction, domain_prediction = model(image.float())
#         prediction = prediction.view(-1)  # Flattens [batch_size, 1] → [batch_size]
#         label = label.view(-1)
#         # print(prediction['y_pred'])
#         # lable_box = Box({'y_trues': label})
#         # print("prediction: ", prediction['y_pred'].shape)
#         # print("lable: ", lable_box['y_trues'])
#         prediction_loss = loss_metric(prediction, label)
#         domain_loss = domain_metric(domain_prediction, domain_label.long())

#         # log_print(f"Epoch {epoch}, Batch {i}: label is: {label}, prediction is: {prediction}", context="CLIENT TRAINING")
#         # log_print(f"Epoch {epoch}, Batch {i}: Loss={loss.item():.4f}", context="CLIENT TRAINING")
        
#         # print('prediction: ', prediction['y_pred'], "actual: ", lable_box['y_trues'])
#         # loss = prediction_loss + lambda_d*domain_loss
#         loss = prediction_loss*(1-lambda_d) + lambda_d * domain_loss

#         # print('loss is :', float(loss))
#         # print(prediction['y_pred'].shape, lable_box['y_trues'].shape)
#         loss.backward()
#         optimizer.step()
        
#         y_preds.extend(prediction.detach().cpu().view(-1).tolist())
#         y_trues.extend(label.detach().cpu().view(-1).tolist())

#         # print('prediction :', y_preds)
#         # print("accuracy is : {0:.16f}".format( metrics.accuracy_score(y_trues, y_preds)))
#         acc = acc_metric(prediction, 
#                         label)

#         loss_value = loss.item()
#         d_losses.append(domain_loss.item())
#         p_losses.append(prediction_loss.item())
#         losses.append(loss_value)
#         accs.append(acc.item())

#     # log_print(f"Epoch {epoch}: Loss={losses}, ACC={accs}", context="CLIENT TRAINING")
#     return np.mean(losses), np.mean(accs), np.mean(d_losses), np.mean(p_losses)

# @debug_function(context="CLIENT EVALUATION")
# @torch.no_grad()
# def evaluate_dann3d_model(dataloader, model):
#     model.eval()

#     y_preds = []
#     y_trues = []
#     d_losses = []
#     p_losses = []
#     losses = []
#     accs = []
#     loss_metric = nn.MSELoss()
#     acc_metric = nn.L1Loss()
#     domain_metric = nn.CrossEntropyLoss()
#     lambda_d = 0.225
    
#     for i, (image, label, metadata) in enumerate(dataloader):
#         domain_label = torch.full(                    # shape (B,)
#             (image.size(0),),                      # batch size
#             0,                        # constant value
#             dtype=torch.long
#         )  
#         if torch.cuda.is_available():
#             image = image.cuda()
#             label = label.cuda()
#             domain_label = domain_label.cuda()

#         prediction, domain_prediction = model(image.float())
#         label = torch.squeeze(label, dim=[1])

#         prediction_loss = loss_metric(prediction, label)
#         domain_loss = domain_metric(domain_prediction, domain_label.long())
        
#         y_pred = prediction.detach().cpu().numpy().flatten()
#         y_true = label.detach().cpu().numpy().flatten()

#         loss = prediction_loss*(1-lambda_d) + lambda_d*domain_loss

#         y_preds.extend(y_pred)
#         y_trues.extend(y_true)

#         acc = acc_metric(prediction, 
#                         label)
        
#         d_losses.append(domain_loss.item())
#         p_losses.append(prediction_loss.item())
#         losses.append(loss.item())
#         accs.append(acc.item())

#     avg_d_loss = np.mean(d_losses)
#     avg_p_loss = np.mean(p_losses)
#     avg_loss = np.mean(losses)
#     avg_acc = np.mean(accs)

#     return avg_loss, avg_acc, avg_d_loss, avg_p_loss

@debug_function(context="CLIENT")
def train_normal_model(epoch, dataloader, model, optimizer): #(self, epoch, writer, log_every=100)

    _ = model.train()

    y_preds = []
    y_trues = []
    losses = []
    aucs = []
    loss_metric = nn.MSELoss()
    auc_metric = nn.L1Loss()
    

    for i, (image, label, metadata) in enumerate(dataloader):
        
        
        optimizer.zero_grad()
        # log_print(f"Epoch {epoch}, Batch {i}: image shape={image.shape}, label shape={label.shape}", context="CLIENT TRAINING")
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            # weight = weight.cuda()

        prediction = model(image.float())
        # prediction = prediction.view(-1)  # Flattens [batch_size, 1] → [batch_size]
        # label = label.view(-1)
        # print(prediction['y_pred'])
        # lable_box = Box({'y_trues': label})
        # print("prediction: ", prediction['y_pred'].shape)
        # print("lable: ", lable_box['y_trues'])
        loss = loss_metric(prediction, label)
        # log_print(f"Epoch {epoch}, Batch {i}: label is: {label}, prediction is: {prediction}", context="CLIENT TRAINING")
        # log_print(f"Epoch {epoch}, Batch {i}: Loss={loss.item():.4f}", context="CLIENT TRAINING")
        
        # print('prediction: ', prediction['y_pred'], "actual: ", lable_box['y_trues'])
        
        # print('loss is :', float(loss))
        # print(prediction['y_pred'].shape, lable_box['y_trues'].shape)
        loss.backward()
        optimizer.step()
        
        y_preds.extend(prediction.detach().cpu().view(-1).tolist())
        y_trues.extend(label.detach().cpu().view(-1).tolist())

        # print('prediction :', y_preds)
        # print("accuracy is : {0:.16f}".format( metrics.accuracy_score(y_trues, y_preds)))
        acc = auc_metric(prediction, 
                        label)

        loss_value = loss.item()
        losses.append(loss_value)
        aucs.append(acc.item())

    # log_print(f"Epoch {epoch}: Loss={losses}, acc={aucs}", context="CLIENT TRAINING")
    return np.mean(losses), np.mean(aucs)
    

@debug_function(context="CLIENT EVALUATION")
@torch.no_grad()
def evaluate_normal_model(dataloader, model, scheduler=None):
    model.eval()

    y_preds = []
    y_trues = []
    losses = []
    aucs = []
    site_sums = defaultdict(float)
    site_counts = defaultdict(int)
    loss_metric = nn.MSELoss()
    auc_metric = nn.L1Loss()

    for i, (image, label, metadata) in enumerate(dataloader):
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        prediction = model(image.float())
        # label = torch.squeeze(label, dim=[1])

        loss = loss_metric(prediction, label)
        err = (prediction - label).pow(2).view(-1).cpu()      # <-- NOT nn.MSELoss
        sites = metadata[:, 0].long()                         # shape [B]
        for e, s in  zip(err, sites):
            site_sums[s.item()] += e.item()
            site_counts[s.item()] += 1
        
        y_pred = prediction.detach().cpu().numpy().flatten()
        y_true = label.detach().cpu().numpy().flatten()

        y_preds.extend(y_pred)
        y_trues.extend(y_true)

        acc = auc_metric(prediction, 
                        label)

        losses.append(loss.item())
        aucs.append(acc.item())

    avg_loss = np.mean(losses)
    avg_auc = np.mean(aucs)
    site_mse = {s: site_sums[s]/site_counts[s] for s in site_sums}
    macro_mse = sum(site_mse.values())/len(site_mse)   # un-weighted mean
    scheduler.step(avg_loss)

    return avg_loss, avg_auc, site_mse, macro_mse


def run_single_experiment(global_run_id, repeat_idx):
    run_id = global_run_id #time.strftime("%Y-%m-%d_%H-%M-%S")  # Timestamp-based run ID
    run_dir = os.path.join("./runs", f"run_{run_id}")  # Each run has a separate directory
    os.makedirs(run_dir, exist_ok=True)
    server_log_dir = os.path.join(run_dir, "server_log")
    os.makedirs(server_log_dir, exist_ok=True)  # Create directory if it doesn't exist
    train_log_path = os.path.join(server_log_dir, "train_metrics.txt")
    validation_log_path = os.path.join(server_log_dir, "validation_metrics.txt")
    
    ds = OpenBHBDataset()
    train_dataset = ds.get_subset('train')
    val_dataset = ds.get_subset('val')
    model = BrainCancer()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=8e-4) #best : 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_loader = get_train_loader("standard", train_dataset, batch_size=train_batch_size, num_workers=8)
    validation_loader = get_eval_loader("standard", val_dataset, batch_size=1, num_workers=8)
    logs_train = []
    logs_val = []
    for i in range(epochs):
        loss, acc = train_normal_model(i, train_loader, model, optimizer)
        logs_train.append((int(i), float(loss), float(acc)))
        if i%5 == 0:
            loss, acc, site_mse, macro_mse = evaluate_normal_model(validation_loader, model, scheduler)
            logs_val.append((int(i/5), float(loss), float(acc), site_mse, float(macro_mse)))

    with open(train_log_path, "a") as f:
        f.write(f"\n\n====== Run {repeat_idx + 1} ======\n")
        f.write(f"The Dataset Size is : {len(train_dataset)}\n")
        for epoch, loss, acc in logs_train:
            f.write(f"Epoch {epoch + 1}: Loss={loss:.4f}, mae={acc:.4f}\n")
    with open(validation_log_path, "a") as f:
        f.write(f"\n\n====== Run {repeat_idx + 1} ======\n")
        f.write(f"The Dataset Size is : {len(val_dataset)}\n")
        for epoch, loss, acc, site_mse, macro_mse in logs_val:
            f.write(f"Epoch {epoch + 1}: Loss={loss:.4f}, mae={acc:.4f}, macro_mse={macro_mse:.4f}\n")
            site_mse_str = '\n '.join([f'Site {k}: {v:.4f}' for k, v in site_mse.items()])
            f.write(f"Site MSE:\n {site_mse_str}\n")  

    # Save model checkpoint
    model_path = os.path.join(run_dir, f"model_run{repeat_idx + 1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    return logs_val


if __name__ == "__main__":            
    # run_id = time.strftime("%Y-%m-%d_%H-%M-%S")  # Timestamp-based run ID
    global_run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    all_final_val_accs = []

    for repeat_idx in range(10):
        print(f"\n--- Running experiment {repeat_idx + 1}/10 ---\n")
        logs_val = run_single_experiment(global_run_id, repeat_idx)

        # Only take the last validation result
        if logs_val:
            _, final_loss, final_acc, site_mse, macro_mse = logs_val[-1]
            all_final_val_accs.append((final_loss, final_acc, site_mse, macro_mse))

    # Compute mean and std
    losses = [r[0] for r in all_final_val_accs]
    accs = [r[1] for r in all_final_val_accs]
    per_site_mse = [r[2] for r in all_final_val_accs]
    macro_mse = [r[3] for r in all_final_val_accs]
    
    mean_loss, std_loss = np.mean(losses), np.std(losses)
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_per_site_mse = {k: np.mean([r[k] for r in per_site_mse]) for k in per_site_mse[0]}
    mean_macro_mse = np.mean(macro_mse)
    # Save average results
    run_dir = os.path.join("./runs", f"run_{global_run_id}")  # Each run has a separate directory
    summary_path = os.path.join(run_dir, f"summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}\n")
        f.write(f"Validation MAE: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Mean Per-Site MSE:\n")
        for site, mse in mean_per_site_mse.items():
            f.write(f"  Site {site}: {mse:.4f}\n")
        f.write(f"Mean Macro MSE: {mean_macro_mse:.4f}\n")
        
    print("\n✅ All runs completed.")
    print(f"📄 Summary saved to {summary_path}")