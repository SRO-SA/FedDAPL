from calendar import c
from math import log
from model import BrainCancer
from model_feature_regress import BrainCancerFeaturizer, BrainCancerRegressor, DANN3D
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
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from utils import debug_function, log_print, flatten_layer_param_list_for_model, flatten_layer_param_list_for_flower
from collections import defaultdict
from torch.optim.swa_utils import AveragedModel


TOTAL_DATASET_SIZE = 1587
train_batch_size=16
num_domains = 15
NUM_ROUNDS = 15
epochs = 75 + 1
import time
site_counts = [0, 24, 277, 43, 25, 47, 50, 17, 11, 14, 10, 73, 956, 20, 20]
# torch.set_num_threads(32)   # or fewer depending on your benchmarking
path_to_load_ERM = '/rhome/ssafa013/FL_syft/logs/Centralized_ERM/run_2025-07-02_15-36-14/model_run10.pth'
# torch.autograd.set_detect_anomaly(True)   # catches silent “no grad” bugs


def quick_hist_probe_nobg(train_loader,
                          nbins: int = 32,
                          num_domains: int = 15,
                          thresh: float = 0.0):
    """
    Site-classifier on per-scan intensity histograms,
    with all voxels <= thresh treated as background & ignored.
    """
    import numpy as np, sklearn.linear_model as sklm, sklearn.metrics as skm
    H_list, y_list = [], []

    for imgs, _, meta in train_loader:                      # one pass over data
        B = imgs.size(0)
        v = imgs.numpy().reshape(B, -1)                     # (B, Nvox)

        # ---- mask out background -------------------------
        v_masked = [row[row > thresh] for row in v]         # list length B

        # ---- robust global min / max (over non-bg voxels)-
        all_vals = np.concatenate(v_masked)
        vmin, vmax = all_vals.min(), all_vals.max()

        # ---- per-image histograms ------------------------
        hists = [np.histogram(row, bins=nbins, range=(vmin, vmax))[0]
                 for row in v_masked]                       # list of (nbins,)
        hists = np.stack(hists).astype(np.float32)          # (B, nbins)

        # ---- drop first bin (was mostly near-zero) -------
        hists = hists[:, 1:]                                # (B, nbins-1 = 31)

        H_list.append(hists)
        y_list.append(meta[:, 0].numpy())

    X = np.vstack(H_list)                                   # (N, 31)
    y = np.concatenate(y_list)

    clf = sklm.LogisticRegression(max_iter=400,
                                  solver='lbfgs',
                                  multi_class='multinomial')
    clf.fit(X, y)
    acc = skm.accuracy_score(y, clf.predict(X)) * 100.0
    print(f"[no-bg hist probe] accuracy = {acc:.1f}% "
          f"(chance ≈ {100/num_domains:.1f}%)")
    return acc


class LambdaController:
    def __init__(self, init_lambda=0.0):
        self.lam = float(init_lambda)
        self.g_pred_sum = 0.0     # accumulate gradient norms
        self.g_dom_sum  = 0.0
        self.n          = 0       # number of batches seen

    def update_batch(self, g_pred, g_dom):
        self.g_pred_sum += g_pred
        self.g_dom_sum  += g_dom
        self.n += 1

    def end_epoch(self):
        if self.n == 0:        # safety
            return
        return
        # --- average gradient norms over the epoch -------------------
        g_pred_avg = self.g_pred_sum / max(self.n, 1)
        g_dom_avg  = self.g_dom_sum  / max(self.n, 1)

        if g_dom_avg > 0:                       # safety check
            ratio = g_pred_avg / g_dom_avg      # desired scale factor
            self.lam *= ratio                   # GradNorm step
            # clip to keep λ in a reasonable range
            
            self.lam = max(0.0, min(self.lam, 30.0))

        # reset accumulators
        self.g_pred_sum = self.g_dom_sum = 0.0
        self.n = 0


@debug_function(context="CLIENT")
def train_dann3d_model(epoch, dataloader, model, optimizer, opt_domain, ema = None):
    _ = model.train()
    # place this right before you create `domain_metric`
    # site_weights = torch.tensor(1.0 / (np.array(site_counts, dtype=np.float32) + 1e-6), dtype=torch.float)
    # site_weights = (site_weights / site_weights.sum()) * num_domains  # re-normalise

    # eps   = 1e-6
    # counts = torch.tensor(site_counts, dtype=torch.float)
    # site_weights = 1.0 / (counts.sqrt() + eps)           # inverse √ frequency
    # site_weights = site_weights.clamp(max=6.0)              # cap rare-site weight max=10.0
    # site_weights = site_weights / site_weights.mean()                   # mean = 1
    
    # if torch.cuda.is_available():
    #     site_weights = site_weights.cuda()
        
    y_preds = []
    y_trues = []
    d_losses = []
    p_losses = []
    losses = []
    maes = []
    ents = []
    ent_accs = []
    loss_metric = nn.MSELoss()
    mae_metric = nn.L1Loss()
    # domain_metric = nn.CrossEntropyLoss(label_smoothing=0.1, weight=site_weights)
    domain_metric = nn.CrossEntropyLoss(label_smoothing=0.05) #label_smoothing=0.1

    p = (epoch) / 75
    lambda_d = 0.0
    # model.grl.coeff = 0.0
    if epoch < 10:             # Phase-A: task focus
        lambda_d = 0.0        # or grl.coeff = 0
        model.grl.coeff = 0.0
    else:
        # lambda_d = 0.3 * (2/(1+math.exp(-10*p)) - 1) + 0.1
        model.grl.coeff = 8.5 * (2/(1+math.exp(-7*p)) - 1)
    # model.grl.coeff = 2.0
    # lambda_d = 1 # 2.0 / (1.0 + math.exp(-10 * progress)) - 1.0
    # model.featurizer.eval()
    # with torch.no_grad():
    #     feats = []
    #     labs  = []
    #     for img, _, meta in dataloader:
    #         feats.append(model.featurizer(img.cuda()).mean([-2,-1,-3]).cpu())  # B×256
    #         labs.append((meta[:, 0].cpu()).long())
    #     X = torch.cat(feats)       # N×256
    #     y = torch.cat(labs)        # N
    # print("balanced accuracy if we memorise sites:",
    #     (y.bincount().max().item()/len(y)))         # just a baseline

    # _ = model.train()
    # print(f"[probe] epoch {epoch}/{epochs}, lambda_d={lambda_d:.4f}, ")
    print(f"[probe] epoch {epoch}/{epochs}, model coeff={model.grl.coeff:.4f}, ")

    for i, (image, label, metadata) in enumerate(dataloader):
        domain_real = metadata[:, 0].long().to(image.device)   # shape (B,)
        # put this literally in the first training batch
        # if i == 0 and epoch == 0:
        #     dom_ids = metadata[:, 0].cpu().numpy()
        #     print("unique site-ids in first batch:", np.unique(dom_ids))
            # print("min / max site-id in whole training set:",
            #     np.min(all_train_sites), np.max(all_train_sites))  # pre-compute once

        # Upper edges of each interval (placed on the same device)
        # boundaries = torch.tensor([3, 6, 9, 12], device=domain_real.device)

        # For x < 3  → 0
        # 3 ≤ x < 6  → 1
        # 6 ≤ x < 9  → 2
        # 9 ≤ x < 12 → 3
        # x ≥ 12     → 4
        domain_label =  domain_real #torch.bucketize(domain_real, boundaries)   # shape (B,), dtype=torch.long
        assert model.domain_classifier[-1].out_features == num_domains  # should be 15
        assert domain_label.max().item() < num_domains                  # no out-of-range ids

        # # ---- build the domain-label tensor ----
        # domain_label = torch.full(                    # shape (B,)
        #     (image.size(0),),                      # batch size
        #     client_id,                        # constant value
        #     dtype=torch.long
        # )
        # log_print(f"Epoch {epoch}, Batch {i}: image shape={image.shape}, label shape={label.shape}", context="CLIENT TRAINING")
        if torch.cuda.is_available():
            
            image = image.cuda()
            label = label.cuda()
            domain_label = domain_label.cuda()
            # weight = weight.cuda()

        prediction, domain_prediction = model(image.float())
        # prediction = prediction.view(-1)  # Flattens [batch_size, 1] → [batch_size]
        # label = label.view(-1)
        # print(prediction['y_pred'])
        # lable_box = Box({'y_trues': label})
        # print("prediction: ", prediction['y_pred'].shape)
        # print("lable: ", lable_box['y_trues'])

        prediction_loss = loss_metric(prediction, label)
        domain_loss = domain_metric(domain_prediction, domain_label.long())
        # prediction_loss = prediction_loss/50.0
        
        with torch.no_grad():
            sm = domain_prediction.softmax(1)
            ent = -(sm * sm.log()).sum(1).mean()
            acc = (sm.argmax(1) == domain_label).float().mean()
            print('softmax entropy:',
            -(sm * sm.log()).sum(1).mean().item(),
            f"acc={acc*100:.1f}")   # in nats
            ents.append(ent.item())
            ent_accs.append(acc.item())

        # log_print(f"Epoch {epoch}, Batch {i}: label is: {label}, prediction is: {prediction}", context="CLIENT TRAINING")
        # log_print(f"Epoch {epoch}, Batch {i}: Loss={loss.item():.4f}", context="CLIENT TRAINING")
        
        # print('prediction: ', prediction['y_pred'], "actual: ", lable_box['y_trues'])
        # loss = prediction_loss + lambda_d*domain_loss
  
        # print(f"Epoch {epoch}, Batch {i}: prediction_loss={prediction_loss.item():.4f}, domain_loss={domain_loss.item():.4f}")
        # loss = (prediction_loss * (1- lambda_d)) + (lambda_d * domain_loss)
        loss = (prediction_loss) + (domain_loss)

        optimizer.zero_grad()
        if(model.grl.coeff > 0.0):
            opt_domain.zero_grad()
        loss.backward()

        # Calculate total L2 norm of all featurizer parameters' gradients
        featurizer_grads = []
        for p in model.featurizer.parameters():
            if p.grad is not None:
                featurizer_grads.append(p.grad.detach().view(-1))

        if featurizer_grads:
            flat = torch.cat(featurizer_grads)
            grad_norm = flat.norm().item()
            print(f"[GRAD CHECK] Featurizer grad norm: {grad_norm:.4f} | GRL coeff = {model.grl.coeff}")
        else:
            print("[GRAD CHECK] No featurizer gradients found.")
        
        # g_feat = torch.norm(torch.stack(
        #             [p.grad.norm() for p in model.featurizer.parameters()]))
        # if(model.grl.coeff > 0.0):
        #     g_dom  = torch.norm(torch.stack(
        #                 [p.grad.norm() for p in model.domain_classifier.parameters()]))
        # else:
        #     g_dom = 0.0
        # print(f"acc={acc*100:.1f}% "
        #     f"  |∇F|={g_feat:.3f}  |∇D|={g_dom:.3f}")
            
        
        torch.nn.utils.clip_grad_norm_(model.domain_classifier.parameters(), 10.0)
        
        optimizer.step()
        if(model.grl.coeff > 0.0):
            opt_domain.step()
            
        ema.update_parameters(model)       # <-- once **per batch**
        y_preds.extend(prediction.detach().cpu().view(-1).tolist())
        y_trues.extend(label.detach().cpu().view(-1).tolist())

        # print('prediction :', y_preds)
        # print("accuracy is : {0:.16f}".format( metrics.accuracy_score(y_trues, y_preds)))
        mae = mae_metric(prediction, 
                        label)
        print(f"[probe] epoch {epoch+1}/{epochs}  ",
              f"domain loss = {domain_loss.item():.4f}  ",
              f"label loss = {prediction_loss.item():.4f}  ",
              f"total loss = {loss.item():.4f}  ",
              f"MAE = {mae.item():.4f}  ")
        
        loss_value = loss.item()
        d_losses.append(domain_loss.item())
        p_losses.append(prediction_loss.item())
        losses.append(loss_value)
        maes.append(mae.item())

    # log_print(f"Epoch {epoch}: Loss={losses}, ACC={accs}", context="CLIENT TRAINING")
    # lambda_ctrl.end_epoch()
    return np.mean(losses), np.mean(maes), np.mean(d_losses), np.mean(p_losses), np.mean(ents), np.mean(ent_accs)

@debug_function(context="CLIENT EVALUATION")
@torch.no_grad()
def evaluate_dann3d_model(dataloader, model, scheduler=None):
    model.eval()

    y_preds = []
    y_trues = []
    # d_losses = []
    # p_losses = []
    losses = []
    maes = []
    site_sums = defaultdict(float)
    site_counts = defaultdict(int)
    loss_metric = nn.MSELoss()
    mae_metric = nn.L1Loss()
    # domain_metric = nn.CrossEntropyLoss()
    lambda_d = 0.0
    
    for i, (image, label, metadata) in enumerate(dataloader):
        # domain_label = torch.full(                    # shape (B,)
        #     (image.size(0),),                      # batch size
        #     6,                        # constant value
        #     dtype=torch.long
        # )  
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            # domain_label = domain_label.cuda()

        prediction, _ = model(image.float())
        # label = torch.squeeze(label, dim=[1])

        prediction_loss = loss_metric(prediction, label).view(-1)
        
        
        err = (prediction - label).pow(2).view(-1).cpu()      # <-- NOT nn.MSELoss
        sites = metadata[:, 0].long()                         # shape [B]
        # domain_loss = domain_metric(domain_prediction, domain_label.long())
        for e, s in  zip(err, sites):
            site_sums[s.item()] += e.item()
            site_counts[s.item()] += 1
        
        y_pred = prediction.detach().cpu().numpy().flatten()
        y_true = label.detach().cpu().numpy().flatten()

        loss = prediction_loss*(1-lambda_d)

        y_preds.extend(y_pred)
        y_trues.extend(y_true)

        mae = mae_metric(prediction, 
                        label)
        
        # d_losses.append(domain_loss.item())
        # p_losses.append(prediction_loss.item())
        losses.append(loss.item())
        maes.append(mae.item())

    # avg_d_loss = np.mean(d_losses)
    # avg_p_loss = np.mean(p_losses)
    avg_loss = np.mean(losses)
    avg_mae = np.mean(maes)
    if scheduler is not None:
        scheduler.step(avg_loss)
    site_mse = {s: site_sums[s]/site_counts[s] for s in site_sums}
    macro_mse = sum(site_mse.values())/len(site_mse)   # un-weighted mean
    # print(f"The MSE per site is: {site_mse}")
    return avg_loss, avg_mae, site_mse, macro_mse



def run_single_experiment(global_run_id, repeat_idx):
    """
    Run a single experiment with the given global run ID and repeat index.
    This function is a placeholder for any specific logic you want to implement.
    """
    run_id = global_run_id# time.strftime("%Y-%m-%d_%H-%M-%S")  # Timestamp-based run ID
    run_dir = os.path.join("./runs", f"run_{run_id}")  # Each run has a separate directory
    os.makedirs(run_dir, exist_ok=True)
    server_log_dir = os.path.join(run_dir, "server_log")
    os.makedirs(server_log_dir, exist_ok=True)  # Create directory if it doesn't exist
    train_log_path = os.path.join(server_log_dir, "train_metrics.txt")
    validation_log_path = os.path.join(server_log_dir, "validation_metrics.txt")
    
    ds = OpenBHBDataset()
    train_dataset = ds.get_subset('train')
    val_dataset = ds.get_subset('val')
    featurizer = BrainCancerFeaturizer()
    regressor = BrainCancerRegressor()
    
    # Load pre-trained weights if available
    # saved_state = torch.load(path_to_load_ERM, map_location='cpu')
    # Get their current state dicts
    # feat_state = featurizer.state_dict()
    # reg_state = regressor.state_dict()
    # Filter matching keys
    # feat_state.update({k: v for k, v in saved_state.items() if k in feat_state})
    # reg_state.update({k: v for k, v in saved_state.items() if k in reg_state})
    # # Load updated weights
    # missing, unexpected = featurizer.load_state_dict(feat_state)
    
    # print("featurizer Missing keys:", missing)
    # print("featurizer Unexpected keys:", unexpected)
    # missing , unexpected = regressor.load_state_dict(reg_state)
    # print("regressor Missing keys:", missing)
    # print("regressor Unexpected keys:", unexpected)
    # Done loading pre-trained weights
    
    model = DANN3D(featurizer=featurizer, regressor=regressor, n_domains=num_domains, hidden_size=512)
    
    if torch.cuda.is_available():
        model = model.cuda()
    # optimizer = optim.Adam(model.parameters(), lr=25e-5) # best: 3e-4
    optimizer = optim.Adam(list(model.featurizer.parameters()) + list(model.regressor.parameters()), lr=8.5e-4) #best 1e-3
    opt_domain = optim.Adam(model.domain_classifier.parameters(), lr= 2e-3) #2e-5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_loader = get_train_loader("standard", train_dataset, batch_size=train_batch_size, num_workers=8)
    validation_loader = get_eval_loader("standard", val_dataset, batch_size=1, num_workers=8)
    logs_train = []
    logs_val = []
    # ctrl = LambdaController(init_lambda=0.3)   # start λ small
    # ---------------------------------------------------------------
    # 0️⃣  Quick probe: “Can the domain head learn at all?”
    # ---------------------------------------------------------------
    import torch.nn.functional as F
    from torch.utils.data import WeightedRandomSampler

    # def probe_domain_head(model, train_ds, val_ds,
    #                     batch_size=16, epochs=20, num_domains=15):

        
    #     device = next(model.parameters()).device
    #     # print("🔎 probe: number of items =", len(train_ds))
    #     # print("🔎 probe: unique site IDs in train set =",
    #     #     np.unique([meta[0] for _, _, meta in train_ds]))

    #     # ---------- balanced sampler (oversample minority sites) ----
    #     site_ids    = np.array([m[0] for _, _, m in train_ds], dtype=np.int64)
    #     site_counts = np.bincount(site_ids, minlength=num_domains).astype(np.float32)
    #     sample_wts  = torch.DoubleTensor(1. / (site_counts[site_ids] + 1e-6))
    #     sampler     = WeightedRandomSampler(sample_wts,
    #                                         num_samples=len(train_ds),
    #                                         replacement=True)
    #     grouper = CombinatorialGrouper(
    #                 dataset=train_ds.dataset,
    #                 groupby_fields=['site'])   # group index == site ID
    #     train_loader = get_train_loader("standard", train_ds,
    #                                     batch_size=batch_size,
    #                                     num_workers=1,
    #                                     uniform_over_groups = True,    # ★ balance!
    #                                     grouper = grouper) # ★ tells WILDS what “group” means
    #     val_loader = get_train_loader("standard",
    #                                 train_ds,
    #                                 batch_size=batch_size,
    #                                 num_workers=1,
    #                                 uniform_over_groups=True,
    #                                 grouper=grouper,
    #                                 drop_last=False)
    #     # acc = quick_hist_probe_nobg(train_loader, nbins=32, num_domains=15)
    #     # exit()
    #     # ❷── verify first *three* batches really balanced ────────────────
    #     # for k, (_, _, meta) in enumerate(train_loader):
    #     #     if k == 3: break
    #     #     print("❷ batch", k, "site histogram:",
    #     #         torch.bincount(meta[:,0].long(), minlength=num_domains)[:15])
    #     # -----------------------------------------------------------------


    #     # ---------- freeze featurizer & regressor -------------------
    #     # model.featurizer.eval()
    #     model.regressor.eval()
    #     for p in model.featurizer.parameters():
    #         p.requires_grad_(True)

    #     for p in model.regressor.parameters():            # not used here
    #         p.requires_grad_(False)

    #     model.grl.coeff = 3.0     # disable gradient reversal for the probe
    #     model.domain_classifier.train()
    #     model.featurizer.train()
    #     # model.scanner.train()              # enable dropout / norms
    #     # for p in model.scanner.parameters():
    #     #     p.requires_grad_(True)         # should already be True, but explicit is safe
            
    #     for p in model.domain_classifier.parameters():
    #         p.requires_grad_(True)         # should already be True, but explicit is safe
    #     # ❸── (re-)initialise head so weights not ~0 ————————————————
    #     # for m in model.domain_classifier.modules():
    #     #     if isinstance(m, nn.Linear):
    #     #         # nn.init.xavier_uniform_(m.weight, 1.0)
    #     #         nn.init.kaiming_uniform_(m.weight, a=0.1)
    #     #         nn.init.zeros_(m.bias)
    #     # ❹── show head weight std so it isn’t tiny ————————————————
    #     first_lin = next(m for m in model.domain_classifier.modules()
    #                     if isinstance(m, nn.Linear))
    #     print("❹ head weight std:", first_lin.weight.std().item())


    #     # opt = torch.optim.Adam(
    #     #                         list(model.featurizer.parameters()) +  # ← add these
    #     #                         # list(model.scanner.parameters()) +         # ← add these
    #     #                         list(model.domain_classifier.parameters()),
    #     #                         lr=5e-4)
    #     opt = torch.optim.Adam([
    #                             {"params": model.featurizer.parameters(), "lr": 1e-5},
    #                             {"params": model.domain_classifier.parameters(), "lr": 15e-6},
    #     # {"params": model.scanner.parameters(), "lr": 1e-4},  # ← add these
    #                             ])
    #     # one-off after creating opt
    #     print('#opt params:', sum(p.numel() for p in opt.param_groups[0]['params']))
    #     # print('#scanner params:', sum(p.numel() for p in model.scanner.parameters()))
    #     # one batch, no scanner
    #     # x,_,meta = next(iter(train_loader))
    #     # vox   = x.to(device).float().flatten(1)           # B × (121·145·121)
    #     # dom   = meta[:,0].long().to(device)

    #     # lin   = nn.Linear(vox.size(1), 15).to(device)
    #     # opti   = torch.optim.Adam(lin.parameters(), lr=1e-2)

    #     # for _ in range(200):              # ~1 s on CPU-only
    #     #     logit = lin(vox)
    #     #     loss  = F.cross_entropy(logit, dom)
    #     #     opti.zero_grad(); loss.backward(); opti.step()

    #     # print('raw-voxel CE after 200 steps:', loss.item())
        
    #     for ep in range(epochs):
    #         for imgs, label, meta in train_loader:
    #             x   = imgs.to(device).float()
    #             age_label = label.to(device).float()  # (B,1,121,145,121)
    #             dom = meta[:, 0].long().to(device)        # site IDs

    #             feat = model.featurizer(x)    # try 'conv3'

    #             with torch.no_grad():
    #                 age = model.regressor(feat)  # (B,1,121,145,121)
                
                
    #             # print(f'shape of feature before flattening: {feat.shape}')
    #             feat = feat.flatten(1)         # B×(256·7·9·7) = B×112 896
    #             feat = model.grl(feat)                # (B,256) – gradient-reversed
    #             # print(f" feature shape = {feat.shape}")
    #             logits = model.domain_classifier(feat)
    #             with torch.no_grad():
    #                 sm = logits.softmax(1)
    #                 print('softmax entropy:',
    #                     -(sm * sm.log()).sum(1).mean().item())   # in nats
    #             loss   = F.cross_entropy(logits, dom, label_smoothing=0.1)
    #             loss_label = F.mse_loss(age, age_label)  # MSE loss on the prediction
    #             accuracy = F.l1_loss(age, age_label)  # L1 loss on the prediction

    #             print(f"[probe] epoch {ep+1}/{epochs}  ",
    #                 f"domain loss = {loss.item():.4f}  ",
    #                 f"label loss = {loss_label.item():.4f}  ",
    #                 f"accuracy = {accuracy.item():.4f}  ")
    #             opt.zero_grad()
    #             loss.backward()
    #             # for name, param in model.featurizer.named_parameters():
    #             #     if param.grad is not None:
    #             #         print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
    #             # print('scanner grad mean:',
    #             #     model.scanner.conv1.weight.grad.abs().mean().item())   # or any layer
    #             # print(f"[probe] epoch {ep+1}/{epochs}  "
    #             #     f"loss = {loss.item():.4f}  "
    #             #     f"dom acc = {100*(logits.argmax(1) == dom).float().mean():.1f}%")
    #             torch.nn.utils.clip_grad_norm_(model.domain_classifier.parameters(), 10.0)
    #             # old = model.scanner.conv1.weight.clone()
    #             opt.step()
    #             # print('scanner Δ:',
    #             #     (model.scanner.conv1.weight - old).abs().mean().item())

    #         # ------------ quick val accuracy each epoch ------------
    #         correct = total = 0
    #         with torch.no_grad():
    #             for imgs, label, meta in val_loader:
    #                 age_label = label.to(device).float()  # (B,1,121,145,121)
    #                 x   = imgs.to(device).float()
    #                 dom = meta[:, 0].long().to(device)
    #                 # feat = model.featurizer(x, tap='conv3')
                    
    #                 feat = model.featurizer(x)    # try 'conv3'
    #                 age = model.regressor(feat)  # (B,1,121,145,121)
    #                 feat = feat.flatten(1)         # B×(256·7·9·7) = B×112 896
    #                 feat = model.grl(feat)                # (B,256) – gradient-reversed
    #                 # feat = feat.flatten(1)
                    
    #                 pred = model.domain_classifier(feat).argmax(1)  
    #                 loss_label = F.mse_loss(age, age_label)  # MSE loss on the prediction
    #                 accuracy = F.l1_loss(age, age_label)  # L1 loss on the prediction

    #                 correct += (pred == dom).sum().item()
    #                 # print(f'pred is {pred}\n dom is {dom}\n')
    #                 total   += dom.size(0)

    #         print(f"[probe] epoch {ep+1}/{epochs}  "
    #             f"val acc = {100*correct/total:.1f}%")

    #         # ❻── print grad magnitude every epoch ——————————
    #         gnorm = torch.norm(torch.stack([
    #                     p.grad.data.norm() for p in model.domain_classifier.parameters()]))
    #         print("❻ grad ‖∇‖ =", gnorm.item())

    #     # ❼── final check no gradient missing ——————————————
    #     any_none = any(p.grad is None for p in model.domain_classifier.parameters())
    #     print("❼ any grad None in head?", any_none)

    #     return correct / total


    # # -------------- call the probe once -----------------------------
    # probe_acc = probe_domain_head(model, train_dataset, val_dataset)

    # if probe_acc > 0.60:      # >60 % accuracy on 15 classes
    #     print("✅  Domain head CAN learn – proceed to adversarial training.")
    # else:
    #     print("❌  Head is near chance.  Feed earlier features (tap='conv3') "
    #         "or remove InstanceNorm3d, then re-run the probe.")

    for p in model.featurizer.parameters():
        p.requires_grad_(True)
    for p in model.regressor.parameters():            # not used here
        p.requires_grad_(True)
    for p in model.domain_classifier.parameters():            # not used here
        p.requires_grad_(True)
    # model.grl.coeff = 3.0     # disable gradient reversal for the probe
    model.domain_classifier.train()
    model.featurizer.train()
    model.regressor.train()
    ema = AveragedModel(model, avg_fn=lambda w_ema,w, n: 0.995*w_ema + 0.005*w)

    for i in range(epochs):
        loss, mae, d_loss, p_loss, ent, ent_acc = train_dann3d_model(i, train_loader, model, optimizer, opt_domain, ema)
        logs_train.append((int(i), float(loss), float(mae), float(d_loss), float(p_loss), float(ent), float(ent_acc)))
        if i%5 == 0:
            loss, mae, site_mse, macro_mse  = evaluate_dann3d_model(validation_loader, ema.module, scheduler)
            logs_val.append((int(i/5), float(loss), float(mae), site_mse, float(macro_mse)))
        # print(f"Epoch is {i}, Lambda is {ctrl.lam:.4f}")

    # Append mode, add separator/header
    with open(train_log_path, "a") as f:
        f.write(f"\n\n====== Run {repeat_idx + 1} ======\n")
        f.write(f"The Dataset Size is : {len(train_dataset)}\n")
        for epoch, loss, mae, d_loss, p_loss, ent, ent_acc in logs_train:
            f.write(f"Epoch {epoch + 1}: Loss={loss:.4f}, MAE={mae:.4f}, Domain Loss={d_loss:.4f}, Prediction Loss={p_loss:.4f}, Entropy={ent:.4f}, Accuracy={ent_acc*100:.1f}\n")

    with open(validation_log_path, "a") as f:
        f.write(f"\n\n====== Run {repeat_idx + 1} ======\n")
        f.write(f"The Dataset Size is : {len(val_dataset)}\n")
        for epoch, loss, mae, site_mse, macro_mse in logs_val:
            f.write(f"Epoch {epoch * 5 + 1}: Loss={loss:.4f}, mae={mae:.4f}, macro_mse={macro_mse:.4f}\n")
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
    all_final_val_maes = []

    for repeat_idx in range(10):
        print(f"\n--- Running experiment {repeat_idx + 1}/10 ---\n")
        logs_val = run_single_experiment(global_run_id, repeat_idx)

        # Only take the last validation result
        if logs_val:
            _, final_loss, final_mae, site_mse, macro_mse = logs_val[-1]
            all_final_val_maes.append((final_loss, final_mae, site_mse, macro_mse))

    # Compute mean and std
    losses = [r[0] for r in all_final_val_maes]
    maes = [r[1] for r in all_final_val_maes]
    per_site_mse = [r[2] for r in all_final_val_maes]
    macro_mse = [r[3] for r in all_final_val_maes]
    
    mean_loss, std_loss = np.mean(losses), np.std(losses)
    mean_mae, std_mae = np.mean(maes), np.std(maes)
    mean_per_site_mse = {k: np.mean([r[k] for r in per_site_mse]) for k in per_site_mse[0]}
    mean_macro_mse = np.mean(macro_mse)

    # Save average results
    run_dir = os.path.join("./runs", f"run_{global_run_id}")  # Each run has a separate directory
    summary_path = os.path.join(run_dir, f"summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}\n")
        f.write(f"Validation MAE: {mean_mae:.4f} ± {std_mae:.4f}\n")
        f.write(f"Mean Per-Site MSE:\n")
        for site, mse in mean_per_site_mse.items():
            f.write(f"  Site {site}: {mse:.4f}\n")
        f.write(f"Mean Macro MSE: {mean_macro_mse:.4f}\n")

    print("\n✅ All runs completed.")
    print(f"📄 Summary saved to {summary_path}")
    
    
