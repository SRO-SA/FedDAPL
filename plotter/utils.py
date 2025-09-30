import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path("/rhome/ssafa013/FL_syft/logs")
export = Path("/rhome/ssafa013/FL_syft/plotter/export")
fedAvg_path = Path(base, "FedAVG")
fedAvg_DANN_path = Path(base, "FedAVG_DANN")
custom_strategy_path = Path(base, "custom_strategy")
centralized_DANN = Path(base, "Centralized_DANN")

def get_server_validation_logs(base_path, run_id):
    return Path(base_path, run_id,  "server_log/validation_metrics.txt")

def get_server_metrics_csv(base_path, run_id):
    return Path(base_path, run_id, "server_log/server_metrics.csv")


# ---------- helper to parse logs ----------
def parse_centralized(file):
    rounds, loss, acc, pred, dom = [], [], [], [], []
    with open(file) as f:
        for line in f:
            m = re.search(r'Epoch\s+([\d.]+):\s+Loss=([\d.]+),\s+acc=([\d.]+)', line)
            if m:
                r = int(float(m.group(1)))
                if r == 0:
                    continue
                rounds.append(r)
                loss.append(float(m.group(2)))
                acc.append(float(m.group(3)))
                m2 = re.search(r'Domain Loss=([\d.]+),\s*Prediction Loss=([\d.]+)', line)
                if m2:
                    dom.append(float(m2.group(1)))
                    pred.append(float(m2.group(2)))
                else:
                    dom.append(None)
                    pred.append(None)
    # return 
    df = pd.DataFrame({'round': rounds, 'loss': loss, 'acc': acc, 'pred_loss': pred, 'domain_loss': dom})
    df['round'] -= 1
    return df

def parse_fedavg(file):
    rounds, loss, acc = [], [], []
    with open(file) as f:
        for line in f:
            m = re.search(r'Round\s+(\d+):\s+Loss=([\d.]+),\s+ACC=([\d.]+)', line)
            if m:
                r = int(m.group(1))
                if r == 0:
                    continue                
                rounds.append(r)
                loss.append(float(m.group(2)))
                acc.append(float(m.group(3)))
    return pd.DataFrame({'round': rounds, 'loss': loss, 'acc': acc})

def parse_custom(file):
    rounds, loss, acc = [], [], []
    with open(file) as f:
        for line in f:
            m = re.search(r'Loss=([\d.]+),\s*AUC=([\d.]+)', line)
            if m:
                loss.append(float(m.group(1)))
                acc.append(float(m.group(2)))
    rounds = list(range(1, len(loss)+1))
    df = pd.DataFrame({'round': rounds, 'loss': loss, 'acc': acc})
    df = df[df['round']>1]
    df['round'] -= 1
    return df

def parse_dann(file):
    rounds, loss, acc, pred, dom = [], [], [], [], []
    with open(file) as f:
        for line in f:
            m = re.search(r'Round\s+(\d+):\s+Loss=([\d.]+),\s+acc=([\d.]+)', line)
            if m:
                r = int(m.group(1))
                if r == 0:
                    continue
                rounds.append(r)
                loss.append(float(m.group(2)))
                acc.append(float(m.group(3)))
                m2 = re.search(r'Domain Loss=([\d.]+),\s*Prediction Loss=([\d.]+)', line)
                if m2:
                    dom.append(float(m2.group(1)))
                    pred.append(float(m2.group(2)))
                else:
                    dom.append(None)
                    pred.append(None)
    # return 
    df = pd.DataFrame({'round': rounds, 'loss': loss, 'acc': acc, 'pred_loss': pred, 'domain_loss': dom})
    df['round'] -= 1
    return df