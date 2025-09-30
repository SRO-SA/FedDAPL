from pdb import run

from grpc import server
from utils import get_server_validation_logs, get_server_metrics_csv
from utils import parse_fedavg, parse_custom, parse_dann, parse_centralized
from utils import fedAvg_path, fedAvg_DANN_path, custom_strategy_path, centralized_DANN , export

import matplotlib.pyplot as plt
import pandas as pd
# Custom starategy
# run_id_c_1 = "run_2025-04-27_20-56-11" # no Alpha
# run_id_c_2 = "run_2025-05-01_13-45-58" # Alpha mix 80/20
# run_id_c_3 = "run_2025-05-05_14-17-31" # Alpha mix 90/10
run_id_c_4 = "run_2025-05-09_11-28-41" # Alpha mix 80/20 Beta 0.01
# run_id_c_5 = "run_2025-05-17_00-01-54" # Alpha mix 80/20 Beta 0.01 Contrastive 0.1, 0.02
run_id_c_6 = "run_2025-05-19_10-57-22" # Alpha mix 80/20 Beta 0.01 Contrastive 0.2, 0.05

# FedAvg_DANN
run_id_fd_1 = "run_2025-06-06_20-13-07" # Static lambda
run_id_fd_2 = "run_2025-06-06_23-43-07" # Static lambda Proximal Term
run_id_fd_3 =  "run_2025-06-12_11-30-11"# "run_2025-06-11_02-48-37" # Static lambda proximal Term warmup with scheduler

# FedAvg
run_id_f_1 = "run_2025-05-31_12-39-15"

# Centralized DANN
run_id_cd_1 = "run_2025-06-04_14-19-03" # Static lambda 0.2

validation_paths = {
                    "FedAvg": get_server_validation_logs(fedAvg_path, run_id_f_1),
                    "FedAvg_DANN": get_server_validation_logs(fedAvg_DANN_path, run_id_fd_1),
                    "FedAvg_DANN_Proximal": get_server_validation_logs(fedAvg_DANN_path, run_id_fd_2),
                    "FedAvg_DANN_Proximal_warmup": get_server_validation_logs(fedAvg_DANN_path, run_id_fd_3),
                    # "Custom_003": get_server_validation_logs(custom_strategy_path, run_id_c_1),
                    # "Custom_003_alpha_mix_0.8": get_server_validation_logs(custom_strategy_path, run_id_c_2),
                    # "Custom_003_alpha_mix_0.9": get_server_validation_logs(custom_strategy_path, run_id_c_3),
                    "Custom_01_alpha_mix_0.8": get_server_validation_logs(custom_strategy_path, run_id_c_4),
                    # "Custom_01_alpha_mix_0.8_C_01_N_002": get_server_validation_logs(custom_strategy_path, run_id_c_5),
                    "Custom_01_alpha_mix_0.8_C_02_N_005": get_server_validation_logs(custom_strategy_path, run_id_c_6),
                    "Centralized_DANN": get_server_validation_logs(centralized_DANN, run_id_cd_1)
                    }


server_metrics_paths = {
                        # "Custom_003": get_server_metrics_csv(custom_strategy_path, run_id_c_1),
                        # "Custom_003_alpha_mix_0.8": get_server_metrics_csv(custom_strategy_path, run_id_c_2),
                        # "Custom_003_alpha_mix_0.9": get_server_metrics_csv(custom_strategy_path, run_id_c_3),
                        "Custom_01_alpha_mix_0.8": get_server_metrics_csv(custom_strategy_path, run_id_c_4),
                        # "Custom_01_alpha_mix_0.8_C_01_N_002": get_server_metrics_csv(custom_strategy_path, run_id_c_5),
                        "Custom_01_alpha_mix_0.8_C_02_N_005": get_server_metrics_csv(custom_strategy_path, run_id_c_6)
                        }

df_fed = parse_fedavg(validation_paths["FedAvg"])

# df_custom = parse_custom(validation_paths["Custom_003"])
# df_custom_mix_08_003 = parse_custom(validation_paths["Custom_003_alpha_mix_0.8"])
# df_custom_mix_09 = parse_custom(validation_paths["Custom_003_alpha_mix_0.9"])
df_custom_mix_08_01 = parse_custom(validation_paths["Custom_01_alpha_mix_0.8"])
# df_custom_mix_08_01_C_01_N_002 = parse_custom(validation_paths["Custom_01_alpha_mix_0.8_C_01_N_002"])
df_custom_mix_08_01_C_02_N_005 = parse_custom(validation_paths["Custom_01_alpha_mix_0.8_C_02_N_005"])

df_dann = parse_dann(validation_paths["FedAvg_DANN"])
df_dann_proximal = parse_dann(validation_paths["FedAvg_DANN_Proximal"])
df_dann_proximal_warmup = parse_dann(validation_paths["FedAvg_DANN_Proximal_warmup"])

df_centralized_dann = parse_centralized(validation_paths["Centralized_DANN"])

print("Centralized_DANN:", df_centralized_dann.shape)

plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.plot(df_centralized_dann['round'], df_centralized_dann['pred_loss'], '-o', label='Centralized DANN λ=0.2')
plt.plot(df_fed['round'], df_fed['loss'], '-o', label='FedAvg')
plt.plot(df_dann['round'], df_dann['pred_loss'], '-o', label='FedAvg_DANN λ=0.2 (total)')
plt.plot(df_dann_proximal['round'], df_dann_proximal['pred_loss'], '-o', label='FedAvg_DANN λ=0.2 MU=3e-2 (proximal)')
plt.plot(df_dann_proximal_warmup['round'], df_dann_proximal_warmup['pred_loss'], '-o', label='FedAvg_DANN λ=0.4 warmup=2 MU=25e-3 (proximal)')

# plt.plot(df_custom['round'], df_custom['loss'], '-o', label='Custom β=0.003')
# plt.plot(df_custom_mix_08_003['round'], df_custom_mix_08_003['loss'], '-o', label='Custom α‑mix %80 β=0.003')
# plt.plot(df_custom_mix_09['round'], df_custom_mix_09['loss'], '-o', label='Custom α‑mix %90 β=0.003')
plt.plot(df_custom_mix_08_01['round'], df_custom_mix_08_01['loss'], '-o', label='Custom α‑mix %80 β=0.01')
# plt.plot(df_custom_mix_08_01_C_01_N_002['round'], df_custom_mix_08_01_C_01_N_002['loss'], '-o', label='Custom α‑mix %80 β=0.01 Contrastive Loss 0.1, std noise 0.02')
plt.plot(df_custom_mix_08_01_C_02_N_005['round'], df_custom_mix_08_01_C_02_N_005['loss'], '-o', label='Custom α‑mix %80 β=0.01 Contrastive Loss 0.2, std noise 0.05')

plt.yscale('log')  # Set y-axis to logarithmic
plt.xlabel("Round"); plt.ylabel("Validation Loss"); plt.title("Validation Loss vs Round"); plt.grid(True); plt.legend()


plt.subplot(1,2,2)
plt.plot(df_centralized_dann['round'], df_centralized_dann['acc'], '-o', label='Centralized DANN λ=0.2')
plt.plot(df_fed['round'], df_fed['acc'], '-o', label='FedAvg')
plt.plot(df_dann['round'], df_dann['acc'], '-o', label='FedAvg_DANN λ=0.2')
plt.plot(df_dann_proximal['round'], df_dann_proximal['acc'], '-o', label='FedAvg_DANN λ=0.2 MU=3e-2 (proximal)')
plt.plot(df_dann_proximal_warmup['round'], df_dann_proximal_warmup['acc'], '-o', label='FedAvg_DANN λ=0.4 warmup=2 MU=25e-3 (proximal)')

# plt.plot(df_custom['round'], df_custom['acc'], '-o', label='Custom β=0.003')
# plt.plot(df_custom_mix_08_003['round'], df_custom_mix_08_003['acc'], '-o', label='Custom α‑mix %80 β=0.003')
# plt.plot(df_custom_mix_09['round'], df_custom_mix_09['acc'], '-o', label='Custom α‑mix %90 β=0.003')
plt.plot(df_custom_mix_08_01['round'], df_custom_mix_08_01['acc'], '-o', label='Custom α‑mix %80 β=0.01')
# plt.plot(df_custom_mix_08_01_C_01_N_002['round'], df_custom_mix_08_01_C_01_N_002['acc'], '-o', label='Custom α‑mix %80 β=0.01 Contrastive Loss 0.1, std noise 0.02')
plt.plot(df_custom_mix_08_01_C_02_N_005['round'], df_custom_mix_08_01_C_02_N_005['acc'], '-o', label='Custom α‑mix %80 β=0.01 Contrastive Loss 0.2, std noise 0.05')

plt.yscale('log')  # Set y-axis to logarithmic
plt.xlabel("Round"); plt.ylabel("ACC (lower is better)"); plt.title("Validation ACC vs Round"); plt.grid(True); plt.legend()

plt.tight_layout()
loss_acc_path = export/"comparison_loss_acc_2.png"
plt.savefig(loss_acc_path)
plt.close()


# ---------- PLOT 2 : Scale‑Normalised diversity ----------
# df_no = pd.read_csv(server_metrics_paths["Custom_003"])
# df_mix_08_003 = pd.read_csv(server_metrics_paths["Custom_003_alpha_mix_0.8"])
# df_mix_09 = pd.read_csv(server_metrics_paths["Custom_003_alpha_mix_0.9"])
df_mix_08_01 = pd.read_csv(server_metrics_paths["Custom_01_alpha_mix_0.8"])
# df_mix_08_01_01_002 = pd.read_csv(server_metrics_paths["Custom_01_alpha_mix_0.8_C_01_N_002"])
df_mix_08_01_02_005 = pd.read_csv(server_metrics_paths["Custom_01_alpha_mix_0.8_C_02_N_005"])

# ratio_no  = df_no.groupby('round')['param_diversity'].mean() /  df_no.groupby('round')['spec_norm'].mean()
# ratio_mix_08_003 = df_mix_08_003.groupby('round')['param_diversity'].mean() / df_mix_08_003.groupby('round')['spec_norm'].mean()
# ratio_mix_09 = df_mix_09.groupby('round')['param_diversity'].mean() / df_mix_09.groupby('round')['spec_norm'].mean()
ratio_mix_08_01 = df_mix_08_01.groupby('round')['param_diversity'].mean() / df_mix_08_01.groupby('round')['spec_norm'].mean()
# ratio_mix_08_01_01_002 = df_mix_08_01_01_002.groupby('round')['param_diversity'].mean() / df_mix_08_01_01_002.groupby('round')['spec_norm'].mean()
ratio_mix_08_01_02_005 = df_mix_08_01_02_005.groupby('round')['param_diversity'].mean() / df_mix_08_01_02_005.groupby('round')['spec_norm'].mean()

plt.figure(figsize=(7,4))
# plt.plot(ratio_no.index, ratio_no.values*100, '-o', label='β=0.003 (no mix)')
# plt.plot(ratio_mix_08_003.index, ratio_mix_08_003.values*100, '-o', label='β=0.003 α‑mix 80/20')
# plt.plot(ratio_mix_09.index, ratio_mix_09.values*100, '-o', label='β=0.003 α‑mix 90/10')
plt.plot(ratio_mix_08_01.index, ratio_mix_08_01.values*100, '-o', label='β=0.01 α‑mix 80/20')
# plt.plot(ratio_mix_08_01_01_002.index, ratio_mix_08_01_01_002.values*100, '-o', label='β=0.01 α‑mix 80/20, Contarstive 0.1, std noise 0.02')
plt.plot(ratio_mix_08_01_02_005.index, ratio_mix_08_01_02_005.values*100, '-o', label='β=0.01 α‑mix 80/20, Contarstive 0.2, std noise 0.05')

plt.yscale('log')  # Set y-axis to logarithmic
plt.xlabel("Round"); plt.ylabel("Diversity (% of ‖M_spec‖₂)"); plt.title("Scale‑Normalised Diversity"); plt.grid(True); plt.legend()
div_path = export/"diversity_ratio.png"
plt.tight_layout()
plt.savefig(div_path)
plt.close()

print("Saved plots:")
print(loss_acc_path)
print(div_path)