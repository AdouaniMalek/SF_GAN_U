# =============================================================================
# SF-GAN: Differentially Private Fair Synthetic Data Generation
# Privacy Mechanism : zCDP only (Gaussian mechanism)
# =============================================================================

# =============================================================================
# SECTION 1: IMPORTS & ENVIRONMENT SETUP
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

import sklearn.metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,   # <<< NEW: for AUPRC
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from zcdp_accountant import compute_zcdp, get_privacy_spent

warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 2: GLOBAL CONFIGURATION & HYPERPARAMETERS
# =============================================================================

# --- Paths ---
DATASET_DIR = "C:/Users/Malek Adouani/Desktop/SF_GAN/Dataset_1/"
MODEL_DIR   = os.path.join(DATASET_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Hardware ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Using device  :", device)

# --- Optimizer hyperparameters ---
HYP_LR               = 0.0001
HYP_BATCH_SIZE       = 64
HYP_B1               = 0.9
HYP_B2               = 0.999
HYP_NOISE_MULTIPLIER = 0.0001
HYP_WEIGHT_DECAY     = 0.001

# --- Architecture ---
LATENT_DIM    = 14
MAX_GRAD_NORM = 1.1

# --- Training epochs ---
CVAE_EPOCHS  = 200
WGAN_EPOCHS  = 500

# --- Loss regularisation weights ---
LAMBDA_L1   = 0.0001
LAMBDA_L1_F = 0.0001
LAMBDA_FAIR = 0.5

# --- Fairness: Age discretisation ---
AGE_BINS        = [0, 0.25, 0.40, 1.20]
AGE_NUM_CLASSES = len(AGE_BINS) - 1

# --- Evaluation ---
RHO_VALUES       = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
RHO_SWEEP_VALUES = [5, 10, 15, 20, 30, 50]
N_EVAL_RUNS      = 30
NUM_FAKE         = 668
SAMPLE_INTERVAL  = 1

# --- Reproducibility ---
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# =============================================================================
# SECTION 3: DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_preprocess_data(dataset_dir: str, filename: str):
    df = pd.read_csv(os.path.join(dataset_dir, filename))
    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)

    col_names  = df.columns.tolist()
    col_to_idx = {col: i for i, col in enumerate(col_names)}

    binary_cols  = []
    numeric_cols = []

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        unique_vals = set(series.unique())
        normalized  = set(
            int(v) if isinstance(v, (int, float)) and v in (0, 1, 0.0, 1.0) else v
            for v in unique_vals
        )
        if normalized.issubset({0, 1}) and len(normalized) <= 2:
            binary_cols.append(col)
        else:
            numeric_cols.append(col)

    binary_idx     = [col_to_idx[c] for c in binary_cols  if c in col_to_idx]
    continuous_idx = [col_to_idx[c] for c in numeric_cols if c in col_to_idx]

    data   = torch.from_numpy(df.to_numpy()).float().to(device)
    labels = torch.from_numpy(df["Biopsy"].to_numpy()).float().to(device)

    print(f"Dataset shape  : {data.shape}")
    print(f"Binary cols    : {len(binary_cols)}")
    print(f"Continuous cols: {len(numeric_cols)}")

    return df, col_names, binary_idx, continuous_idx, data, labels


train_df, COLUMN_NAMES, BINARY_INDICES, CONTINUOUS_INDICES, trainData, trainlabels = \
    load_and_preprocess_data(DATASET_DIR, "reconstructed_cancer.csv")


# =============================================================================
# SECTION 4: DATASET CLASSES
# =============================================================================

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data        = data
        self.labels      = labels
        self.sampleSize  = data.shape[0]
        self.featureSize = data.shape[1]

    def __len__(self) -> int:
        return self.sampleSize

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]


class TabularDatasetWGAN(torch.utils.data.Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        protected_attributes: torch.Tensor,
    ) -> None:
        self.data                 = data
        self.labels               = labels
        self.protected_attributes = protected_attributes
        self.sampleSize           = data.shape[0]
        self.featureSize          = data.shape[1]

    def __len__(self) -> int:
        return self.sampleSize

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx], self.protected_attributes[idx]


# =============================================================================
# SECTION 5: UTILITY FUNCTIONS
# =============================================================================

def compute_class_weights(labels: torch.Tensor, imbalance_threshold: float = 0.6) -> dict:
    unique_classes, counts = torch.unique(labels, return_counts=True)
    imbalance_rate = (counts.float() / len(labels)).max().item()
    print(f"Imbalance rate: {imbalance_rate:.3f}")
    if imbalance_rate > imbalance_threshold:
        w = 1.0 / counts.float()
        w = w / w.sum()
        return {cls.item(): wi.item() for cls, wi in zip(unique_classes, w)}
    return {cls.item(): 1.0 for cls in unique_classes}


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def binarise_last_column(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    t = tensor.clone()
    t[:, -1] = (t[:, -1] >= threshold).float()
    return t


def discretise_age(age_tensor: torch.Tensor, bins=None) -> torch.Tensor:
    if bins is None:
        bins = AGE_BINS
    out = torch.zeros(age_tensor.size(0), dtype=torch.long)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        out[(age_tensor >= lo) & (age_tensor < hi)] = i
    return out


def generate_synthetic(
    gen_model: nn.Module,
    n_samples: int,
    feature_dim: int,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    gen_model.eval()
    out = np.zeros((n_samples, feature_dim), dtype=np.float32)
    ptr = 0
    with torch.no_grad():
        while ptr < n_samples:
            bs   = min(batch_size, n_samples - ptr)
            z    = torch.randn(bs, feature_dim, device=device)
            samp = gen_model(z).cpu().numpy()
            out[ptr : ptr + bs] = samp
            ptr += bs
    return out


# =============================================================================
# SECTION 5b: FAIRNESS METRIC — DEMOGRAPHIC PARITY DIFFERENCE (NEW)
# =============================================================================

def compute_demographic_parity_difference(
    synthetic_data_np: np.ndarray,
    label_col_idx: int,
    age_col_idx: int,
    age_bins: list,
    threshold: float = 0.5,
) -> float:
    """
    Demographic Parity Difference (DPD) on synthetic data.

    Definition
    ----------
    For each pair of age groups (g_i, g_j):
        DPD_ij = |P(Y=1 | group=g_i) - P(Y=1 | group=g_j)|

    The reported DPD is the **maximum** pairwise difference across all
    age-group pairs (worst-case fairness violation). Lower is better.

    Parameters
    ----------
    synthetic_data_np : np.ndarray  shape [N, F]
    label_col_idx     : int         column index of the binary outcome (Biopsy)
    age_col_idx       : int         column index of the Age feature
    age_bins          : list        bin edges used to discretise Age
    threshold         : float       binarisation threshold for soft labels

    Returns
    -------
    float : max pairwise DPD  (NaN if any group is empty)
    """
    age_vals   = synthetic_data_np[:, age_col_idx]
    labels     = (synthetic_data_np[:, label_col_idx] >= threshold).astype(int)

    # Discretise age into group indices
    group_ids  = np.zeros(len(age_vals), dtype=int)
    for g, (lo, hi) in enumerate(zip(age_bins[:-1], age_bins[1:])):
        mask = (age_vals >= lo) & (age_vals < hi)
        group_ids[mask] = g

    unique_groups = np.unique(group_ids)
    pos_rates     = {}
    for g in unique_groups:
        g_labels = labels[group_ids == g]
        if len(g_labels) == 0:
            pos_rates[g] = float("nan")
        else:
            pos_rates[g] = g_labels.mean()

    # Max pairwise absolute difference
    rates_list = list(pos_rates.values())
    if any(np.isnan(r) for r in rates_list):
        return float("nan")

    max_dpd = 0.0
    for i in range(len(rates_list)):
        for j in range(i + 1, len(rates_list)):
            max_dpd = max(max_dpd, abs(rates_list[i] - rates_list[j]))
    return max_dpd


# =============================================================================
# SECTION 6: MODEL ARCHITECTURES
# =============================================================================

class CondVAE(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int, class_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + class_dim, 2048), nn.BatchNorm1d(2048), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),                    nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),                     nn.BatchNorm1d(512),  nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + class_dim, 512),   nn.BatchNorm1d(512),  nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),                     nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),                    nn.BatchNorm1d(2048), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, feature_dim),             nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, c):
        if c.dim() == 1:
            c = c.unsqueeze(1)
        h          = self.encoder(torch.cat([x, c], dim=1))
        mu, logvar = h.chunk(2, dim=1)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decoder(torch.cat([z, c], dim=1))
        return x_hat, mu, logvar


class Generator(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),  nn.BatchNorm1d(256),  nn.ReLU(True),
            nn.Linear(256, 512),          nn.BatchNorm1d(512),  nn.ReLU(True),
            nn.Linear(512, 1024),         nn.BatchNorm1d(1024), nn.ReLU(True),
            nn.Linear(1024, 2048),        nn.BatchNorm1d(2048), nn.ReLU(True),
            nn.Linear(2048, 1024),        nn.BatchNorm1d(1024), nn.ReLU(True),
            nn.Linear(1024, 512),         nn.BatchNorm1d(512),  nn.ReLU(True),
            nn.Linear(512, 256),          nn.BatchNorm1d(256),  nn.ReLU(True),
            nn.Linear(256, feature_dim),  nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),        nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),        nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),         nn.LeakyReLU(0.2),
            nn.Linear(512, 256),          nn.LeakyReLU(0.2),
            nn.Linear(256, 512),          nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),         nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.model(x)


class FairnessCritic(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 768),         nn.LeakyReLU(0.2),
            nn.Linear(768, 512),          nn.LeakyReLU(0.2),
            nn.Linear(512, 256),          nn.LeakyReLU(0.2),
            nn.Linear(256, 128),          nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class GeneratorMinimax(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),  nn.BatchNorm1d(256),  nn.ReLU(True),
            nn.Linear(256, 512),          nn.BatchNorm1d(512),  nn.ReLU(True),
            nn.Linear(512, 1024),         nn.BatchNorm1d(1024), nn.ReLU(True),
            nn.Linear(1024, feature_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class DiscriminatorMinimax(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256),         nn.LeakyReLU(0.2),
            nn.Linear(256, 1),           nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# =============================================================================
# SECTION 7: LOSS FUNCTIONS
# =============================================================================

def cvae_loss(x_hat, x, mu, logvar, class_weights, continuous_idx, binary_idx, device):
    eps      = 1e-12
    cont_t   = torch.tensor(continuous_idx, device=device)
    mse_loss = F.mse_loss(x_hat[:, cont_t], x[:, cont_t], reduction="mean")

    bin_t     = torch.tensor(binary_idx, device=device)
    x_bin     = x[:, bin_t]
    x_hat_bin = x_hat[:, bin_t]

    label_col = x[:, -1].long().squeeze()
    weights   = torch.tensor(
        [class_weights.get(idx.item(), 1.0) for idx in label_col],
        dtype=torch.float, device=device,
    ).unsqueeze(1)

    bce_term = weights * (
        x_bin * torch.log(x_hat_bin + eps)
        + (1.0 - x_bin) * torch.log(1.0 - x_hat_bin + eps)
    )
    bce_loss = torch.mean(-torch.sum(bce_term, dim=1))
    kl_loss  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return bce_loss + mse_loss + kl_loss


def fairness_critic_loss(critic_logits, sensitive_attr_targets):
    return F.cross_entropy(critic_logits, sensitive_attr_targets)


def generator_loss(discriminator_fake, critic_logits_fake, sensitive_attr_targets, lambda_fair=0.5):
    adv_loss  = -discriminator_fake.mean()
    fair_loss = fairness_critic_loss(critic_logits_fake, sensitive_attr_targets)
    return adv_loss - lambda_fair * fair_loss


def calc_gradient_penalty(netD, real_data, fake_data, device, lambda_gp=10):
    alpha        = torch.rand(real_data.size(0), 1, device=device).expand_as(real_data)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data.detach()).requires_grad_(True)
    disc_interp  = netD(interpolates)
    gradients    = autograd.grad(
        outputs=disc_interp, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interp),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients        = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def discriminator_loss_GP(netD, real_data, fake_data, real_preds, fake_preds, device, lambda_gp=10):
    d_loss = fake_preds.mean() - real_preds.mean()
    gp     = calc_gradient_penalty(netD, real_data, fake_data, device, lambda_gp)
    return d_loss + gp


def minimax_generator_loss(fake_preds):
    return -torch.log(fake_preds + 1e-8).mean()


def minimax_discriminator_loss(real_preds, fake_preds):
    return (
        -torch.log(real_preds + 1e-8).mean()
        - torch.log(1.0 - fake_preds + 1e-8).mean()
    ) / 2.0


# =============================================================================
# SECTION 8: DIFFERENTIALLY PRIVATE (zCDP) OPTIMIZER
# =============================================================================

def create_zcdp_optimizer(cls):
    class ZCDPOptimizer(cls):
        def __init__(self, params, lr, betas, max_per_sample_grad_norm, rho, *args, **kwargs):
            super().__init__(params, lr=lr, betas=betas, *args, **kwargs)
            self.max_per_sample_grad_norm = max_per_sample_grad_norm
            self.rho                      = rho
            self._reset_aggregate_grads()

        def _reset_aggregate_grads(self):
            for group in self.param_groups:
                group["aggregate_grads"] = [
                    torch.zeros_like(p.data) if p.requires_grad else None
                    for p in group["params"]
                ]

        def clip_grads_(self):
            all_params = [
                p for group in self.param_groups
                for p in group["params"] if p.requires_grad
            ]
            clip_grad_norm_(all_params, max_norm=self.max_per_sample_grad_norm, norm_type=2)
            for group in self.param_groups:
                for param, accum in zip(group["params"], group["aggregate_grads"]):
                    if param.requires_grad and param.grad is not None:
                        accum.add_(param.grad.data)

        def add_noise_(self):
            sigma = self.max_per_sample_grad_norm * np.sqrt(1.0 / (2.0 * self.rho))
            for group in self.param_groups:
                for param, accum in zip(group["params"], group["aggregate_grads"]):
                    if param.requires_grad:
                        param.grad.data = accum.clone()
                        noise = torch.normal(
                            mean=0.0, std=sigma,
                            size=param.grad.data.size(),
                            device=param.grad.data.device,
                            dtype=param.grad.data.dtype,
                        )
                        param.grad.data.add_(noise)

        def step(self, *args, **kwargs):
            self.clip_grads_()
            self.add_noise_()
            super().step(*args, **kwargs)
            self._reset_aggregate_grads()

    return ZCDPOptimizer


# =============================================================================
# SECTION 9: PRIVACY BUDGET COMPUTATION
# =============================================================================

dataset_train   = TabularDataset(trainData, trainlabels)
dataloader_cvae = DataLoader(
    dataset_train, batch_size=HYP_BATCH_SIZE,
    shuffle=True, num_workers=0, drop_last=True,
)

feature_s     = dataset_train.featureSize
total_samples = len(dataset_train)
num_batches   = len(dataloader_cvae)

print(f"Features       : {feature_s}")
print(f"Samples        : {total_samples}")
print(f"Batches/epoch  : {num_batches}")

q                 = HYP_BATCH_SIZE / total_samples
rho               = compute_zcdp(q, noise_multiplier=HYP_NOISE_MULTIPLIER, steps=100)
epsilon, delta, _ = get_privacy_spent(rho, target_delta=1e-5)

print(f"zCDP budget rho = {rho:.4f}")
print(f"Achieves ({epsilon:.3f}, {delta:.1e})-DP")

class_weights = compute_class_weights(trainlabels)

# Resolve column indices needed for fairness
age_col_idx   = COLUMN_NAMES.index("Age")
label_col_idx = COLUMN_NAMES.index("Biopsy")


# =============================================================================
# SECTION 10: PART 1 — DIFFERENTIALLY PRIVATE CONDITIONAL VAE (CVAE)
# =============================================================================

ZCDPAdam     = create_zcdp_optimizer(Adam)
CondVAEModel = CondVAE(feature_s, LATENT_DIM, 1).to(device)
CondVAEModel.apply(weights_init)

optimizer_CVAE = ZCDPAdam(
    CondVAEModel.parameters(),
    lr=HYP_LR, betas=(HYP_B1, HYP_B2),
    max_per_sample_grad_norm=MAX_GRAD_NORM,
    rho=rho,
)

print(f"\nTraining CVAE for {CVAE_EPOCHS} epochs ...")

for epoch in range(CVAE_EPOCHS):
    epoch_recons = []
    epoch_originals = []
    for i_batch, (data, labels) in enumerate(dataloader_cvae):
        real_samples = data.to(device)
        real_labels  = labels.to(device)
        optimizer_CVAE.zero_grad()
        x_hat, mu, logvar = CondVAEModel(real_samples, real_labels)
        loss = cvae_loss(x_hat, real_samples, mu, logvar,
                         class_weights, CONTINUOUS_INDICES, BINARY_INDICES, device)
        loss.backward()
        optimizer_CVAE.step()
        epoch_recons.append(x_hat.detach().cpu())
        epoch_originals.append(real_samples.detach().cpu())
        if (epoch * num_batches + i_batch + 1) % SAMPLE_INTERVAL == 0:
            print(f"[CVAE] [Epoch {epoch+1:>3}/{CVAE_EPOCHS}] "
                  f"[Batch {i_batch+1:>3}/{num_batches}] "
                  f"[Loss: {loss.item():.4f}]")

cvae_save_path = os.path.join(MODEL_DIR, f"ConditionalVAE_rho_{rho:.3f}.pth")
torch.save(CondVAEModel.state_dict(), cvae_save_path)
print(f"CVAE saved: {cvae_save_path}")

loaded_cvae = CondVAE(feature_s, LATENT_DIM, 1).to(device)
loaded_cvae.load_state_dict(torch.load(cvae_save_path, map_location=device, weights_only=True))
loaded_cvae.eval()

all_recons, all_originals = [], []
with torch.no_grad():
    for data_batch, label_batch in dataloader_cvae:
        data_b  = data_batch.to(device)
        label_b = label_batch.to(device)
        x_hat, _, _ = loaded_cvae(data_b, label_b)
        all_recons.append(x_hat.cpu())
        all_originals.append(data_b.cpu())

all_recons    = torch.cat(all_recons, dim=0)
all_originals = torch.cat(all_originals, dim=0)

recons_df  = pd.DataFrame(all_recons.numpy(), columns=COLUMN_NAMES)
recons_csv = os.path.join(DATASET_DIR, f"recons_samp_rho_{rho:.3f}.csv")
recons_df.to_csv(recons_csv, index=False)
print(f"Reconstructed data saved: {recons_csv}")


# =============================================================================
# SECTION 11: PART 2 — FAIR WGAN-GP (SF-GAN)
# =============================================================================

WGAN_tensor              = torch.Tensor(recons_df.to_numpy())
wgan_labels              = binarise_last_column(WGAN_tensor)[:, -1]
train_protected_attr     = torch.Tensor(recons_df["Age"].to_numpy())
train_protected_discrete = discretise_age(train_protected_attr)

dataset_wgan = TabularDatasetWGAN(
    data=WGAN_tensor, labels=wgan_labels,
    protected_attributes=train_protected_discrete,
)
dataloader_wgan = DataLoader(
    dataset_wgan, batch_size=HYP_BATCH_SIZE,
    shuffle=True, num_workers=0, drop_last=True,
)

wgan_feature_dim = dataset_wgan.featureSize
print(f"WGAN dataset size: {len(dataset_wgan)}, feature size: {wgan_feature_dim}")

generatorModel      = Generator(wgan_feature_dim).to(device)
discriminatorModel  = Discriminator(wgan_feature_dim).to(device)
fairnessCriticModel = FairnessCritic(wgan_feature_dim, AGE_NUM_CLASSES).to(device)
for m in (generatorModel, discriminatorModel, fairnessCriticModel):
    m.apply(weights_init)

optimizer_G  = Adam(generatorModel.parameters(),
                    lr=HYP_LR, betas=(HYP_B1, HYP_B2), weight_decay=HYP_WEIGHT_DECAY)
optimizer_D  = Adam(discriminatorModel.parameters(),
                    lr=HYP_LR, betas=(HYP_B1, HYP_B2), weight_decay=HYP_WEIGHT_DECAY)
optimizer_FC = Adam(fairnessCriticModel.parameters(),
                    lr=HYP_LR, betas=(HYP_B1, HYP_B2), weight_decay=HYP_WEIGHT_DECAY)

print(f"\nTraining SF-GAN (WGAN-GP) for {WGAN_EPOCHS} epochs ...")
Gen_data = []

for epoch in range(WGAN_EPOCHS):
    for i_batch, (data, labels, prot_attrs) in enumerate(dataloader_wgan):
        real_samples  = data.float().to(device)
        s_targets     = prot_attrs.long().to(device)
        current_batch = real_samples.size(0)
        noise         = torch.randn(current_batch, wgan_feature_dim, device=device)

        optimizer_D.zero_grad()
        fake_samples = generatorModel(noise).detach()
        pred_real    = discriminatorModel(real_samples)
        pred_fake_D  = discriminatorModel(fake_samples)
        loss_D = discriminator_loss_GP(
            discriminatorModel, real_samples, fake_samples, pred_real, pred_fake_D, device)
        loss_D = loss_D + LAMBDA_L1 * sum(p.norm(1) for p in discriminatorModel.parameters())
        loss_D.backward()
        optimizer_D.step()

        optimizer_FC.zero_grad()
        fake_samples_fc  = generatorModel(noise).detach()
        critic_logits_fc = fairnessCriticModel(fake_samples_fc)
        loss_F = fairness_critic_loss(critic_logits_fc, s_targets)
        loss_F = loss_F + LAMBDA_L1_F * sum(p.norm(1) for p in fairnessCriticModel.parameters())
        loss_F.backward()
        optimizer_FC.step()

        optimizer_G.zero_grad()
        fake_samples_G  = generatorModel(noise)
        pred_fake_G     = discriminatorModel(fake_samples_G)
        critic_logits_G = fairnessCriticModel(fake_samples_G)
        loss_G = generator_loss(pred_fake_G, critic_logits_G, s_targets, LAMBDA_FAIR)
        loss_G = loss_G + LAMBDA_L1 * sum(p.norm(1) for p in generatorModel.parameters())
        loss_G.backward()
        optimizer_G.step()
        Gen_data.append(fake_samples_G.detach().cpu().numpy())

        if (epoch * len(dataloader_wgan) + i_batch + 1) % SAMPLE_INTERVAL == 0:
            print(f"[WGAN] [Epoch {epoch+1:>3}/{WGAN_EPOCHS}] "
                  f"[Batch {i_batch+1:>3}/{len(dataloader_wgan)}] "
                  f"[G: {loss_G.item():.4f}] [D: {loss_D.item():.4f}] [F: {loss_F.item():.4f}]")

for model, name in [
    (generatorModel,      f"generator_ep{WGAN_EPOCHS}_rho_{rho:.3f}.pth"),
    (discriminatorModel,  f"discriminator_ep{WGAN_EPOCHS}_rho_{rho:.3f}.pth"),
    (fairnessCriticModel, f"fairness_critic_ep{WGAN_EPOCHS}_rho_{rho:.3f}.pth"),
]:
    path = os.path.join(MODEL_DIR, name)
    torch.save(model.state_dict(), path)
    print(f"Saved: {path}")


# =============================================================================
# SECTION 12: PART 3 — GENERATE & SAVE FINAL SYNTHETIC SAMPLES
# =============================================================================

gen_arr = generate_synthetic(generatorModel, NUM_FAKE, wgan_feature_dim, device)
print(f"Generated samples shape: {gen_arr.shape}")

gen_df  = pd.DataFrame(gen_arr, columns=COLUMN_NAMES)
gen_csv = os.path.join(DATASET_DIR, f"generated_samples_rho_{rho:.3f}.csv")
gen_df.to_csv(gen_csv, index=False)
print(f"Generated data saved: {gen_csv}")


# =============================================================================
# SECTION 13: PART 4 — PRIVACY BUDGET SWEEP (zCDP rho vs. Reconstruction Loss)
# =============================================================================

def run_rho_sweep(
    rho_values, model_dir, train_data, train_labels,
    feature_size, latent_dim, continuous_idx, binary_idx,
    batch_size=64, n_epochs=100, b1=0.9, b2=0.999,
):
    results  = []
    ZCDPAdam = create_zcdp_optimizer(Adam)
    for rho_val in rho_values:
        print(f"\n{'='*60}\nPrivacy sweep — rho = {rho_val:.4f}")
        eps_val, delta_val, _ = get_privacy_spent(rho_val, target_delta=1e-5)
        ds     = TabularDataset(train_data, train_labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        cw     = compute_class_weights(train_labels)
        model  = CondVAE(feature_size, latent_dim, 1).to(device)
        model.apply(weights_init)
        opt = ZCDPAdam(model.parameters(), lr=0.0001, betas=(b1, b2),
                       max_per_sample_grad_norm=1.1, rho=rho_val)
        final_loss = None
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for data_b, label_b in loader:
                data_b, label_b = data_b.to(device), label_b.to(device)
                opt.zero_grad()
                x_hat, mu, logvar = model(data_b, label_b)
                loss = cvae_loss(x_hat, data_b, mu, logvar, cw, continuous_idx, binary_idx, device)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / len(loader)
            print(f"  Epoch {epoch+1:>3}/{n_epochs}  loss={final_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(model_dir, f"cvae_sweep_rho_{rho_val:.4f}.pth"))
        results.append({"rho": rho_val, "epsilon": eps_val, "delta": delta_val, "final_loss": final_loss})
    return results


sweep_results = run_rho_sweep(
    rho_values=RHO_VALUES, model_dir=MODEL_DIR,
    train_data=trainData, train_labels=trainlabels,
    feature_size=feature_s, latent_dim=LATENT_DIM,
    continuous_idx=CONTINUOUS_INDICES, binary_idx=BINARY_INDICES,
    batch_size=HYP_BATCH_SIZE, n_epochs=CVAE_EPOCHS,
)
sweep_df = pd.DataFrame(sweep_results)
print("\nPrivacy-Utility Trade-off:")
print(sweep_df.to_string(index=False))
sweep_df.to_csv(os.path.join(DATASET_DIR, "rho_sweep_results.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sweep_df["epsilon"], sweep_df["final_loss"], "o-", color="steelblue")
ax.set_xlabel("epsilon (privacy budget)")
ax.set_ylabel("Final Reconstruction Loss")
ax.set_title("Privacy-Utility Trade-off (zCDP)")
plt.tight_layout()
plt.savefig(os.path.join(DATASET_DIR, "privacy_utility_tradeoff.png"), dpi=150)
plt.show()


# =============================================================================
# SECTION 14: PART 5 — ABLATION STUDY
# =============================================================================

def run_sfgan_pipeline(
    train_data, train_labels, feature_dim, latent_dim,
    continuous_idx, binary_idx, model_dir, rho_val,
    batch_size, cvae_epochs, wgan_epochs, lr, b1, b2,
    lambda_l1, lambda_l1_f, lambda_fair, age_num_classes,
    use_privacy=True, use_cvae=True, use_fairness_critic=True,
    run_tag="full", verbose=True,
):
    os.makedirs(model_dir, exist_ok=True)
    losses = {"cvae": [], "G": [], "D": [], "F": []}

    if use_cvae:
        if verbose:
            print(f"\n[{run_tag}] Stage 1: CVAE "
                  f"(privacy={'ON' if use_privacy else 'OFF'}) ...")
        cw      = compute_class_weights(train_labels)
        ds_cvae = TabularDataset(train_data, train_labels)
        dl_cvae = DataLoader(ds_cvae, batch_size=batch_size,
                             shuffle=True, num_workers=0, drop_last=True)
        cvae_model = CondVAE(feature_dim, latent_dim, 1).to(device)
        cvae_model.apply(weights_init)
        if use_privacy:
            _ZCDPAdam = create_zcdp_optimizer(Adam)
            opt_cvae  = _ZCDPAdam(cvae_model.parameters(), lr=lr, betas=(b1, b2),
                                   max_per_sample_grad_norm=1.1, rho=rho_val)
        else:
            opt_cvae = Adam(cvae_model.parameters(), lr=lr, betas=(b1, b2))

        for epoch in range(cvae_epochs):
            ep_loss = 0.0
            for data_b, label_b in dl_cvae:
                data_b, label_b = data_b.to(device), label_b.to(device)
                opt_cvae.zero_grad()
                x_hat, mu, logvar = cvae_model(data_b, label_b)
                loss = cvae_loss(x_hat, data_b, mu, logvar, cw, continuous_idx, binary_idx, device)
                loss.backward()
                opt_cvae.step()
                ep_loss += loss.item()
            avg = ep_loss / len(dl_cvae)
            losses["cvae"].append(avg)
            if verbose:
                print(f"  [CVAE] Epoch {epoch+1:>3}/{cvae_epochs}  loss={avg:.4f}")

        cvae_model.eval()
        recons_list = []
        with torch.no_grad():
            for data_b, label_b in dl_cvae:
                x_hat, _, _ = cvae_model(data_b.to(device), label_b.to(device))
                recons_list.append(x_hat.cpu())
        recons_tensor = torch.cat(recons_list, dim=0)
    else:
        if verbose:
            print(f"\n[{run_tag}] Stage 1: CVAE BYPASSED — using raw data.")
        recons_tensor  = train_data.cpu()
        losses["cvae"] = []

    wgan_labels_raw    = binarise_last_column(recons_tensor)[:, -1]
    prot_attr_discrete = discretise_age(recons_tensor[:, 0], bins=AGE_BINS)
    ds_wgan = TabularDatasetWGAN(
        data=recons_tensor, labels=wgan_labels_raw,
        protected_attributes=prot_attr_discrete,
    )
    dl_wgan = DataLoader(ds_wgan, batch_size=batch_size,
                         shuffle=True, num_workers=0, drop_last=True)

    if verbose:
        print(f"\n[{run_tag}] Stage 2: WGAN-GP "
              f"(fairness={'ON' if use_fairness_critic else 'OFF'}) ...")

    gen_model  = Generator(feature_dim).to(device)
    disc_model = Discriminator(feature_dim).to(device)
    gen_model.apply(weights_init)
    disc_model.apply(weights_init)
    opt_G = Adam(gen_model.parameters(),  lr=lr, betas=(b1, b2))
    opt_D = Adam(disc_model.parameters(), lr=lr, betas=(b1, b2))

    if use_fairness_critic:
        fc_model = FairnessCritic(feature_dim, age_num_classes).to(device)
        fc_model.apply(weights_init)
        opt_FC   = Adam(fc_model.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(wgan_epochs):
        ep_G = ep_D = ep_F = 0.0
        for data_b, _, prot_b in dl_wgan:
            real  = data_b.float().to(device)
            s_tgt = prot_b.long().to(device)
            bs    = real.size(0)
            z     = torch.randn(bs, feature_dim, device=device)

            opt_D.zero_grad()
            fake_D = gen_model(z).detach()
            loss_D = discriminator_loss_GP(disc_model, real, fake_D,
                                           disc_model(real), disc_model(fake_D), device)
            loss_D = loss_D + lambda_l1 * sum(p.norm(1) for p in disc_model.parameters())
            loss_D.backward()
            opt_D.step()
            ep_D += loss_D.item()

            if use_fairness_critic:
                opt_FC.zero_grad()
                fake_FC   = gen_model(z).detach()
                logits_FC = fc_model(fake_FC)
                loss_F    = fairness_critic_loss(logits_FC, s_tgt)
                loss_F    = loss_F + lambda_l1_f * sum(p.norm(1) for p in fc_model.parameters())
                loss_F.backward()
                opt_FC.step()
                ep_F += loss_F.item()

            opt_G.zero_grad()
            fake_G = gen_model(z)
            pred_G = disc_model(fake_G)
            if use_fairness_critic:
                logits_G = fc_model(fake_G)
                loss_G   = generator_loss(pred_G, logits_G, s_tgt, lambda_fair)
            else:
                loss_G   = -pred_G.mean()
            loss_G = loss_G + lambda_l1 * sum(p.norm(1) for p in gen_model.parameters())
            loss_G.backward()
            opt_G.step()
            ep_G += loss_G.item()

        n = len(dl_wgan)
        losses["G"].append(ep_G / n)
        losses["D"].append(ep_D / n)
        losses["F"].append(ep_F / n if use_fairness_critic else 0.0)
        if verbose:
            print(f"  [WGAN] Epoch {epoch+1:>3}/{wgan_epochs}  "
                  f"G={losses['G'][-1]:.4f}  D={losses['D'][-1]:.4f}  F={losses['F'][-1]:.4f}")

    gen_path = os.path.join(model_dir, f"generator_{run_tag}_rho_{rho_val:.3f}.pth")
    torch.save(gen_model.state_dict(), gen_path)
    if verbose:
        print(f"  Generator saved: {gen_path}")
    return gen_model, losses


PIPELINE_KWARGS = dict(
    train_data=trainData, train_labels=trainlabels,
    feature_dim=feature_s, latent_dim=LATENT_DIM,
    continuous_idx=CONTINUOUS_INDICES, binary_idx=BINARY_INDICES,
    model_dir=MODEL_DIR, rho_val=rho,
    batch_size=HYP_BATCH_SIZE, cvae_epochs=CVAE_EPOCHS, wgan_epochs=WGAN_EPOCHS,
    lr=HYP_LR, b1=HYP_B1, b2=HYP_B2,
    lambda_l1=LAMBDA_L1, lambda_l1_f=LAMBDA_L1_F, lambda_fair=LAMBDA_FAIR,
    age_num_classes=AGE_NUM_CLASSES,
)

print("\n" + "=" * 70 + "\nABLATION 1: No Differential Privacy\n" + "=" * 70)
set_seed(42)
gen_no_privacy, losses_no_privacy = run_sfgan_pipeline(
    **PIPELINE_KWARGS, use_privacy=False, use_cvae=True,
    use_fairness_critic=True, run_tag="abl_no_privacy")
synth_no_privacy = generate_synthetic(gen_no_privacy, len(trainData), feature_s, device)

print("\n" + "=" * 70 + "\nABLATION 2: No CVAE\n" + "=" * 70)
set_seed(42)
gen_no_cvae, losses_no_cvae = run_sfgan_pipeline(
    **PIPELINE_KWARGS, use_privacy=True, use_cvae=False,
    use_fairness_critic=True, run_tag="abl_no_cvae")
synth_no_cvae = generate_synthetic(gen_no_cvae, len(trainData), feature_s, device)

print("\n" + "=" * 70 + "\nABLATION 3: No Fairness Critic\n" + "=" * 70)
set_seed(42)
gen_no_fairness, losses_no_fairness = run_sfgan_pipeline(
    **PIPELINE_KWARGS, use_privacy=True, use_cvae=True,
    use_fairness_critic=False, run_tag="abl_no_fairness")
synth_no_fairness = generate_synthetic(gen_no_fairness, len(trainData), feature_s, device)

# ── Ablation loss curves ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Ablation Study — WGAN-GP Generator Loss per Epoch",
             fontsize=14, fontweight="bold")
ablation_configs = [
    (losses_no_privacy,  "Ablation 1: No Privacy",         "tab:orange"),
    (losses_no_cvae,     "Ablation 2: No CVAE",            "tab:red"),
    (losses_no_fairness, "Ablation 3: No Fairness Critic", "tab:purple"),
]
for ax, (losses_dict, title, color) in zip(axes, ablation_configs):
    epochs_x = range(1, len(losses_dict["G"]) + 1)
    ax.plot(epochs_x, losses_dict["G"], label="Generator",     color=color,       lw=2)
    ax.plot(epochs_x, losses_dict["D"], label="Discriminator", color="steelblue", lw=2, ls="--")
    if any(v != 0.0 for v in losses_dict["F"]):
        ax.plot(epochs_x, losses_dict["F"], label="Fairness Critic", color="seagreen", lw=2, ls=":")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
ablation_plot_path = os.path.join(DATASET_DIR, "ablation_loss_curves.png")
plt.savefig(ablation_plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Ablation loss curves saved: {ablation_plot_path}")


# =============================================================================
# SECTION 15: PART 6 — UTILITY EVALUATION (TSTR AUROC + AUPRC)
# =============================================================================

CLASSIFIER_STYLES = {
    "LogisticRegression": {"color": "steelblue",  "marker": "o", "ls": "-"},
    "XGBoost":            {"color": "darkorange",  "marker": "s", "ls": "--"},
    "RandomForest":       {"color": "seagreen",    "marker": "^", "ls": "-."},
}


def evaluate_utility_tstr(
    synthetic_data_np: np.ndarray,
    real_data_np: np.ndarray,
    label_col_idx: int = -1,
    n_runs: int = 30,
    test_size: float = 0.25,
    random_seed_base: int = 0,
) -> dict:
    """
    Train-on-Synthetic / Test-on-Real (TSTR) evaluation.

    Returns BOTH AUROC and AUPRC for each classifier across n_runs.

    Returns
    -------
    dict : {
        classifier_name: {
            "auroc": [run_1, ..., run_n],
            "auprc": [run_1, ..., run_n],
        }
    }
    """
    feat_cols = [i for i in range(real_data_np.shape[1]) if i != label_col_idx]

    X_synth = synthetic_data_np[:, feat_cols]
    y_synth = (synthetic_data_np[:, label_col_idx] >= 0.5).astype(int)
    X_real  = real_data_np[:, feat_cols]
    y_real  = (real_data_np[:, label_col_idx] >= 0.5).astype(int)

    nan_result = {k: {"auroc": [float("nan")] * n_runs, "auprc": [float("nan")] * n_runs}
                  for k in ["LogisticRegression", "XGBoost", "RandomForest"]}

    if len(np.unique(y_synth)) < 2 or len(np.unique(y_real)) < 2:
        print("  [WARNING] Only one class present — skipping evaluation.")
        return nan_result

    results    = {k: {"auroc": [], "auprc": []} for k in ["LogisticRegression", "XGBoost", "RandomForest"]}
    scaler     = StandardScaler()
    X_synth_sc = scaler.fit_transform(X_synth)

    for run_idx in range(n_runs):
        seed_i = random_seed_base + run_idx
        _, X_test, _, y_test = train_test_split(
            X_real, y_real,
            test_size=test_size,
            random_state=seed_i,
            stratify=y_real if len(np.unique(y_real)) > 1 else None,
        )
        X_test_sc = scaler.transform(X_test)

        # ---- Logistic Regression ----
        lr_clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                                    random_state=seed_i, class_weight="balanced")
        lr_clf.fit(X_synth_sc, y_synth)
        lr_prob = lr_clf.predict_proba(X_test_sc)[:, 1]
        results["LogisticRegression"]["auroc"].append(roc_auc_score(y_test, lr_prob))
        results["LogisticRegression"]["auprc"].append(average_precision_score(y_test, lr_prob))

        # ---- XGBoost ----
        xgb_clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                 use_label_encoder=False, eval_metric="logloss",
                                 random_state=seed_i, verbosity=0)
        xgb_clf.fit(X_synth_sc, y_synth)
        xgb_prob = xgb_clf.predict_proba(X_test_sc)[:, 1]
        results["XGBoost"]["auroc"].append(roc_auc_score(y_test, xgb_prob))
        results["XGBoost"]["auprc"].append(average_precision_score(y_test, xgb_prob))

        # ---- Random Forest ----
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                        random_state=seed_i, class_weight="balanced")
        rf_clf.fit(X_synth_sc, y_synth)
        rf_prob = rf_clf.predict_proba(X_test_sc)[:, 1]
        results["RandomForest"]["auroc"].append(roc_auc_score(y_test, rf_prob))
        results["RandomForest"]["auprc"].append(average_precision_score(y_test, rf_prob))

    return results


# ── Utility sweep across rho values ───────────────────────────────────────────

real_data_np  = trainData.cpu().numpy()
N_GEN_SAMPLES = len(trainData)
sweep_records = []

print("\n" + "=" * 70)
print("UTILITY EVALUATION: AUROC + AUPRC vs. Privacy Budget (rho)")
print(f"  Classifiers : Logistic Regression | XGBoost | Random Forest")
print(f"  Runs/rho    : {N_EVAL_RUNS}")
print(f"  rho values  : {RHO_SWEEP_VALUES}")
print("=" * 70)

for rho_val in RHO_SWEEP_VALUES:
    print(f"\n--- rho = {rho_val:.3f} ---")
    eps_val, delta_val, _ = get_privacy_spent(rho_val, target_delta=1e-5)
    print(f"    => ({eps_val:.3f}, {delta_val:.1e})-DP")
    set_seed(42)

    gen_sweep_i, _ = run_sfgan_pipeline(
        **{**PIPELINE_KWARGS, "rho_val": rho_val},
        use_privacy=True, use_cvae=True, use_fairness_critic=True,
        run_tag=f"sweep_rho_{rho_val:.3f}", verbose=False,
    )

    synth_np    = generate_synthetic(gen_sweep_i, N_GEN_SAMPLES, feature_s, device)
    util_dict   = evaluate_utility_tstr(
        synthetic_data_np=synth_np,
        real_data_np=real_data_np,
        label_col_idx=label_col_idx,
        n_runs=N_EVAL_RUNS,
    )

    for clf_name, metrics in util_dict.items():
        mean_auroc = np.nanmean(metrics["auroc"])
        std_auroc  = np.nanstd(metrics["auroc"])
        mean_auprc = np.nanmean(metrics["auprc"])
        std_auprc  = np.nanstd(metrics["auprc"])
        print(f"    {clf_name:<22} "
              f"AUROC = {mean_auroc:.4f} +/- {std_auroc:.4f}  |  "
              f"AUPRC = {mean_auprc:.4f} +/- {std_auprc:.4f}")
        sweep_records.append({
            "rho":        rho_val,
            "epsilon":    eps_val,
            "delta":      delta_val,
            "classifier": clf_name,
            "mean_auroc": mean_auroc,
            "std_auroc":  std_auroc,
            "mean_auprc": mean_auprc,
            "std_auprc":  std_auprc,
            "all_auroc":  metrics["auroc"],
            "all_auprc":  metrics["auprc"],
        })

    del gen_sweep_i
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Save summary ───────────────────────────────────────────────────────────────

sweep_summary = pd.DataFrame([
    {k: v for k, v in rec.items() if k not in ("all_auroc", "all_auprc")}
    for rec in sweep_records
])
print("\nFull AUROC + AUPRC Summary:")
print(sweep_summary.to_string(index=False))
sweep_summary.to_csv(os.path.join(DATASET_DIR, "utility_auroc_auprc_rho_sweep.csv"), index=False)

# ── Plot: AUROC and AUPRC ribbon panels ───────────────────────────────────────

for metric_key, metric_label in [("mean_auroc", "AUROC"), ("mean_auprc", "AUPRC")]:
    std_key  = metric_key.replace("mean_", "std_")
    all_key  = "all_" + metric_key.replace("mean_", "")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Utility (TSTR {metric_label}) vs. Privacy Budget\n"
        f"Mean +/- 1 Std over {N_EVAL_RUNS} runs",
        fontsize=13, fontweight="bold",
    )

    ax1 = axes[0]
    for clf_name, style in CLASSIFIER_STYLES.items():
        sub = sweep_summary[sweep_summary["classifier"] == clf_name].sort_values("epsilon")
        mu  = sub[metric_key].values
        sig = sub[std_key].values
        eps = sub["epsilon"].values
        ax1.plot(eps, mu, color=style["color"], marker=style["marker"],
                 ls=style["ls"], lw=2, label=clf_name)
        ax1.fill_between(eps, mu - sig, mu + sig, color=style["color"], alpha=0.15)
    ax1.set_xlabel("Privacy Budget (epsilon)", fontsize=11)
    ax1.set_ylabel(f"Mean {metric_label} (TSTR)", fontsize=11)
    ax1.set_title(f"Mean {metric_label} vs. Epsilon", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.0)

    ax2            = axes[1]
    all_rho_sorted = sorted(RHO_SWEEP_VALUES)
    n_clf          = len(CLASSIFIER_STYLES)
    bar_width      = 0.25
    x_positions    = np.arange(len(all_rho_sorted))
    eps_labels     = []

    for ci, (clf_name, style) in enumerate(CLASSIFIER_STYLES.items()):
        means = [np.nanmean(next(r for r in sweep_records
                                 if abs(r["rho"] - rv) < 1e-9
                                 and r["classifier"] == clf_name)[all_key])
                 for rv in all_rho_sorted]
        stds  = [np.nanstd(next(r for r in sweep_records
                                if abs(r["rho"] - rv) < 1e-9
                                and r["classifier"] == clf_name)[all_key])
                 for rv in all_rho_sorted]
        offset = (ci - n_clf / 2.0 + 0.5) * bar_width
        ax2.bar(x_positions + offset, means, bar_width,
                yerr=stds, label=clf_name, color=style["color"],
                alpha=0.8, capsize=4, error_kw={"elinewidth": 1.5})

    for rho_v in all_rho_sorted:
        e, _, _ = get_privacy_spent(rho_v, target_delta=1e-5)
        eps_labels.append(f"rho={rho_v}\n(eps={e:.1f})")

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(eps_labels, fontsize=8)
    ax2.set_xlabel("Privacy Budget", fontsize=11)
    ax2.set_ylabel(f"Mean {metric_label}", fontsize=11)
    ax2.set_title(f"{metric_label} by Budget and Classifier", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0.4, 1.0)

    plt.tight_layout()
    fname = f"utility_{metric_label.lower()}_sweep.png"
    plt.savefig(os.path.join(DATASET_DIR, fname), dpi=150, bbox_inches="tight")
    plt.show()

# ── Plot: violin distributions for AUROC and AUPRC ────────────────────────────

for metric_label, all_key in [("AUROC", "all_auroc"), ("AUPRC", "all_auprc")]:
    n_clf_count = len(CLASSIFIER_STYLES)
    fig, axes   = plt.subplots(1, n_clf_count, figsize=(6 * n_clf_count, 5), sharey=True)
    fig.suptitle(f"Per-Run {metric_label} Distribution over {N_EVAL_RUNS} Runs",
                 fontsize=13, fontweight="bold")

    clf_names_list = list(CLASSIFIER_STYLES.keys())
    for ax, (clf_name, style) in zip(axes, CLASSIFIER_STYLES.items()):
        score_matrix  = []
        x_tick_labels = []
        for rho_v in all_rho_sorted:
            rec = next(r for r in sweep_records
                       if abs(r["rho"] - rho_v) < 1e-9 and r["classifier"] == clf_name)
            score_matrix.append(rec[all_key])
            e, _, _ = get_privacy_spent(rho_v, target_delta=1e-5)
            x_tick_labels.append(f"rho={rho_v}\neps={e:.1f}")

        parts = ax.violinplot(score_matrix, positions=range(len(all_rho_sorted)),
                              showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(style["color"])
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(all_rho_sorted)))
        ax.set_xticklabels(x_tick_labels, fontsize=8)
        ax.set_title(clf_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Privacy Budget")
        ax.set_ylabel(metric_label if clf_name == clf_names_list[0] else "")
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random baseline")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fname = f"utility_{metric_label.lower()}_violin.png"
    plt.savefig(os.path.join(DATASET_DIR, fname), dpi=150, bbox_inches="tight")
    plt.show()

# ── Ablation utility comparison (AUROC + AUPRC) ────────────────────────────────

print("\n" + "=" * 70)
print("ABLATION UTILITY COMPARISON (TSTR AUROC + AUPRC, 30 runs)")
print("=" * 70)

ablation_variants = [
    ("Full SF-GAN",         generate_synthetic(gen_no_privacy, N_GEN_SAMPLES, feature_s, device)),
    ("No Privacy (Abl-1)",  synth_no_privacy),
    ("No CVAE (Abl-2)",     synth_no_cvae),
    ("No Fairness (Abl-3)", synth_no_fairness),
]

abl_records = []
for variant_name, synth_np in ablation_variants:
    util_dict = evaluate_utility_tstr(
        synthetic_data_np=synth_np, real_data_np=real_data_np,
        label_col_idx=label_col_idx, n_runs=N_EVAL_RUNS,
    )
    print(f"\n  {variant_name}")
    for clf_name, metrics in util_dict.items():
        mu_roc  = np.nanmean(metrics["auroc"])
        sig_roc = np.nanstd(metrics["auroc"])
        mu_prc  = np.nanmean(metrics["auprc"])
        sig_prc = np.nanstd(metrics["auprc"])
        print(f"    {clf_name:<22} "
              f"AUROC = {mu_roc:.4f} +/- {sig_roc:.4f}  |  "
              f"AUPRC = {mu_prc:.4f} +/- {sig_prc:.4f}")
        abl_records.append({
            "variant": variant_name, "classifier": clf_name,
            "mean_auroc": mu_roc, "std_auroc": sig_roc,
            "mean_auprc": mu_prc, "std_auprc": sig_prc,
        })

abl_df = pd.DataFrame(abl_records)
abl_df.to_csv(os.path.join(DATASET_DIR, "ablation_auroc_auprc_comparison.csv"), index=False)

# ── Ablation bar chart (side-by-side AUROC and AUPRC) ─────────────────────────
for metric_label, mean_col, std_col in [
    ("AUROC", "mean_auroc", "std_auroc"),
    ("AUPRC", "mean_auprc", "std_auprc"),
]:
    fig, ax       = plt.subplots(figsize=(14, 5))
    variants_list = abl_df["variant"].unique().tolist()
    n_var         = len(variants_list)
    n_clf_abl     = len(CLASSIFIER_STYLES)
    bw            = 0.22
    xs            = np.arange(n_var)

    for ci, (clf_name, style) in enumerate(CLASSIFIER_STYLES.items()):
        sub    = abl_df[abl_df["classifier"] == clf_name]
        means  = [sub[sub["variant"] == v][mean_col].values[0] for v in variants_list]
        stds   = [sub[sub["variant"] == v][std_col].values[0]  for v in variants_list]
        offset = (ci - n_clf_abl / 2.0 + 0.5) * bw
        ax.bar(xs + offset, means, bw, yerr=stds, label=clf_name,
               color=style["color"], alpha=0.85, capsize=4, error_kw={"elinewidth": 1.5})

    ax.set_xticks(xs)
    ax.set_xticklabels(variants_list, fontsize=10)
    ax.set_ylabel(f"Mean {metric_label} (TSTR)", fontsize=11)
    ax.set_title(
        f"Ablation Study: {metric_label} per Component Removal\n"
        f"(Mean +/- Std over {N_EVAL_RUNS} runs)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random baseline")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    abl_bar_path = os.path.join(DATASET_DIR, f"ablation_{metric_label.lower()}_bar.png")
    plt.savefig(abl_bar_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"{metric_label} ablation bar chart saved: {abl_bar_path}")


# =============================================================================
# SECTION 16: PART 7 — FAIRNESS EVALUATION (DEMOGRAPHIC PARITY DIFFERENCE)
#             Privacy Budget vs. DPD Sweep
# =============================================================================

print("\n" + "=" * 70)
print("FAIRNESS EVALUATION: Demographic Parity Difference (DPD) vs. rho")
print(f"  Sensitive attr : Age (discretised into {AGE_NUM_CLASSES} groups)")
print(f"  Metric         : Max pairwise |P(Y=1|group_i) - P(Y=1|group_j)|")
print(f"  rho values     : {RHO_SWEEP_VALUES}")
print(f"  Lower DPD = More Fair")
print("=" * 70)

dpd_records = []

for rho_val in RHO_SWEEP_VALUES:
    print(f"\n--- rho = {rho_val:.3f} ---")
    eps_val, delta_val, _ = get_privacy_spent(rho_val, target_delta=1e-5)
    set_seed(42)

    gen_dpd_i, _ = run_sfgan_pipeline(
        **{**PIPELINE_KWARGS, "rho_val": rho_val},
        use_privacy=True, use_cvae=True, use_fairness_critic=True,
        run_tag=f"dpd_sweep_rho_{rho_val:.3f}", verbose=False,
    )

    # Generate N_EVAL_RUNS independent synthetic batches to get a distribution of DPD
    dpd_run_scores = []
    for run_idx in range(N_EVAL_RUNS):
        set_seed(run_idx)
        synth_np_run = generate_synthetic(gen_dpd_i, N_GEN_SAMPLES, feature_s, device)
        dpd_val      = compute_demographic_parity_difference(
            synthetic_data_np=synth_np_run,
            label_col_idx=label_col_idx,
            age_col_idx=age_col_idx,
            age_bins=AGE_BINS,
        )
        dpd_run_scores.append(dpd_val)

    mean_dpd = np.nanmean(dpd_run_scores)
    std_dpd  = np.nanstd(dpd_run_scores)
    print(f"    DPD = {mean_dpd:.4f} +/- {std_dpd:.4f}  "
          f"(epsilon={eps_val:.3f}, delta={delta_val:.1e})")

    dpd_records.append({
        "rho":       rho_val,
        "epsilon":   eps_val,
        "delta":     delta_val,
        "mean_dpd":  mean_dpd,
        "std_dpd":   std_dpd,
        "all_dpd":   dpd_run_scores,
    })

    del gen_dpd_i
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Save DPD summary ──────────────────────────────────────────────────────────

dpd_summary = pd.DataFrame([
    {k: v for k, v in rec.items() if k != "all_dpd"}
    for rec in dpd_records
])
print("\nFull DPD Summary (lower is fairer):")
print(dpd_summary.to_string(index=False))
dpd_summary.to_csv(os.path.join(DATASET_DIR, "fairness_dpd_rho_sweep.csv"), index=False)

# ── Plot 1: DPD ribbon vs. epsilon ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(
    f"Fairness (DPD) vs. Privacy Budget (rho)\n"
    f"Mean +/- 1 Std over {N_EVAL_RUNS} runs  |  Lower DPD = More Fair",
    fontsize=13, fontweight="bold",
)

ax1 = axes[0]
mu_arr  = dpd_summary["mean_dpd"].values
sig_arr = dpd_summary["std_dpd"].values
eps_arr = dpd_summary["epsilon"].values
ax1.plot(eps_arr, mu_arr, "o-", color="crimson", lw=2, label="Mean DPD")
ax1.fill_between(eps_arr, mu_arr - sig_arr, mu_arr + sig_arr, color="crimson", alpha=0.2)
ax1.set_xlabel("Privacy Budget (epsilon)", fontsize=11)
ax1.set_ylabel("Demographic Parity Difference (DPD)", fontsize=11)
ax1.set_title("DPD vs. Epsilon\n(lower = fairer)", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0.0)

# ── Plot 2: DPD bar chart ─────────────────────────────────────────────────────

ax2         = axes[1]
x_positions = np.arange(len(dpd_records))
means_bar   = [r["mean_dpd"] for r in dpd_records]
stds_bar    = [r["std_dpd"]  for r in dpd_records]
eps_tick    = [f"rho={r['rho']}\n(eps={r['epsilon']:.1f})" for r in dpd_records]

ax2.bar(x_positions, means_bar, width=0.5, yerr=stds_bar,
        color="crimson", alpha=0.75, capsize=5, error_kw={"elinewidth": 1.5},
        label="Mean DPD")
ax2.set_xticks(x_positions)
ax2.set_xticklabels(eps_tick, fontsize=9)
ax2.set_xlabel("Privacy Budget", fontsize=11)
ax2.set_ylabel("Demographic Parity Difference (DPD)", fontsize=11)
ax2.set_title("DPD by Privacy Budget\n(lower = fairer)", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")
ax2.legend(fontsize=10)
ax2.set_ylim(bottom=0.0)

plt.tight_layout()
dpd_sweep_path = os.path.join(DATASET_DIR, "fairness_dpd_sweep.png")
plt.savefig(dpd_sweep_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"DPD sweep plot saved: {dpd_sweep_path}")

# ── Plot 3: DPD violin distribution across rho values ─────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(f"Per-Run DPD Distribution over {N_EVAL_RUNS} Runs\n(lower = more fair)",
             fontsize=13, fontweight="bold")

score_matrix  = [rec["all_dpd"] for rec in dpd_records]
x_tick_labels = [f"rho={r['rho']}\neps={r['epsilon']:.1f}" for r in dpd_records]

parts = ax.violinplot(score_matrix, positions=range(len(dpd_records)),
                      showmeans=True, showmedians=True)
for pc in parts["bodies"]:
    pc.set_facecolor("crimson")
    pc.set_alpha(0.55)

ax.set_xticks(range(len(dpd_records)))
ax.set_xticklabels(x_tick_labels, fontsize=9)
ax.set_xlabel("Privacy Budget", fontsize=11)
ax.set_ylabel("Demographic Parity Difference (DPD)", fontsize=11)
ax.set_title("DPD Distribution per Privacy Budget", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(bottom=0.0)

plt.tight_layout()
dpd_violin_path = os.path.join(DATASET_DIR, "fairness_dpd_violin.png")
plt.savefig(dpd_violin_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"DPD violin plot saved: {dpd_violin_path}")

# ── Plot 4: Joint Privacy-Utility-Fairness overview ──────────────────────────

# Use XGBoost as the representative classifier for the joint plot
xgb_rows    = sweep_summary[sweep_summary["classifier"] == "XGBoost"].sort_values("epsilon")
eps_shared  = xgb_rows["epsilon"].values

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))

ln1, = ax1.plot(eps_shared, xgb_rows["mean_auroc"].values,
                "o-", color="steelblue",  lw=2, label="AUROC (XGBoost)")
ln2, = ax1.plot(eps_shared, xgb_rows["mean_auprc"].values,
                "s--", color="darkorange", lw=2, label="AUPRC (XGBoost)")
ln3, = ax2.plot(dpd_summary["epsilon"].values, dpd_summary["mean_dpd"].values,
                "^:", color="crimson", lw=2, label="DPD (fairness, lower=better)")

ax1.set_xlabel("Privacy Budget (epsilon)", fontsize=11)
ax1.set_ylabel("Utility Score (AUROC / AUPRC)", fontsize=11, color="steelblue")
ax2.set_ylabel("DPD (Demographic Parity Difference)", fontsize=11, color="crimson")
ax2.tick_params(axis="y", labelcolor="crimson")
ax1.set_ylim(0.4, 1.0)
ax2.set_ylim(bottom=0.0)

lines  = [ln1, ln2, ln3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=10, loc="lower right")
ax1.grid(True, alpha=0.3)
ax1.set_title(
    "Privacy-Utility-Fairness Trade-off\n"
    "(higher epsilon = weaker privacy, better utility, but potentially less fair)",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
joint_path = os.path.join(DATASET_DIR, "privacy_utility_fairness_joint.png")
plt.savefig(joint_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Joint trade-off plot saved: {joint_path}")

# ── Ablation DPD comparison ────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ABLATION FAIRNESS COMPARISON (DPD, 30 synthetic batches per variant)")
print("=" * 70)

ablation_dpd_variants = [
    ("Full SF-GAN (DP+Fair)",  gen_no_privacy,   True),
    ("No Privacy (Abl-1)",     gen_no_privacy,   False),
    ("No CVAE (Abl-2)",        gen_no_cvae,      False),
    ("No Fairness (Abl-3)",    gen_no_fairness,  False),
]

abl_dpd_records = []
for variant_name, gen_model_v, _ in ablation_dpd_variants:
    dpd_scores_v = []
    for run_idx in range(N_EVAL_RUNS):
        set_seed(run_idx)
        synth_v  = generate_synthetic(gen_model_v, N_GEN_SAMPLES, feature_s, device)
        dpd_v    = compute_demographic_parity_difference(
            synthetic_data_np=synth_v,
            label_col_idx=label_col_idx,
            age_col_idx=age_col_idx,
            age_bins=AGE_BINS,
        )
        dpd_scores_v.append(dpd_v)

    mu_dpd  = np.nanmean(dpd_scores_v)
    std_dpd = np.nanstd(dpd_scores_v)
    print(f"  {variant_name:<26} DPD = {mu_dpd:.4f} +/- {std_dpd:.4f}")
    abl_dpd_records.append({
        "variant":  variant_name,
        "mean_dpd": mu_dpd,
        "std_dpd":  std_dpd,
    })

abl_dpd_df = pd.DataFrame(abl_dpd_records)
abl_dpd_df.to_csv(os.path.join(DATASET_DIR, "ablation_dpd_comparison.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 5))
xs  = np.arange(len(abl_dpd_df))
ax.bar(xs, abl_dpd_df["mean_dpd"], width=0.5,
       yerr=abl_dpd_df["std_dpd"], color="crimson", alpha=0.75,
       capsize=6, error_kw={"elinewidth": 1.5})
ax.set_xticks(xs)
ax.set_xticklabels(abl_dpd_df["variant"].tolist(), fontsize=10)
ax.set_ylabel("Demographic Parity Difference (DPD)", fontsize=11)
ax.set_title(
    f"Ablation Study: Fairness (DPD) per Component Removal\n"
    f"(Mean +/- Std over {N_EVAL_RUNS} runs  |  lower = more fair)",
    fontsize=12, fontweight="bold",
)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(bottom=0.0)
plt.tight_layout()
abl_dpd_path = os.path.join(DATASET_DIR, "ablation_dpd_bar.png")
plt.savefig(abl_dpd_path, dpi=150, bbox_inches="tight")
plt.show()
