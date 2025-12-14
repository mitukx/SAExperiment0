import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment


DEVICE = torch.device("cpu")
print(f"Running on: {DEVICE}")

#再現性のため固定
torch.manual_seed(42)
np.random.seed(42)

class ToySAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.apply_decoder_constraint()

    def apply_decoder_constraint(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_reconstruct = self.decoder(z)
        return x_reconstruct, z

def generate_synthetic_data(n_samples, n_features, input_dim, sparsity_prob):
    true_features = torch.randn(input_dim, n_features).to(DEVICE)
    true_features = F.normalize(true_features, p=2, dim=0)
    
    mask = (torch.rand(n_samples, n_features) < sparsity_prob).float().to(DEVICE)
    coeffs = torch.randn(n_samples, n_features).to(DEVICE) * mask
    data = torch.matmul(coeffs, true_features.T)
    return data, true_features

#
def calculate_recovery_rate(sae_decoder, true_features, threshold=0.9):
    with torch.no_grad():
        W_est = F.normalize(sae_decoder.weight, p=2, dim=0).T
        W_true = true_features.T
        
        #類似度行列
        similarity_matrix = torch.matmul(W_est, W_true.T).abs().cpu().numpy()
        
        #最適マッチング
        row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
        matched_sims = similarity_matrix[row_ind, col_ind]
        
        #閾値判定
        success_count = (matched_sims > threshold).sum()
        total_features = len(W_true)
        
        return success_count / total_features

def run_single_experiment(n_samples, sparsity_prob):
    n_features = 20
    input_dim = 10
    hidden_dim = 80 
    epochs = 300 
    
    data, true_features = generate_synthetic_data(n_samples, n_features, input_dim, sparsity_prob)
    loader = torch.utils.data.DataLoader(data, batch_size=min(256, n_samples), shuffle=True)
    
    model = ToySAE(input_dim, hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            x_recon, z = model(batch)
            recon_loss = F.mse_loss(x_recon, batch)
            l1_loss = z.abs().sum() / batch.size(0)
            loss = recon_loss + 0.1 * l1_loss
            loss.backward()
            optimizer.step()
            model.apply_decoder_constraint()
            
    #ここで新しい評価関数を呼ぶ
    return calculate_recovery_rate(model.decoder, true_features, threshold=0.9)

if __name__ == "__main__":
    print("Starting Phase Transition Analysis (Metric: Recovery Rate)...")
    
    sparsity_list = [0.05, 0.1, 0.2, 0.4, 0.6] 
    sample_list = [500, 1000, 3000, 5000, 10000]
    
    results = np.zeros((len(sparsity_list), len(sample_list)))
    
    for i, sp in enumerate(sparsity_list):
        for j, n in enumerate(sample_list):
            print(f"Testing: Sparsity={sp}, Samples={n} ... ", end="")
            score = run_single_experiment(n, sp)
            results[i, j] = score
            #ログ表示が変わっているか確認！
            print(f"RecoveryRate={score:.2f}") 

    plt.figure(figsize=(10, 8))
    #vmin=0, vmax=1 で固定し、色の対比を最大化
    sns.heatmap(results, annot=True, fmt=".2f", 
                xticklabels=sample_list, yticklabels=sparsity_list, 
                cmap="viridis", vmin=0, vmax=1)
    
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Sparsity Probability (p)")
    plt.title("SAE Feature Recovery Phase Transition (Success Rate)")
    
    save_path = "sae_phase_transition_final.png"
    plt.savefig(save_path)
    print(f"\nSaved final heatmap to: {save_path}")