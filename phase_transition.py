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
    
    #sparsity_prob は非ゼロになる確率
    mask = (torch.rand(n_samples, n_features) < sparsity_prob).float().to(DEVICE)
    coeffs = torch.randn(n_samples, n_features).to(DEVICE) * mask
    data = torch.matmul(coeffs, true_features.T)
    return data, true_features

def calculate_recovery_score(sae_decoder, true_features):
    with torch.no_grad():
        W_est = F.normalize(sae_decoder.weight, p=2, dim=0).T
        W_true = true_features.T
        similarity_matrix = torch.matmul(W_est, W_true.T).abs().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
        return similarity_matrix[row_ind, col_ind].mean()

def run_single_experiment(n_samples, sparsity_prob):
    #固定パラメータ
    n_features = 20
    input_dim = 10
    hidden_dim = 80 
    epochs = 300
    
    data, true_features = generate_synthetic_data(n_samples, n_features, input_dim, sparsity_prob)
    loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True)
    
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
            
    return calculate_recovery_score(model.decoder, true_features)

if __name__ == "__main__":
    print("Starting Phase Transition Analysis...")
    
    #値が小さいほどスパース
    #スパースすぎても、デンスすぎても難しいはず
    sparsity_list = [0.05, 0.1, 0.2, 0.4, 0.6] 
    
    #サンプル数が増えれば成功するはず
    sample_list = [500, 1000, 3000, 5000, 10000]
    
    results = np.zeros((len(sparsity_list), len(sample_list)))
    
    print(f"Grid size: {len(sparsity_list)} x {len(sample_list)} = {len(sparsity_list)*len(sample_list)} runs")

    for i, sp in enumerate(sparsity_list):
        for j, n in enumerate(sample_list):
            print(f"Testing: Sparsity={sp}, Samples={n} ... ", end="")
            score = run_single_experiment(n, sp)
            results[i, j] = score
            print(f"Score={score:.4f}")

    
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, annot=True, fmt=".2f", 
                xticklabels=sample_list, yticklabels=sparsity_list, 
                cmap="viridis", vmin=0, vmax=1)
    
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Sparsity Probability (p)")
    plt.title("SAE Feature Recovery Phase Transition")
    
    save_path = "sae_phase_transition.png"
    plt.savefig(save_path)
    print(f"\nPhase Transition Map saved to: {save_path}")