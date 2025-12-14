import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

DEVICE = torch.device("cpu") 
print(f"Running on: {DEVICE}")

class ToySAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)
        x_reconstruct = self.decoder(z)
        return x_reconstruct, z

def generate_data(n_samples=5000, n_features=20, input_dim=10, sparsity=0.1):
    true_features = torch.randn(input_dim, n_features).to(DEVICE)
    true_features = F.normalize(true_features, p=2, dim=0)
    mask = (torch.rand(n_samples, n_features) < sparsity).float().to(DEVICE)
    coeffs = torch.randn(n_samples, n_features).to(DEVICE) * mask
    data = torch.matmul(coeffs, true_features.T)
    return data, true_features

def get_score(model, true_features):
    with torch.no_grad():
        W_est = model.decoder.weight.T
        W_true = true_features.T
        sim_matrix = torch.matmul(W_est, W_true.T).abs().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)
        return sim_matrix[row_ind, col_ind].mean()

def run_width_scaling():
    print("Starting Width Scaling Experiment (Mean-Field Limit Check)...")
    
    n_features = 20    #真の特徴数
    input_dim = 10     #入力次元
    
    #幅の倍率
    ratios = [1, 2, 4, 8, 16, 32, 64]
    scores = []
    
    data, true_features = generate_data()
    loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True)
    
    for r in ratios:
        hidden_dim = n_features * r
        print(f"Testing Width Ratio = {r}x (Hidden Dim = {hidden_dim}) ... ", end="")
        
        model = ToySAE(input_dim, hidden_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        #学習
        for epoch in range(300):
            model.train()
            for batch in loader:
                optimizer.zero_grad()
                x_recon, z = model(batch)
                loss = F.mse_loss(x_recon, batch) + 0.1 * z.abs().sum() / batch.size(0)
                loss.backward()
                optimizer.step()
        
        s = get_score(model, true_features)
        scores.append(s)
        print(f"Score = {s:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(ratios, scores, marker='o', linestyle='-', linewidth=2, color='blue')
    plt.xscale('log', base=2) 
    plt.xlabel("Width Ratio (m / n_features)")
    plt.ylabel("Feature Recovery Score")
    plt.title("Convergence to Mean-Field Limit")
    plt.grid(True, which="both", ls="--")
    plt.axhline(y=1.0, color='r', linestyle=':', label='Perfect Recovery')
    plt.legend()
    
    plt.savefig("sae_width_scaling.png")
    print("\nPlot saved to sae_width_scaling.png")

if __name__ == "__main__":
    run_width_scaling()