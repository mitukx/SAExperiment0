import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

#再現性のため
torch.manual_seed(42)
np.random.seed(42)


class ToySAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        #バイアスなし
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.apply_decoder_constraint()

    def apply_decoder_constraint(self):
        #デコーダ重みの列ノルムを1に固定
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x):
        #ReLU
        z = F.relu(self.encoder(x))
        x_reconstruct = self.decoder(z)
        return x_reconstruct, z

#
def generate_synthetic_data_fixed(n_samples, n_features, input_dim, sparsity_prob):
    #真の辞書
    true_features = torch.randn(input_dim, n_features).to(DEVICE)
    true_features = F.normalize(true_features, p=2, dim=0)
    
    #スパース係数の生成
    mask = (torch.rand(n_samples, n_features) < sparsity_prob).float().to(DEVICE)
    
    #係数を非負にする (ReLUエンコーダとの整合性確保)
    #これにより、モデルが表現不可能な負の相関を排除
    coeffs = torch.randn(n_samples, n_features).abs().to(DEVICE) * mask
    
    data = torch.matmul(coeffs, true_features.T)
    return data, true_features


def calculate_recovery_rate(sae_decoder, true_features, threshold=0.9):
    with torch.no_grad():
        W_est = F.normalize(sae_decoder.weight, p=2, dim=0).T
        W_true = true_features.T
        
        #類似度行列
        similarity_matrix = torch.matmul(W_est, W_true.T).abs().cpu().numpy()
        
        #線形割当問題で最適なマッチングを見つける
        row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
        matched_sims = similarity_matrix[row_ind, col_ind]
        

        success_count = (matched_sims > threshold).sum()
        return success_count / len(W_true)


def run_single_experiment(n_samples, sparsity_prob):
    n_features = 20
    input_dim = 10
    hidden_dim = 80 
    
    batch_size = min(256, n_samples)
    min_total_updates = 6000  #最低更新回数
    steps_per_epoch = max(1, n_samples // batch_size)
    epochs = int(np.ceil(min_total_updates / steps_per_epoch))
    
    data, true_features = generate_synthetic_data_fixed(n_samples, n_features, input_dim, sparsity_prob)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
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
            
            #制約は更新ごとに適用
            model.apply_decoder_constraint()
            
    return calculate_recovery_rate(model.decoder, true_features)

if __name__ == "__main__":
    print("Starting Refined Phase Transition Analysis...")
    
    #検証するパラメータグリッド
    sparsity_list = [0.05, 0.1, 0.2, 0.4, 0.6] 
    sample_list = [500, 1000, 3000, 5000, 10000]
    
    results = np.zeros((len(sparsity_list), len(sample_list)))
    
    for i, sp in enumerate(sparsity_list):
        for j, n in enumerate(sample_list):
            score = run_single_experiment(n, sp)
            results[i, j] = score
            print(f"Sparsity={sp}, Samples={n} -> RecoveryRate={score:.2f}")

    #描画
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, annot=True, fmt=".2f", 
                xticklabels=sample_list, yticklabels=sparsity_list, 
                cmap="viridis", vmin=0, vmax=1)
    
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Sparsity Probability (p)")
    plt.title("Refined SAE Phase Transition (Fixed Sign & Optimization)")
    
    save_path = "sae_phase_transition_refined.png"
    plt.savefig(save_path)
    print(f"\nSaved refined heatmap to: {save_path}")