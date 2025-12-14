import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

#SAE
class ToySAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.apply_decoder_constraint()

    def apply_decoder_constraint(self):
        #Decoder重みの列ノルムを1に固定
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_reconstruct = self.decoder(z)
        return x_reconstruct, z

#データ生成
def generate_synthetic_data(n_samples, n_features, input_dim, sparsity_prob):
    #真の辞書(正規直交に近いランダム)
    true_features = torch.randn(input_dim, n_features).to(DEVICE)
    true_features = F.normalize(true_features, p=2, dim=0)
    
    #スパース係数
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

def run_comparison(reg_type="l1"):
    #固定パラメータ
    n_samples = 5000
    n_features = 20
    input_dim = 10
    hidden_dim = 80 #4倍
    sparsity = 0.1
    epochs = 400
    
    data, true_features = generate_synthetic_data(n_samples, n_features, input_dim, sparsity)
    loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True)
    
    model = ToySAE(input_dim, hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    #deltaが小さいほどL1に近いが、原点付近で二次関数になり滑らか
    huber_fn = nn.HuberLoss(delta=0.05, reduction='sum') 

    print(f"\n--- Training with {reg_type.upper()} Regularization ---")
    
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            x_recon, z = model(batch)
            
            recon_loss = F.mse_loss(x_recon, batch)
            
            #正則化項の切り替え
            if reg_type == "l1":
                reg_loss = z.abs().sum() / batch.size(0)
            elif reg_type == "huber":
                #Huberは通常回帰に使うが、ここではスパース誘導のためzと0の距離に適用
                reg_loss = huber_fn(z, torch.zeros_like(z)) / batch.size(0)
            
            loss = recon_loss + 0.1 * reg_loss
            loss.backward()
            optimizer.step()
            model.apply_decoder_constraint()
            
    score = calculate_recovery_score(model.decoder, true_features)
    print(f"Result ({reg_type}): Feature Recovery Score = {score:.4f}")
    return score

if __name__ == "__main__":
    print("Starting Comparison Experiment...")
    
    #1.L1正則化で実行
    score_l1 = run_comparison("l1")
    
    #2.Huber正則化で実行
    score_huber = run_comparison("huber")
    
    print("\n" + "="*40)
    print("FINAL COMPARISON")
    print("="*40)
    print(f"L1 (Non-smooth) Score : {score_l1:.4f}")
    print(f"Huber (Smooth)  Score : {score_huber:.4f}")
    
    if abs(score_l1 - score_huber) < 0.05:
        print(">> 結論: 滑らかなHuber関数で代替可能 (理論解析に有利)")
    else:
        print(">> 結論: 挙動に有意な差がある (理論修正が必要)")