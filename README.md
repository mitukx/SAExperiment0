# 平均場理論に基づくSAE学習ダイナミクスの数値検証

## 概要
Sparse Autoencoders (SAE) の学習挙動を、平均場理論（Mean-Field Theory）および勾配流（Gradient Flow）の観点から検証するための数値実験プロジェクトです。
本実験では、**「L1正則化の特異点（Singularity）」**と**「学習の相転移（Phase Transition）」**という2つの重要な物理的性質に焦点を当てています。

## 背景
鈴木大慈先生らの Mean-Field Langevin Dynamics (MFLD) の枠組みを SAE に適用することを想定し、以下の仮説を検証しました。

1. **正則化の非平滑性**: 滑らかな近似（Huber損失等）では、SAEの本質的なスパース性を獲得できないのではないか？
2. **相転移の存在**: サンプル数とスパース度の間には、学習の成功/失敗を分ける明確な境界（Critical Boundary）が存在するのではないか？
3. **平均場極限**: 有限幅のネットワークは、無限幅極限の挙動に速やかに収束するか？

---

## 主な実験結果 (Key Findings)

### 1. 相転移の発見 (Phase Transition)
サンプル数 $N$ とデータのスパース度 $p$ を変化させ、特徴復元の成功率をヒートマップ化しました。
従来の平均類似度による評価ではノイズに埋もれていた境界を明確にするため、本実験では厳格な**復元成功率（Recovery Rate）**を導入しました（後述）。

その結果、**学習が完全に成功する領域（Recovery Rate $\approx 1.0$）と、全く学習できない領域（Recovery Rate $\approx 0.0$）の間に、極めて鋭い相転移（Phase Transition）が存在すること**を確認しました。
これは、SAEの学習において「中途半端な成功」は稀であり、ある閾値を超えた瞬間に構造を捉えるという**非自明なダイナミクス**を示唆しています。

![Phase Transition Heatmap](images/sae_phase_transition_final.png)
*(Fig 1. 復元成功率に基づく相転移図。左下の暗部（失敗）と右上の明部（成功）の間に明確な境界が見られる)*

### 2. L1正則化の特異性の重要 (Singularity is Essential)
理論解析を容易にするために用いられる「滑らかなポテンシャル（Huber Loss）」と、本来の「L1正則化」を比較しました。
実験の結果、Huber損失ではパラメータ $\delta$ を $10^{-3}$ まで小さくしても、L1を用いた場合と比較して特徴復元スコアが有意に低下しました。
これは、学習において **$z=0$ での微分不可能性（特異点）がスパース表現獲得に不可欠**であることを示しており、理論構築には平滑化を行わない Proximal Gradient Flow の枠組みが必要であることを示唆しています。

![Comparison of z-distribution](images/z_distribution_comparison.png)
*(Fig 2. L1正則化とHuber損失による獲得特徴の違い)*

### 3. 平均場極限への収束 (Convergence to Mean-Field Limit)
隠れ層のニューロン数 $m$ を増加させた際、性能がどのように変化するかを確認しました。
入力次元 $d$ に対して $m \approx 4d$ 付近で性能が飽和し、それ以上の幅では挙動が安定しました。この結果は、現実的なサイズの有限幅モデルであっても、無限幅極限（Mean-Field Limit）の理論予測が良い近似になることを支持しています。

![Width Scaling](images/sae_width_scaling.png)
*(Fig 3. ネットワーク幅の拡大に伴う復元率の収束)*

---

## 評価指標について (Metrics)

本実験では、相転移挙動を正確に捉えるため、以下の指標を採用しています。

**Recovery Rate (復元成功率):**
単純なコサイン類似度の平均値（Mean Cosine Similarity）では、ランダムな相関やバイアスの影響でベースラインが高くなり、成功/失敗の境界が不明瞭になる問題がありました。
そのため、本研究では**「真の特徴量とのコサイン類似度が 0.9 を超えた特徴の割合」**を Recovery Rate と定義し、これをOrder Parameter（秩序変数）として採用しています。

$$ \text{Recovery Rate} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}(\text{CosineSim}(\hat{A}_{\pi(k)}, A^*_k) > 0.9) $$

---

## 実行方法 (Usage)

### 依存ライブラリ
```bash
pip install torch numpy matplotlib seaborn scipy