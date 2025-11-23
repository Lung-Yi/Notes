# VAE 分子生成模型的迭代 Fine-tuning 研究筆記

## 研究主題概述

本筆記整理了使用 Variational Autoencoder (VAE) 模型進行分子結構生成，並透過**持續蒐集感興趣的分子數據並迭代 fine-tuning VAE**，使其生成的分子越來越接近特定目標結構的相關研究。

## 核心概念

這類研究的核心流程通常包含：

1. **生成階段**：使用 VAE 從潛在空間採樣生成大量候選分子
2. **篩選階段**：根據特定標準（如分子性質、親和力、藥物相似性）篩選出符合條件的分子
3. **Fine-tuning 階段**：使用篩選後的分子數據集重新訓練/fine-tune VAE
4. **迭代循環**：重複上述過程，使模型逐漸聚焦於目標分子空間

---

## 主要研究論文

### 1. Active Learning-Guided Seq2Seq VAE for Multi-target Inhibitor Generation

**論文連結**：https://arxiv.org/html/2506.15309

**發表時間**：2025年6月

**核心貢獻**：
- 提出結構化主動學習（Active Learning, AL）框架，將 Seq2Seq VAE 整合到迭代循環中
- 設計了雙層嵌套的主動學習循環：
  - **Chemical AL cycles（內層循環）**：生成分子 → 化學過濾器篩選 → 累積到數據集 → fine-tune VAE
  - **Affinity AL cycles（外層循環）**：對累積分子進行多靶點對接評估 → 篩選高親和力分子 → fine-tune VAE

**實驗設計**：
- 目標：針對三種冠狀病毒主蛋白酶（SARS-CoV-2, SARS-CoV, MERS-CoV）生成泛抑制劑
- Chemical AL 循環在每次迭代中生成 3,500 個分子，使用不良結構基序過濾器和化學信息學預測器閾值進行篩選
- 通過的分子被累積並用於 fine-tune VAE（從通用訓練權重開始）
- 完成 n 次 Chemical AL 循環後，執行 Affinity AL 循環，基於多靶點親和力進行篩選

**計算效率**：
- 在單個 GPU H1000 上 fine-tuning 1,000 個分子的特定數據集需要約 10.52 ± 0.73 分鐘
- 一個完整的 Chemical AL 循環（包含 fine-tuning、生成 3,500 個分子和化學過濾）需要約 1 小時 42 分鐘

**關鍵發現**：
- 通過迭代 fine-tuning，VAE 能夠逐漸學習生成符合特定化學空間的分子
- 累積的特定數據集具有累積性質，隨著每個 Chemical AL 循環而增長，導致時間逐漸增加

---

### 2. Optimizing Drug Design by Merging Generative AI with Physics-Based Active Learning

**論文連結**：https://www.nature.com/articles/s42004-025-01635-7, [PDF](./pdf_repository/vae/Optimizing_drug_design_CC_2025.pdf)

**期刊**：Communications Chemistry

**發表時間**：2025年8月

**核心創新**：
這是一篇非常完整的研究，提出了將 VAE 與雙層嵌套主動學習循環結合的完整工作流程。

#### 工作流程設計

**1. 數據表示**：
- 訓練分子表示為 SMILES，經過 tokenization 和 one-hot encoding 後輸入 VAE

**2. 初始訓練**：
- VAE 首先在通用訓練集上訓練，學習生成可行的化學分子
- 然後在目標特定訓練集（initial-specific training set）上 fine-tune，學習生成具有增強靶標結合能力的分子

**3. 內層 Active Learning 循環（Inner AL cycles）**：
- 生成的分子使用化學信息學預測器評估：
  - QED（Quantitative Estimate of Drug Likeliness）> 0.6
  - SA score（合成可及性）< 7.0
  - Tanimoto 相似度 < 0.6（與訓練集的差異性）
- 符合閾值的分子被添加到 **temporal-specific set**
- 使用此數據集 fine-tune VAE，優先考慮具有期望性質的分子
- 從第二個循環開始，相似度是針對累積的 temporal-specific set 評估

**4. 外層 Active Learning 循環（Outer AL cycles）**：
- 在設定數量的內層循環後開始
- temporal-specific set 中累積的分子進行分子對接模擬（作為親和力預測器）
- 符合對接分數閾值的分子轉移到 **permanent-specific set**
- 使用此數據集 fine-tune VAE
- 繼續迭代進行嵌套的內層 AL 循環、對接評估和 fine-tuning
- 在後續的內層循環中，相似度針對 permanent-specific set 評估

**5. 候選選擇**：
- 經過設定數量的外層循環後，應用嚴格的過濾和選擇過程
- 使用深度分子建模模擬（如 PELE）提供結合相互作用和穩定性的深入評估

#### 實驗案例 1：CDK2（Cyclin-Dependent Kinase 2）

**目標**：尋找新型 CDK2 抑制劑，避開典型的激酶鉸鏈結合 scaffold

**數據集**：
- 初始特定訓練集：1,061 個實驗性 CDK2 抑制劑

**實驗結果**：

**第一個外層循環（16 個內層循環）**：
- 生成 49,796 個分子
- 30.3% 符合內層循環閾值（QED > 0.6, SA < 7.0, 相似度 < 0.6）
- 僅 1.8% 符合外層循環閾值（Glide gscore ≤ -8.0 kcal·mol⁻¹）

**第二個外層循環**：
- 生成 24,766 個分子
- 48.6% 符合內層循環閾值
- 4.8% 符合外層循環閾值

**第三和第四個外層循環**（降低相似度閾值至 0.4）：
- 第三循環從累積的 permanent-specific set 開始
- 第四循環從初始特定集重新開始（提供獨立起點）
- 僅 7.8% 和 7.3% 被內層循環接受
- 但其中 8.1% 和 6.1% 也被外層 Glide gscore 閾值接受

**第五個外層循環**：
- 將所有前四個循環接受的分子加入 permanent-specific set
- 相似度閾值恢復至 0.6
- 生成 36,465 個分子
- 51.1% 符合內層循環閾值
- 6.3% 符合外層循環閾值

**親和力改善**：
- Glide gscore 範圍 -8.0 到 -11.5 kcal·mol⁻¹ 的分子增加到 4,627 個（相比初始集增加 21 倍）
- 其中 3,142 個與初始集的最大相似度 < 0.3
- 28 個分子的 gscore < -11.5 kcal·mol⁻¹（初始集僅 3 個），其中 11 個相似度 < 0.3

**最終驗證**：
- 候選選擇：使用嚴格閾值（gscore < -10 kcal·mol⁻¹，相似度 < 0.3）保留 324 個分子
- 使用 PELE 軟體進行 Monte Carlo 模擬精煉
- 選擇 10 個分子進行合成
- **成功合成 6 個分子（加上 2 個類似物和 1 個手性化合物）**
- **8/9 的合成分子顯示體外活性，IC₅₀ < 50 μM**
- **4 個分子 IC₅₀ 在 5-10 μM 範圍**
- **1 個分子達到 71 nM 的納摩爾級效力**

#### 實驗案例 2：KRAS^G12D^

**挑戰**：數據稀缺環境，僅收集到 73 個已知親和力的 KRAS SII 抑制劑

**解決方案**：
- 創建額外的 1,891 個未知實驗親和力的分子集（通過高通量虛擬篩選獲得）
- 進行兩個並行生成過程：
  1. 使用已知特定集（known generative process）
  2. 使用未知特定集（unknown generative process）

**結果**：
- KRAS unknown 生成的外層循環接受率（15.8 ± 2.9%）顯著高於 KRAS known（2.3 ± 1.3%）和 CDK2（2.8 ± 2.3%）
- Known 分子顯示異常化學結構（如破碎的雙環或七元環），可能因為 VAE 難以解釋訓練集中複雜的帶正電雙環結構
- Unknown 分子顯示顯著的親和力改善：
  - 生成 23,488 個 gscore < -8.0 kcal·mol⁻¹ 的分子（20 倍增長）
  - 125 個分子 gscore < -10.0 kcal·mol⁻¹（初始集僅 1 個）

**驗證**：
- 應用嚴格過濾器（相似度 < 0.25，gscore < -9 kcal·mol⁻¹）保留 279 個分子
- 使用 PELE 精煉，選擇 19 個候選分子
- 基於 CDK2 中觀察到的 ABFE 與實驗數據的強相關性，使用 ABFE（絕對結合自由能）模擬進行體外驗證
- **4 個分子預測 Kd < 15 μM**

#### 技術細節

**VAE 架構**：
- Encoder：LSTM 層 → 全連接層（256 units）→ 潛在空間（128 維）
- Decoder：從潛在向量擴展 → LSTM 層 → 全連接層（256 units）→ Softmax 激活層
- 激活函數：ReLU

**Temporal-specific set**：
- 在內層 AL 循環迭代中構建
- 只保留通過化學信息學過濾器的分子
- 與 initial-specific set 一起累積形成用於 fine-tune VAE 權重的數據集
- 由於未通過過濾器的化合物不被保留，該集合僅捕獲最有前景但仍具探索性的 scaffold

**Permanent-specific set**：
- 在外層 AL 循環迭代中構建
- 只保留通過對接閾值的分子
- 跨所有循環累積，提供兩個功能：
  1. 提供穩定增強的訓練語料庫，隨時間將 VAE 偏向具有改善靶標結合能力的化學空間區域
  2. 構成最終用於體外和體內驗證的頂級候選化學物質庫

**關鍵洞察**：
- 嵌套 AL 循環能夠快速探索具有高體外預測分數的分子空間，預期減少假陽性
- 不僅生成具有高對接分數的類藥分子，還確保 scaffold 多樣性和良好的合成可及性

---

### 3. Enhancing Generative Molecular Design via Uncertainty-guided Fine-tuning of VAE

**論文連結**：https://arxiv.org/html/2405.20573v1

**發表時間**：2024年5月

**核心方法**：
提出一種基於模型不確定性引導的 VAE fine-tuning 新方法，通過主動學習設定中的性能反饋實現。

**主要創新**：
- 量化生成模型中的模型不確定性
- 在 VAE 高維參數的低維活躍子空間中工作，解釋模型輸出的大部分變異性
- 模型不確定性的納入通過解碼器多樣性擴展了可行分子的空間
- 通過黑盒優化探索結果模型不確定性類別，由活躍子空間的低維性使其可行

**實驗驗證**：
- 在六個目標分子性質上進行實驗
- 使用多個基於 VAE 的生成模型（JT-VAE, SELFIES-VAE, SMILES-VAE）
- 與兩種優化方法結合：貝葉斯優化（BO）和 REINFORCE
- 不確定性引導的 fine-tuning 方法持續優於預訓練模型

**技術細節**：
- 在每次 REINFORCE 迭代中，從分佈 p(ω; μ_f, σ_f) 中抽樣並計算對應的 φ(μ_f, σ_f, Q)
- 使用 Adam 優化器進行訓練
- 通過利用模型不確定性來增強下游性能的閉環優化方案

---

### 4. Combining Predictive Models and Reinforcement Learning for Tailored Molecule Generation

**論文連結**：https://www.sciencedirect.com/science/article/abs/pii/B978044328824150507X, [PDF](./pdf_repository/vae/Combining_CACE_2024.pdf)

**發表時間**：2024年6月

**核心方法**：
將預測模型和強化學習結合用於定制分子生成的方法論。

**方法論整合**：
- **生成 AI**：使用 SELFIES（Self-Referencing Embedded Strings）分子表示與深度生成模型（VAE）
- **預測建模**：採用圖神經網絡（GNN）進行性質預測，無需傳統 QSPR 中的資訊描述符
- **強化學習**：作為分子生成的關鍵引導機制，使用 VAE。提出定制的 RL 學習算法來引導 VAE 的潛在表示生成具有特定期望性質的分子

**Transfer Learning 策略**：
- VAE 首先在大型聚合物數據庫（超過 20,000 個分子）上預訓練
- 然後在較小的非離子表面活性劑數據集上 fine-tuning
- **關鍵發現**：在預訓練 VAE 上使用較小數據集進行 fine-tuning，擴展了潛在空間的分佈，超越了僅在較小數據集上訓練的結果

**案例研究**：表面活性劑分子的臨界膠束濃度（CMC）

**實驗結果**：
- Fine-tuning 過程使潛在空間中的分佈擴展
- VAE + RL 生成的分子形成一個獨特的聚類（綠色），在較大的 VAE 聚類（紅色）內
- 綠色聚類定義明確且擴散較少，表明 RL 成功磨練了分子生成過程
- 生成的分子符合指定的性質閾值，導致更集中的結果

**潛在空間可視化**：
- 展示了 64 維潛在空間的兩個主成分
- VAE 聚類（紅點）：僅由 VAE 模型生成的分子
- VAE+RL 聚類（綠點）：由 VAE 模型在 RL 引導下生成的分子
- RL 引導方法對生成的分子有顯著影響

---

### 5. ScaffoldGVAE: Scaffold Generation and Hopping via Multi-view Graph VAE

**論文連結**：https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00766-0

**期刊**：Journal of Cheminformatics

**發表時間**：2023年10月

**核心創新**：
基於多視圖圖神經網絡的變分自編碼器，專門用於藥物分子的 scaffold 生成和 scaffold hopping。

**模型特點**：
- 整合了幾個重要組件：
  - Node-central 和 edge-central message passing
  - Side-chain embedding
  - Scaffold 的 Gaussian mixture distribution

**Fine-tuning 策略**：
- 針對五個激酶靶標（kinase targets），預先定義參考化合物的 scaffold
- 基於鉸鏈結合物附近的藥效團核心結構確定 scaffold，因為結合到鉸鏈區域的結構是激酶抑制劑設計中最重要的部分
- 使用 fine-tuned 模型採樣 5,000 個新 scaffold
- 根據為每個參考化合物添加側鏈的原則將 scaffold 安裝到分子中
- 對每個激酶生成 100,000 個分子

**基準模型比較**：
- VAE、AAE、LatentGAN 和 QBMG 模型按照 MOSES 框架提出的管道在五個靶標上進行預訓練和 fine-tuning

**評估指標**：
- 7 個通用生成模型評估指標（GEM）
- 4 個 scaffold hopping 生成模型評估指標（SEM）

**消融實驗結果**：
- 缺少 node-central MPN 的模型（Model 1）
- 缺少 edge-central MPN 的模型（Model 2）
- 缺少 side-chain embedding 的模型（Model 3）
- 這些模型的性能都略遜於完整模型（Model 6）
- 沒有 side-chain-adding 策略的直接分子到分子生成模型（Model 4）表現較差
- Model 5（本質上是圖 VAE 模型）表現最差，成功率僅約 10%

---

### 6. Generative and Reinforcement Learning for Automated De Novo Design

**論文連結**：https://www.nature.com/articles/s42004-022-00733-0, [PDF](./pdf_repository/vae/Generative_CC_2022.pdf)

**期刊**：Communications Chemistry

**發表時間**：2022年10月

**研究重點**：
優化生成模型與強化學習的協議，應用於設計高效的表皮生長因子受體（EGFR）抑制劑。

**核心挑戰**：
針對特定靶標的活性分子是稀有事件，生成模型可能很少觀察到有前景的分子，導致"過度探索"問題。

**解決方案**：
- 使用經典策略梯度算法結合以下啟發式方法來平衡利用和探索：
  1. **通過遷移學習進行 Fine-tuning**：使用高獎勵的生成分子進行 fine-tuning
  2. **Experience Replay**：維護一個 replay buffer，存儲預測活性超過概率閾值的分子

**訓練協議**：
- 模型在 ChEMBL 數據上預訓練
- 訓練 20 個 epoch，每個 epoch 包含三個步驟：
  1. Policy gradient
  2. Policy experience replay
  3. Fine-tuning
- 每個步驟結束時生成 3,200 個分子
- 預測活性超過概率閾值的分子被納入 replay buffer
- Replay buffer 反過來影響 policy replay 和 fine-tuning 步驟的訓練

**關鍵發現**：
- Fine-tuning 通過遷移學習使用高獎勵範例是探索的第一個算法進步
- Replay buffer 的初始化會影響模型行為
- 通過調整三個步驟的迭代次數可以理解它們對訓練的影響

---

### 7. Improving Targeted Molecule Generation through Language Model Fine-Tuning

**論文連結**：https://arxiv.org/html/2405.06836v1

**發表時間**：2024年5月

**方法**：
使用強化學習通過語言模型 fine-tuning 改進目標分子生成。

**技術實現**：
- 使用 Transformer Reinforcement Learning (TRL) 庫進行 fine-tuning
- 採用 Proximal Policy Optimization (PPO) 方法
- 整合 MolT5 模型（最初 fine-tuned 用於基於輸入蛋白質的化合物生成）

**獎勵計算**：
- 結合藥物-靶標相互作用（DTI）評估和分子有效性評估
- β 值（對無效分子的獎勵懲罰）通過試錯確定，範圍從 0.1 到 0.7
- 經過嚴格評估，0.3 產生最佳結果

**實驗結果**：
- 展示了 RL fine-tuning 前後生成的有效分子百分比
- 驗證了方法的有效性，通過實驗改進模型專注於生成有效分子
- 修改獎勵計算以優先考慮有效性而非評估藥物-靶標相互作用

---

### 8. Molecular Generative Model Based on Conditional VAE

**論文連結**：https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0286-7

**期刊**：Journal of Cheminformatics

**發表時間**：2018年7月

**核心方法**：
使用條件變分自編碼器（CVAE）進行多變量控制的分子生成模型。

**CVAE 與 VAE 的關鍵區別**：
- 分子性質直接整合到編碼器和解碼器中
- 生成的潛在向量由兩部分組成：
  1. 第一部分用於目標分子性質
  2. 第二部分涉及分子結構和其他性質
- 可以通過設置條件向量將期望的分子性質嵌入目標分子結構中
- 可以獨立控制結構和性質（除了性質與分子 scaffold 強耦合的情況）

**應用場景**：
同時控制五個目標性質生成類藥分子：
- 分子量（MW）
- 分配係數（LogP）
- 氫鍵供體數（HBD）
- 氫鍵受體數（HBA）
- 拓撲極性表面積（TPSA）

**優勢**：
- CVAE 對潛在空間的連續性和平滑性不太敏感
- 不需要潛在空間關於分子結構潛在向量的導數
- 資訊分離使得生成新分子時對分子結構和性質的控制更加靈活

**潛在空間分析**：
- 使用主成分分析（PCA）提取兩個主軸
- 在聯合訓練的 VAE 中，具有相似性質的分子可能位於潛在空間的同一區域附近
- 在 CVAE 模型中，潛在向量的特定區域不一定與目標分子性質有相關性（由條件向量控制）

---

## 技術要點總結

### Fine-tuning 策略

1. **預訓練 + 特定數據集 Fine-tuning**：
   - 先在大型通用化學數據集上預訓練 VAE
   - 然後在小型目標特定數據集上 fine-tune
   - 預訓練使 VAE 學習通用分子特徵，fine-tuning 時能更好地泛化

2. **迭代 Fine-tuning**：
   - 每次循環後使用篩選的分子 fine-tune VAE
   - 從預訓練權重或上一次的權重繼續訓練
   - 累積更多符合條件的分子數據

3. **雙層嵌套循環**：
   - 內層循環：快速化學性質篩選 + fine-tuning（快速迭代）
   - 外層循環：深度物理模擬驗證 + fine-tuning（高質量篩選）

### 評估指標

**化學性質指標**：
- QED (Quantitative Estimate of Drug Likeliness): 0-1，越接近1越類藥
- SA Score (Synthetic Accessibility): 1-10，越小越容易合成
- Tanimoto Similarity: 0-1，評估與參考分子的相似度

**親和力指標**：
- Glide gscore: 分子對接評分，越負越好（單位：kcal·mol⁻¹）
- PELE BFE: PELE 軟體計算的結合自由能
- ABFE: 絕對結合自由能（可轉換為 Kd 值）

### 常見挑戰與解決方案

**挑戰1：數據稀缺**
- 解決方案：使用高通量虛擬篩選（HTVS）生成虛擬數據
- 案例：KRAS 研究中使用 Enamine REAL 子集的 HTVS 結果

**挑戰2：過度探索 vs 過度利用**
- 解決方案：
  - 使用相似度閾值控制探索範圍
  - 動態調整相似度閾值（如從 0.6 降至 0.4）
  - 使用 experience replay buffer
  - 強化學習結合策略梯度和 fine-tuning

**挑戰3：合成可及性**
- 解決方案：
  - 將 SA score 納入篩選標準
  - 使用反應性構建塊的採樣方法
  - 估計器通過強化學習整合

**挑戰4：模型產生無效分子**
- 解決方案：
  - 使用 ChemFixer 等修正框架
  - 數據增強技術
  - Mutual information driven training protocol

---

## 實驗設計建議

### 基本工作流程

```
1. 數據準備
   └─ 通用數據集（如 ChEMBL, ZINC）
   └─ 目標特定數據集（實驗數據或 HTVS）

2. 預訓練
   └─ 在通用數據集上訓練 VAE
   └─ 驗證重構能力

3. Fine-tuning（循環）
   ├─ 在目標數據集上 fine-tune
   ├─ 生成候選分子（採樣潛在空間）
   ├─ 篩選分子（化學性質、親和力）
   ├─ 累積符合條件的分子
   └─ 使用累積數據 re-fine-tune

4. 候選驗證
   ├─ 深度模擬（PELE, ABFE）
   ├─ 視覺檢查結合姿態
   └─ 實驗驗證（合成、生物活性測試）
```

### 參數設置參考

基於 CDK2 案例：
- 內層循環閾值：QED > 0.6, SA < 7.0, Similarity < 0.6
- 外層循環閾值：Glide gscore ≤ -8.0 kcal·mol⁻¹
- 每個內層循環：生成數千至數萬個分子
- 嵌套循環：通常 15-20 個內層循環，4-5 個外層循環

基於表面活性劑案例：
- VAE 預訓練：20,000+ 聚合物分子
- Fine-tuning：特定表面活性劑數據集（較小）
- Epoch 數：預先確定，直到收斂（重構損失和 KL 散度最小化）

---

## 軟體工具與資源

### 開源代碼

1. **ALGen-1**（Communications Chemistry 研究）
   - 網址：https://github.com/IFilella/ALGen-1
   - 許可：CC BY-NC-SA 4.0（學術非商業）
   - 功能：完整的 VAE-AL 生成模型工作流程

2. **MolecularAnalysis**
   - 網址：https://github.com/IFilella/MolecularAnalysis
   - 功能：分子分析和可視化 Python 腳本

3. **TransVAE**
   - 網址：https://github.com/oriondollar/TransVAE
   - 功能：Transformer-based VAE 模型

### 關鍵軟體套件

**分子表示與化學信息學**：
- RDKit：開源化學信息學工具包
- Scopy：包含 SAScore 模組

**深度學習框架**：
- PyTorch / TensorFlow
- MOSES：分子生成模型基準測試框架

**分子建模**：
- Schrödinger Suite：
  - Glide（分子對接）
  - LigPrep（配體準備）
  - SiteMap（結合位點分析）
- PELE：蛋白質能量景觀探索（Monte Carlo）
- ABFE 模擬工具

**分子表示格式**：
- SMILES（Simplified Molecular Input Line Entry System）
- SELFIES（Self-Referencing Embedded Strings）：更適合深度學習

**可視化**：
- UMAP（Uniform Manifold Approximation and Projection）：潛在空間可視化
- MarvinSketch：分子結構可視化
- Seaborn / Matplotlib：數據可視化

---

## 未來研究方向

1. **更複雜的模型架構**：
   - Two-stage VAE
   - VampPrior 替代 Gaussian prior
   - 結合 Transformer 的 VAE

2. **改進的訓練技術**：
   - 更複雜的採樣方案
   - 端到端調優方法
   - 多目標優化策略

3. **增強泛化能力**：
   - 解決適用域問題
   - 跨靶標遷移學習
   - Few-shot learning 應用

4. **實驗驗證閉環**：
   - 更緊密整合計算和實驗
   - 自動化合成和測試
   - 人在環路的主動學習

5. **可解釋性研究**：
   - 理解模型如何直覺和壓縮結構資訊
   - 開發新的目標函數
   - 潛在空間組織的深入分析

---

## 關鍵論文時間線

```
2018 ──┐
       │ Conditional β-VAE for Molecular Generation (CVAE 多變量控制)
       │ Journal of Cheminformatics
       │
2022 ──┤
       │ Generative and Reinforcement Learning for De Novo Design
       │ Communications Chemistry (EGFR 抑制劑)
       │
2023 ──┤
       │ ScaffoldGVAE (Scaffold Hopping)
       │ Journal of Cheminformatics
       │
2024 ──┤
       │ Combining Predictive Models and RL (表面活性劑)
       │ ScienceDirect
       │
       │ Uncertainty-guided Fine-tuning of VAE
       │ arXiv
       │
       │ Language Model Fine-Tuning via RL (MolT5)
       │ arXiv
       │
2025 ──┤
       │ Active Learning-Guided Seq2Seq VAE (多靶點抑制劑)
       │ arXiv (最新)
       │
       └ Optimizing Drug Design with VAE-AL (CDK2, KRAS)
         Communications Chemistry (最完整實驗驗證)
```

---

## 參考文獻

### 主要論文（按發表時間排序）

1. Kim, S., et al. (2018). "Molecular generative model based on conditional variational autoencoder for de novo molecular design." *Journal of Cheminformatics*, 10, 31. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0286-7

2. Korshunova, M., et al. (2022). "Generative and reinforcement learning approaches for the automated de novo design of bioactive compounds." *Communications Chemistry*, 5, 1-11. https://www.nature.com/articles/s42004-022-00733-0

3. Zhao, T., et al. (2023). "ScaffoldGVAE: scaffold generation and hopping of drug molecules via a variational autoencoder based on multi-view graph neural networks." *Journal of Cheminformatics*, 15, 91. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00766-0

4. Abeer, A.N.M.N., et al. (2024). "Enhancing Generative Molecular Design via Uncertainty-guided Fine-tuning of Variational Autoencoders." *arXiv preprint* arXiv:2405.20573. https://arxiv.org/abs/2405.20573

5. Hosseini, S.R., et al. (2024). "Improving Targeted Molecule Generation through Language Model Fine-Tuning Via Reinforcement Learning." *arXiv preprint* arXiv:2405.06836. https://arxiv.org/html/2405.06836v1

6. Kumar, A., et al. (2024). "Combining Predictive Models and Reinforcement Learning for Tailored Molecule Generation." In *Computer Aided Chemical Engineering* (Vol. 53, pp. 1981-1986). Elsevier. https://www.sciencedirect.com/science/article/abs/pii/B978044328824150507X

7. Filella-Merce, I., et al. (2025). "Optimizing drug design by merging generative AI with a physics-based active learning framework." *Communications Chemistry*, 8, 238. https://www.nature.com/articles/s42004-025-01635-7

8. Park, J., et al. (2025). "Active Learning-Guided Seq2Seq Variational Autoencoder for Multi-target Inhibitor Generation." *arXiv preprint* arXiv:2506.15309. https://arxiv.org/html/2506.15309

### 相關論文

9. Gómez-Bombarelli, R., et al. (2018). "Automatic chemical design using a data-driven continuous representation of molecules." *ACS Central Science*, 4(2), 268-276.

10. Polykovskiy, D., et al. (2020). "Molecular sets (MOSES): a benchmarking platform for molecular generation models." *Frontiers in Pharmacology*, 11, 565644.

11. Yoshikai, Y., et al. (2024). "A novel molecule generative model of VAE combined with Transformer for unseen structure generation." *arXiv preprint* arXiv:2402.11950. https://arxiv.org/abs/2402.11950

12. Mizuno, T., et al. (2023). "Variational autoencoder-based chemical latent space for large molecular structures with 3D complexity." *Communications Chemistry*, 6, 250. https://www.nature.com/articles/s42004-023-01054-6

---

## 筆記更新記錄

- **2025-11-23**：建立初始筆記，整理 8 篇主要研究論文
- 重點：VAE 分子生成的迭代 fine-tuning 策略與實驗驗證

---

*本筆記由 Claude (Sonnet 4.5) 協助整理，基於學術論文檢索和分析*
