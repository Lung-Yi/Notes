# BDE 預測模型開發筆記：數據、理論與策略

這份筆記整合了關於 BSE49 基準數據集的探討、理論計算策略建議，以及建立 BDE（鍵解離能）預測模型時可用的相關資源。

## 1. 核心基準數據集：BSE49
此數據集適合作為模型精度的「高標」驗證（Benchmark），而非大規模訓練。

* **論文標題**：*BSE49, a diverse, high-quality benchmark dataset of separation energies of chemical bonds*
* **連結/DOI**：[Scientific Data (2021) 8:300](https://doi.org/10.1038/s41597-021-01088-2)
* **數據集內容**：
    * **規模**：4502 個數據點（1969 個現有分子 + 2533 個假設分子）。
    * **鍵類型**：涵蓋 49 種 X-Y 單鍵（H, B, C, N, O, F, Si, P, S, Cl）。
    * **精度標準**：**(RO)CBS-QB3** 複合方法。這是一種接近 CCSD(T)/CBS 的高精度方法，誤差通常 < 1 kcal/mol。
    * **能量定義**：氣相、非相對論基態的**電子能量差 ($E_{elec}$)**，**不包含**零點能量 (ZPE) 與熱修正。

### 數據格式解讀 (.db 檔)
每一筆資料包含四個區塊，順序如下：
1.  **Header**：參考能量值 (kcal/mol)。
2.  **Fragment A** (`molc 1.0`): 第一個自由基產物（電荷、多重度、座標）。
3.  **Fragment B** (`molc 1.0`): 第二個自由基產物。
4.  **Parent Molecule** (`molc -1.0`): 母體分子（通常為單重態）。

---

## 2. 理論計算策略與精度評估
在建立模型或生成訓練數據時，需注意以下理論細節：

### 方法精度層級
若需自行生成額外數據，建議參考以下精度順序：
1.  **黃金標準**：**(RO)CBS-QB3** 或 **W1w** (如 BDE261)。
2.  **推薦 DFT**：**$\omega$B97X-D**, **CAM-B3LYP**, **M06-2X**。這類範圍分離 (Range-separated) 或混合泛函能較好地處理自由基與離域誤差。
3.  **基礎 DFT**：**B3LYP**。雖然計算快，但在斷鍵反應中誤差較大（~3-5 kcal/mol），需謹慎使用。

### 零點能量 (ZPE) 與熱修正處理
* **純電子能量訓練 (推薦)**：
    * 若目標是訓練機器學習位能面 (ML-PES) 或純粹比較電子結構方法，**忽略 ZPE**，直接學習 $E_{elec}$。這是最嚴謹的理論驗證方式。
* **預測實驗值 (應用)**：
    * 若需預測真實世界的 BDE (焓變 $\Delta H$)，必須加上 ZPE 和熱修正。
    * **混合策略**：使用高精度方法（如 CBS-QB3 或 高階 DFT）算能量 + 使用較低成本方法（如 B3LYP/6-31G*）算頻率。
    * **關鍵操作**：務必使用**頻率校正因子 (Frequency Scaling Factors)** 來修正 DFT 高估頻率的傾向（可參考 NIST CCCBDB 資料庫）。

---

## 3. 擴充資源：BDE 相關數據集清單
建立模型時，建議使用「大規模數據集」進行訓練，並使用「高精度/實驗數據集」進行測試。

### A. 大規模機器學習訓練集 (Training Sets)
這些數據集擁有數十萬筆資料，適合 Deep Learning / GNN 模型訓練。

| 數據集名稱 | 數據量 | 理論方法 | 論文/連結參考 | 備註 |
| :--- | :--- | :--- | :--- | :--- |
| **BDE-db2** | ~531,000 | M06-2X / def2-TZVP | [Digital Discovery 2023](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00169e) | 目前最大、元素覆蓋最廣 (含鹵素) |
| **BDE-db** | ~290,000 | M06-2X / def2-TZVP | [Nat. Commun. 2020](https://www.nature.com/articles/s41467-020-16201-z) | ALFABET 模型的訓練資料 |
| **BonDNet (BDNCM)** | ~60,000 | DFT | [Chem. Sci. 2021](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05951a) | 特色是包含**帶電分子** (-1, 0, +1) |

### B. 高精度與實驗驗證集 (Validation/Benchmark Sets)
用於評估模型在「化學精度」要求下的表現。

| 數據集名稱 | 數據量 | 來源/方法 | 論文/連結參考 | 備註 |
| :--- | :--- | :--- | :--- | :--- |
| **BSE49** | 4,502 | (RO)CBS-QB3 | [Scientific Data (2021)](https://doi.org/10.1038/s41597-021-01088-2) | 純電子能量基準 |
| **BDE261** | 261 | W1w (CCSD(T) level) | [J. Phys. Chem. A 2012](https://pubs.acs.org/doi/10.1021/jp300520r) | 極高精度理論值 |
| **iBonD / Luo's Handbook** | ~20,000 | 實驗值 (Experimental) | [CRC Handbook](https://www.taylorfrancis.com/books/mono/10.1201/9781420007282/comprehensive-handbook-chemical-bond-energies-yu-ran-luo) | 真實世界的標準答案 (Ground Truth) |

---

## 4. 實作建議流程
1.  **數據準備**：下載 **BDE-db2** 作為主要訓練資料。
2.  **特徵工程**：將分子轉換為圖形 (Graph) 或指紋 (Fingerprint) 特徵。
3.  **模型訓練**：訓練 GNN 或其他回歸模型預測 $E_{elec}$ 或 BDE。
4.  **初步驗證**：從 BDE-db2 切分出的 Test set 查看收斂狀況。
5.  **黃金驗證**：使用 **BSE49** 數據集測試模型。
    * *注意*：若模型預測的是含 ZPE 的 BDE，需先扣除 BSE49 結構計算出的 ZPE 才能與 BSE49 的 $E_{elec}$ 進行公平比較；或者直接讓模型學習 $E_{elec}$。
