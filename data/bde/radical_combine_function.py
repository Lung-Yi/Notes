"""
radical_smiles_combiner.py
==========================
自由基SMILES組合工具 - 將兩個自由基根據未成對電子連接成完整分子

使用方法:
---------
from radical_smiles_combiner import combine_radicals

# 基本用法
molecule = combine_radicals("[CH3]", "[OH]")  # 返回 "CO" (甲醇)

# 批量處理
from radical_smiles_combiner import batch_combine
results = batch_combine([("[CH3]", "[H]"), ("[CH3]", "[CH3]")])
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Tuple, Optional, Dict


def find_radical_center(mol) -> Tuple[Optional[int], int]:
    """
    找到分子中的自由基中心（具有未成對電子的原子）
    
    Returns:
        (atom_index, num_radical_electrons) 或 (None, 0)
    """
    for atom in mol.GetAtoms():
        n_rad = atom.GetNumRadicalElectrons()
        if n_rad > 0:
            return atom.GetIdx(), n_rad
    return None, 0


def combine_radicals(
    smiles1: str, 
    smiles2: str, 
    bond_type: Chem.BondType = Chem.BondType.SINGLE
) -> Optional[str]:
    """
    將兩個自由基SMILES連接成完整分子
    
    Parameters:
        smiles1: 第一個自由基的SMILES（例如 "[CH3]", "C[C](C)C"）
        smiles2: 第二個自由基的SMILES
        bond_type: 連接鍵類型，默認為單鍵
        
    Returns:
        連接後分子的canonical SMILES，失敗則返回 None
        
    Example:
        >>> combine_radicals("[CH3]", "[OH]")
        'CO'
        >>> combine_radicals("C[C](C)C", "[F]")
        'CC(C)(C)F'
    """
    # 解析SMILES
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return None
    
    # 找到自由基中心
    idx1, rad_e1 = find_radical_center(mol1)
    idx2, rad_e2 = find_radical_center(mol2)
    
    if idx1 is None or idx2 is None:
        return None
    
    # 合併分子
    combined = Chem.CombineMols(mol1, mol2)
    rw_mol = Chem.RWMol(combined)
    
    # 計算mol2原子在合併後的新索引
    new_idx2 = idx2 + mol1.GetNumAtoms()
    
    # 添加連接鍵
    rw_mol.AddBond(idx1, new_idx2, bond_type)
    
    # 更新自由基電子數
    rw_mol.GetAtomWithIdx(idx1).SetNumRadicalElectrons(max(0, rad_e1 - 1))
    rw_mol.GetAtomWithIdx(new_idx2).SetNumRadicalElectrons(max(0, rad_e2 - 1))
    
    # 獲取最終分子
    final_mol = rw_mol.GetMol()
    
    try:
        Chem.SanitizeMol(final_mol)
    except:
        pass
    
    return Chem.MolToSmiles(final_mol)


def batch_combine(
    radical_pairs: List[Tuple[str, str]]
) -> List[Dict[str, Optional[str]]]:
    """
    批量組合自由基對
    
    Parameters:
        radical_pairs: [(radical1, radical2), ...] 自由基對列表
        
    Returns:
        [{"rad1": str, "rad2": str, "molecule": str|None}, ...]
    """
    results = []
    for rad1, rad2 in radical_pairs:
        mol = combine_radicals(rad1, rad2)
        results.append({
            "rad1": rad1,
            "rad2": rad2,
            "molecule": mol
        })
    return results


def analyze_radical(smiles: str) -> Dict:
    """
    分析自由基SMILES的詳細信息
    
    Returns:
        包含自由基中心、原子符號等信息的字典
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"無法解析SMILES: {smiles}"}
    
    idx, n_rad = find_radical_center(mol)
    
    info = {
        "input_smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "num_atoms": mol.GetNumAtoms(),
        "radical_center_idx": idx,
        "num_radical_electrons": n_rad,
        "is_valid_radical": idx is not None
    }
    
    if idx is not None:
        atom = mol.GetAtomWithIdx(idx)
        info["radical_atom"] = atom.GetSymbol()
        info["hybridization"] = str(atom.GetHybridization())
    
    return info


# 常用自由基SMILES參考
COMMON_RADICALS = {
    # 碳自由基
    "methyl": "[CH3]",           # 甲基
    "ethyl": "[CH2]C",           # 乙基
    "isopropyl": "C[CH]C",       # 異丙基
    "tert-butyl": "C[C](C)C",    # 叔丁基
    "trifluoromethyl": "F[C](F)F",  # 三氟甲基
    
    # 氮自由基
    "amino": "[NH2]",            # 氨基
    "methylamino": "C[NH]",      # 甲氨基
    "dimethylamino": "C[N]C",    # 二甲氨基
    
    # 氧自由基
    "hydroxyl": "[OH]",          # 羥基
    "methoxy": "C[O]",           # 甲氧基
    
    # 硫自由基
    "mercapto": "[SH]",          # 巰基
    "methylthio": "C[S]",        # 甲硫基
    
    # 矽自由基
    "silyl": "[SiH3]",           # 矽烷基
    "trimethylsilyl": "C[Si](C)C",  # 三甲基矽基
    
    # 磷自由基
    "phosphino": "[PH2]",        # 膦基
    
    # 鹵素自由基
    "hydrogen": "[H]",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "iodine": "[I]",
}


# ============ 命令行介面 ============

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 3:
        # 命令行使用: python radical_smiles_combiner.py "[CH3]" "[OH]"
        rad1, rad2 = sys.argv[1], sys.argv[2]
        result = combine_radicals(rad1, rad2)
        if result:
            print(f"{rad1} + {rad2} → {result}")
        else:
            print(f"無法組合: {rad1} + {rad2}")
    else:
        # 顯示範例
        print("自由基SMILES組合器")
        print("=" * 50)
        print("\n使用方法:")
        print("  python radical_smiles_combiner.py [radical1] [radical2]")
        print("\n範例:")
        
        examples = [
            ("[CH3]", "[H]", "甲烷"),
            ("[CH3]", "[CH3]", "乙烷"),
            ("[CH3]", "[OH]", "甲醇"),
            ("C[C](C)C", "[F]", "叔丁基氟"),
            ("[SiH3]", "[CH3]", "甲基矽烷"),
            ("[F]", "[F]", "氟氣"),
        ]
        
        for r1, r2, name in examples:
            result = combine_radicals(r1, r2)
            print(f"  {r1:<12} + {r2:<8} → {result:<15} ({name})")
        
        print("\n常用自由基:")
        for name, smiles in list(COMMON_RADICALS.items())[:10]:
            print(f"  {name:<18}: {smiles}")