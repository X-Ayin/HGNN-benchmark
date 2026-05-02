# MCM_Hypergraph_learning

在 **Masked Conditional Modeling（MCM）** 设定下，对比多种「潜结构发现前端 + 超图/GNN 后端」的 PyTorch 实验代码。任务为固定长度的二进制序列：在给定 parity 约束下随机 mask 一位，预测该位的取值（二分类）。

本目录既可作为上层仓库 [`HGNN-benchmark`](../) 的子模块使用，也可单独拷贝后配合说明放置外部依赖运行。

## 任务与数据

- **序列长度**：24  
- **Parity groups（固定）**：`[1,5,9,13]`、`[2,7,11,19]`、`[4,8,12,16,20]`  
- **数据构造**：[`data/parity_dataset.py`](data/parity_dataset.py)（`ParityConfig`、`ParitySequenceDataset`）  
- **常用指标**：验证集最佳准确率 `best_val_acc`、测试集 `test_acc`  

## 环境依赖

- **Python 3** + **PyTorch**（脚本使用 `torch`、`torch.nn`、`DataLoader`）  
- **CUDA**：可选；未安装 GPU 时自动使用 CPU  
- **SPHINX 原生前端**（`frontend_type=sphinx_native`）：需在仓库根目录下放置 SPHINX 源码目录，名称须为 `SPHINX-main`（解析逻辑见 [`models/sphinx_path.py`](models/sphinx_path.py)）  
- **NRI 前端**（`run_nri_gnn_end_to_end.py`）：需在仓库根目录下放置 [`NRI-master`](../NRI-master)（与脚本中的路径约定一致）  

## 如何从仓库根目录运行

在 **`HGNN-benchmark` 根目录** 下执行（保证 `gnn_mcm` 可作为包导入）：

```bash
python -m gnn_mcm.run_sphinx_native_end_to_end --help
python -m gnn_mcm.run_tdhnn_native_end_to_end --help
python -m gnn_mcm.run_nri_gnn_end_to_end --help
```

端到端脚本会在 `gnn_mcm/logs/` 写入按时间戳命名的 **`*.jsonl`**（逐 epoch）与 **`*_summary.json`**（汇总及配置）。

### 示例：SPHINX / EvolveHypergraph 前端 + 后端

```bash
# SPHINX 原生发现器 + HCHA 或 all_deepsets 后端
python -m gnn_mcm.run_sphinx_native_end_to_end ^
  --frontend_type sphinx_native --backend hcha --selector simple ^
  --epochs 10 --train_samples 512 --val_samples 128 --test_samples 128 ^
  --batch_size 64 --lr 0.001 --seed 42

# 静态 EvolveHypergraph 前端（无需 SPHINX 目录）
python -m gnn_mcm.run_sphinx_native_end_to_end ^
  --frontend_type evolvehypergraph_static --backend all_deepsets --selector simple ^
  --epochs 10 --batch_size 64 --train_samples 512 --val_samples 128 --test_samples 128 --seed 42
```

（Linux / macOS 将行末 `^` 改为 `\`；PowerShell 可用反引号续行，或改为单行命令。）

### 示例：TDHNN 风格前端

```bash
python -m gnn_mcm.run_tdhnn_native_end_to_end ^
  --backend all_deepsets --epochs 50 --batch_size 128 ^
  --train_samples 4000 --val_samples 1000 --test_samples 1000 --seed 42
```

### 示例：NRI 前端 + MCM GNN 后端

需事先准备好 `NRI-master` 目录：

```bash
python -m gnn_mcm.run_nri_gnn_end_to_end --help
```

### 超参搜索与其它工具脚本

| 脚本 | 说明 |
|------|------|
| [`run_sphinx_hyperparam_search.py`](run_sphinx_hyperparam_search.py) | SPHINX 端到端超参搜索 |
| [`run_tdhnn_hyperparam_search.py`](run_tdhnn_hyperparam_search.py) | TDHNN 端到端超参搜索 |
| [`run_sphinx_omega_compare.py`](run_sphinx_omega_compare.py) | `omega_strategy` 等对照 |
| [`run_experiment9_special_relations.py`](run_experiment9_special_relations.py) | 实验 9：特殊关系设定 |
| [`train_mcm.py`](train_mcm.py) | 在 **包内相对路径** 下使用的 GNN 基线训练（通常在 `gnn_mcm` 目录内 `python train_mcm.py` 运行） |

## 目录结构（概要）

```
gnn_mcm/
├── data/              # Parity 数据集
├── models/            # 后端（HCHA、AllDeepSets）、发现器、SPHINX/TDHNN 封装、mcm 接口契约
├── adapters/          # Parity → SPHINX 格式适配
├── logs/              # 运行日志（建议 .gitignore 大文件或仅提交 summary）
├── train_mcm.py       # GNN + 给定邻接模式的训练脚本
└── run_*_end_to_end.py / run_*_hyperparam_search.py
```

## 接入新后端

后端输入约定与校验见 [`models/mcm_interface.py`](models/mcm_interface.py)，扩展步骤见 [`BACKEND_INTEGRATION_CHECKLIST.md`](BACKEND_INTEGRATION_CHECKLIST.md)。

## 说明

- 部分入口脚本通过 `sys.path` 将 **仓库根目录** 加入路径，以便 `import gnn_mcm`；请始终在 **`HGNN-benchmark` 根** 下使用 `python -m gnn_mcm.<模块>`，或与该布局等效的路径。  
- `train_mcm.py` 使用 `from data...` / `from models...`，适合在 `gnn_mcm` 文件夹内直接运行，与 `-m gnn_mcm` 混用时注意当前工作目录。  

---

如需上层 benchmark 的总览与实验结论，参见仓库根目录的 [`benchmark_handoff.md`](../benchmark_handoff.md)。
