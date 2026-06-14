# MLA Flash Attention 基准测试结果分析

> 数据来源: `consolidated_mla_results/`  
> 平台: Wormhole N300s (WH) vs NVIDIA L40S FlashInfer MLA 参考  
> 方法: S-FMLA, FlashMLA, MLA baseline, L40S FlashMLA (raw & /3 带宽归一化)

---

## 数据概览

| 指标 | 数值 |
|------|------|
| 总测点 | 560 (ok=383, error=9, skipped=168) |
| L40S 参考 | 140 点 (全部 ok) |
| Wormhole | 420 点 (ok=243, error=9, skipped=168) |
| Decode sweep | batch ∈ {1,2,4,8,16,32,64}, seq ∈ {256..128K} |
| Prefill sweep | 同上；seq>4K 或 batch×seq>16K 多为 skipped (安全上限) |

### 方法对比说明

- **S-FMLA**: Wormhole 上优化的 sparse flash MLA 实现（本报告主要 baseline）
- **FlashMLA**: Wormhole flash MLA 实现
- **MLA baseline**: Wormhole 标准 MLA
- **L40S FlashMLA**: GPU 参考；**/3** 为带宽归一化（约 3× 原始延迟），用于与 WH 公平对比

---

## 可视化图表

图表保存在 `viz/` 目录:

| 文件 | 内容 |
|------|------|
| `01_decode_latency_curves.png` | Decode 延迟随 seq_len 变化 (B=1,8,32) |
| `02_prefill_latency_curves.png` | Prefill 延迟曲线 |
| `03_decode_flash_over_sfmla_heatmap.png` | Decode FlashMLA/S-FMLA 比值热力图 |
| `04_decode_mla_over_sfmla_heatmap.png` | Decode MLA baseline 相对 S-FMLA 慢多少 |
| `05_prefill_flash_over_sfmla_heatmap.png` | Prefill FlashMLA/S-FMLA |
| `06_decode_l40s_div3_over_sfmla_heatmap.png` | Decode L40S/3 vs S-FMLA |
| `07-08_*` | L40S/3 vs S-FMLA 曲线 (decode/prefill) |
| `09-10_*` | FlashMLA vs S-FMLA 散点图 |
| `11_*` | MLA baseline 随 batch 放大倍数 |
| `12_wh_coverage_status.png` | WH 测试覆盖/跳过/失败统计 |

---

## 核心统计摘要

### Decode
- 有效 S-FMLA 测点: 59；FlashMLA 有效: 59；两者同时有效: 59
- FlashMLA / S-FMLA: median=1.061x, mean=1.106x, min=0.810x, max=1.576x
- FlashMLA 快于 S-FMLA 的配置: 9/59 (15.3%)
- MLA baseline / S-FMLA: median=31.0x, max=145.4x
- L40S/3 vs S-FMLA: median ratio=0.446x; S-FMLA 更快于 L40S/3 的配置: 9/59 (15.3%)

### Prefill
- 有效 S-FMLA 测点: 28；FlashMLA 有效: 28；两者同时有效: 28
- FlashMLA / S-FMLA: median=1.960x, mean=1.879x, min=0.972x, max=2.291x
- FlashMLA 快于 S-FMLA 的配置: 2/28 (7.1%)
- MLA baseline / S-FMLA: median=4.2x, max=6.1x
- L40S/3 vs S-FMLA: median ratio=0.925x; S-FMLA 更快于 L40S/3 的配置: 13/28 (46.4%)

### Decode — S-FMLA 快于 L40S/3 的 Top 5 配置
(ratio = L40S/3 ÷ S-FMLA，>1 表示 S-FMLA 更快)

- B=16, L=128K: S-FMLA=7.466ms, ratio=2.225x
- B=16, L=64K: S-FMLA=3.933ms, ratio=2.077x
- B=16, L=32K: S-FMLA=2.034ms, ratio=1.635x
- B=8, L=128K: S-FMLA=4.803ms, ratio=1.621x
- B=32, L=64K: S-FMLA=9.070ms, ratio=1.418x

### Decode — L40S/3 快于 S-FMLA 的 Top 5 配置
(ratio < 1 表示 L40S/3 更快)

- B=1, L=64K: S-FMLA=2.336ms, ratio=0.153x
- B=1, L=128K: S-FMLA=4.471ms, ratio=0.167x
- B=1, L=32K: S-FMLA=1.295ms, ratio=0.177x
- B=1, L=16K: S-FMLA=0.777ms, ratio=0.251x
- B=2, L=32K: S-FMLA=1.305ms, ratio=0.258x

### Prefill — L40S/3 快于 S-FMLA 的配置 (短 seq)
- B=64, L=512: ratio=0.01x
- B=32, L=1K: ratio=0.01x
- B=64, L=1K: ratio=0.02x

### Prefill — S-FMLA 快于 L40S/3 的配置 (长 seq / 大 batch)
- B=1, L=256: ratio=7.32x
- B=2, L=256: ratio=5.23x
- B=1, L=512: ratio=4.27x