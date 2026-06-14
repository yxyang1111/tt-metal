# DRAM Read Traffic: Measured vs Theoretical

V is NOT read from DRAM in MLA mode (`reuse_k=true`): V reuses K's L1 buffer.
Only K cache DRAM reads are counted.

| Case | Seq Len | B | G | Theo Logical (MB) | Theo Tiled (MB) | Theo Ideal (MB) | Measured DRAM Read (MB) | Meas / Tiled | Redundancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | 1024 | 2 | 4 | 4.719 | 5.014 | 1.253 | n/a | n/a | 4x |

**Notes:**
- *Theo Logical*: logical K bytes = B × G × L × (d_c+d_r) × 1B (BF8_b)
- *Theo Tiled*: K bytes with tile layout overhead (BF8_b tile ≈ 1088B)
- *Theo Ideal*: single-pass K read with multicast (G=1)
- *Measured DRAM Read*: sum of `num_bytes` for NoC READ events targeting DRAM bank coordinates
- *Meas / Tiled*: ratio close to 1.0 confirms the theoretical model; >1 includes page table + Q reads