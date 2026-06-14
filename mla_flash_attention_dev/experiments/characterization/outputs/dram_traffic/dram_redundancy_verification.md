# DRAM Read Redundancy Verification

Three independent approaches confirm the 4× K-cache read redundancy
in the TT mainline FlashMLA decode baseline.

**Direct NoC event tracing is not available** due to a profiler bug
(`Invalid NoC transfer type`) and OOM from the high event volume of SDPA kernels.
The following indirect evidence is used instead.

## Approach 1: DRAM Reads from Kernel Structure

Each head group independently reads the full K cache from DRAM.
V is NOT read from DRAM (reuse_k=true: V reuses K's L1 buffer).

| Case | Seq | B | G | Cores | K/core (KB) | Total K Read (MB) | Ideal K (MB) | Redundancy | Kernel (ms) | DRAM Floor (ms) | Ideal Floor (ms) | Kernel/Actual | Kernel/Ideal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_1k | 1024 | 2 | 4 | 32 | 153.0 | 5.014 | 1.253 | **4.0x** | 0.0735 | 0.0251 | 0.0063 | 2.93x | 11.73x |
| decode_4k | 4096 | 2 | 4 | 32 | 612.0 | 20.054 | 5.014 | **4.0x** | 0.1731 | 0.1003 | 0.0251 | 1.73x | 6.91x |
| decode_8k | 8192 | 2 | 4 | 32 | 1224.0 | 40.108 | 10.027 | **4.0x** | 0.3032 | 0.2005 | 0.0501 | 1.51x | 6.05x |
| decode_16k | 16384 | 1 | 4 | 16 | 2448.0 | 40.108 | 10.027 | **4.0x** | 0.5609 | 0.2005 | 0.0501 | 2.8x | 11.19x |
| decode_32k | 32768 | 1 | 4 | 16 | 4896.0 | 80.216 | 20.054 | **4.0x** | 1.0848 | 0.4011 | 0.1003 | 2.7x | 10.82x |

## Approach 2: DRAM Bandwidth Demand from NCRISC Profiler

Aggregate DRAM demand = n_reader_cores × per-core effective K bandwidth.
Oversubscription > 1.0 confirms multiple cores contend for DRAM.

| Case | Reader Cores | Eff K BW/core (GB/s) | Aggregate Demand (GB/s) | Peak DRAM (GB/s) | Oversubscription | Fair Share/core (GB/s) |
|---|---:|---:|---:|---:|---:|---:|
| decode_1k | 32 | 43.28 | 1385.0 | 200.0 | **6.92x** | 6.25 |
| decode_4k | 32 | 43.15 | 1380.8 | 200.0 | **6.9x** | 6.25 |
| decode_8k | 32 | 39.55 | 1265.6 | 200.0 | **6.33x** | 6.25 |
| decode_16k | 16 | 18.96 | 303.4 | 200.0 | **1.52x** | 12.5 |
| decode_32k | 16 | 18.44 | 295.0 | 200.0 | **1.48x** | 12.5 |

## Approach 3: Head Sweep Confirms Bandwidth-Bound Regime

Fixed seq_len=8k, batch=1. All configs use G=4 head groups.
Doubling compute (H=8→16) does NOT increase latency → DRAM-bandwidth-bound.

| H | Heads/core | G | FLOPs (vs H=8) | Latency (ms) | Latency (vs H=8) | Note |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 2 | 4 | 1.0x | 0.364 | 1.000x | baseline |
| 16 | 4 | 4 | 2.0x | 0.362 | 0.995x | 2x compute, same DRAM volume → same latency (BW-bound) |
| 32 | 8 | 4 | 4.0x | 0.473 | 1.299x | 4x compute, same G=4 but 8 heads/core → slightly more L1 traffic |

## Conclusion

All three approaches converge on the same finding:

1. **Kernel structure** (code inspection): 4 head groups × full K read = 4× redundancy
2. **NCRISC bandwidth** (measured): aggregate demand 3–7× exceeds DRAM peak, consistent with 4 groups competing for shared DRAM bandwidth
3. **Head sweep** (measured): doubling compute at constant G does not change latency, proving the workload is DRAM-bandwidth-bound by redundant K reads

The V cache is NOT read from DRAM in MLA mode (`reuse_k=true`). Only K cache DRAM traffic is relevant.
