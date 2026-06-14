# Baseline MLA Characterization Report

## 1. Overview

This report characterizes the TT mainline FlashMLA baseline on Wormhole,
quantifying three key inefficiency sources that motivate the SF-MLA dataflow.

## 2. Data Reuse Inefficiency

The TT mainline baseline reads the shared latent K cache **independently per head group**, with no hardware multicast. At decode_32k (seq_len=32768), the shared KV cache is 37.749 MB per batch element, but 4 head groups each read the full cache independently, resulting in a **4x redundancy** and 75.0% wasted DRAM traffic.

With multicast, the ideal DRAM read volume would be 71.3 MB instead of the actual 285.21 MB — a 4.0x reduction in off-chip traffic.

## 3. Bandwidth and Compute Analysis

Despite the high DRAM traffic volume, effective K read bandwidth per core drops from 43.28 GB/s at decode_1k to 18.44 GB/s at decode_32k, indicating increasing DRAM bank contention from concurrent independent reads.

FPU utilization (from hardware counters) ranges from 12.17% at decode_1k to 25.61% at decode_32k — well below the hardware peak of 74.0 TFLOP/s.

## 4. Pipeline Coupling

The decoupled reader–compute–writer pipeline transitions through four phases as sequence length grows:
  - **decode_1k**: compute_on_critical_path (NCRISC 37.1%, BRISC 83.5%, Compute 99.6%)
  - **decode_4k**: writer_close_to_critical_path (NCRISC 63.2%, BRISC 93.0%, Compute 99.8%)
  - **decode_8k**: writer_close_to_critical_path (NCRISC 78.7%, BRISC 96.0%, Compute 99.9%)
  - **decode_16k**: reader_close_to_critical_path (NCRISC 88.7%, BRISC 97.8%, Compute 100.0%)
  - **decode_32k**: reader_writer_saturated (NCRISC 94.3%, BRISC 98.9%, Compute 100.0%)

At short sequences, compute dominates the critical path and the pipeline is underutilized. At long sequences, both reader and writer saturate, creating bidirectional back-pressure that prevents compute from sustaining even modest arithmetic throughput.

## 5. Opportunity Summary

Three concrete optimization targets emerge:
1. **Eliminate redundant K reads via multicast**: The shared latent cache should be read once from DRAM and broadcast to all cores via hardware multicast, reducing off-chip traffic by the redundancy factor.
2. **Co-design DRAM bank assignment with worker placement**: Independent reads from all cores contend for the same DRAM banks; bank-affinity-aware placement can recover effective per-core bandwidth.
3. **Pipeline-aware scheduling**: The decoupled pipeline's back-pressure coupling must be explicitly managed through buffering depth, chunk sizing, and overlap parameters.
