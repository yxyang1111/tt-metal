# Decode Effective Read Bandwidth

This report estimates effective KV-cache read bandwidth from decode latency.
The default input is raw WH sweep JSON, so S-FMLA rows are not plot-corrected.

Byte-count conventions:

- `single_pass_*`: one ideal read of the BF8_B latent KV cache.
- `estimated_current_*`: assumes the current H=32 decode path rereads the cache once per 8-head group (4 groups). Values above the peak reference mean this byte model is too pessimistic for actual DRAM reads or the peak reference is not comparable.
- `*_tiled_*`: uses 1088 bytes per BF8_B 32x32 tile; this is usually closer to DRAM tile traffic than the logical element count.
- Peak reference used for percentage columns: 258 GB/s.

| Method | B | L | Latency ms | Single-pass tiled GB/s | Est 4x tiled GB/s | 4x % peak |
| --- | --- | --- | --- | --- | --- | --- |
| flash_mla | 1 | 8K | 0.497 | 10.1 | 40.4 | 15.6 |
| flash_mla | 2 | 8K | 0.544 | 18.4 | 73.7 | 28.6 |
| flash_mla | 4 | 8K | 0.519 | 38.6 | 154.5 | 59.9 |
| flash_mla | 8 | 8K | 0.490 | 81.8 | 327.2 | 126.8 |
| flash_mla | 16 | 8K | 0.819 | 98.0 | 392.0 | 151.9 |
| flash_mla | 32 | 8K | 1.481 | 108.3 | 433.2 | 167.9 |
| flash_mla | 1 | 16K | 0.771 | 13.0 | 52.0 | 20.2 |
| flash_mla | 2 | 16K | 0.762 | 26.3 | 105.3 | 40.8 |
| flash_mla | 4 | 16K | 0.818 | 49.0 | 196.1 | 76.0 |
| flash_mla | 8 | 16K | 0.827 | 97.0 | 388.2 | 150.5 |
| flash_mla | 16 | 16K | 1.228 | 130.7 | 522.8 | 202.6 |
| flash_mla | 32 | 16K | 2.522 | 127.2 | 509.0 | 197.3 |
| flash_mla | 1 | 32K | 1.343 | 14.9 | 59.7 | 23.1 |
| flash_mla | 2 | 32K | 1.363 | 29.4 | 117.7 | 45.6 |
| flash_mla | 4 | 32K | 1.346 | 59.6 | 238.3 | 92.4 |
| flash_mla | 8 | 32K | 1.398 | 114.8 | 459.1 | 178.0 |
| flash_mla | 16 | 32K | 2.019 | 158.9 | 635.7 | 246.4 |
| flash_mla | 32 | 32K | 4.793 | 133.9 | 535.5 | 207.6 |
| flash_mla | 1 | 64K | 2.413 | 16.6 | 66.5 | 25.8 |
| flash_mla | 2 | 64K | 2.409 | 33.3 | 133.2 | 51.6 |
| flash_mla | 4 | 64K | 2.404 | 66.7 | 266.9 | 103.5 |
| flash_mla | 8 | 64K | 2.616 | 122.7 | 490.6 | 190.2 |
| flash_mla | 16 | 64K | 3.814 | 168.3 | 673.1 | 260.9 |
| flash_mla | 32 | 64K | 9.225 | 139.1 | 556.5 | 215.7 |
| sfmla | 1 | 8K | 0.457 | 11.0 | 43.9 | 17.0 |
| sfmla | 2 | 8K | 0.459 | 21.8 | 87.3 | 33.8 |
| sfmla | 4 | 8K | 0.569 | 35.2 | 141.0 | 54.6 |
| sfmla | 8 | 8K | 0.527 | 76.1 | 304.4 | 118.0 |
| sfmla | 16 | 8K | 0.783 | 102.4 | 409.5 | 158.7 |
| sfmla | 32 | 8K | 1.353 | 118.5 | 474.2 | 183.8 |
| sfmla | 1 | 16K | 0.777 | 12.9 | 51.6 | 20.0 |
| sfmla | 2 | 16K | 0.753 | 26.6 | 106.5 | 41.3 |
| sfmla | 4 | 16K | 0.792 | 50.7 | 202.7 | 78.6 |
| sfmla | 8 | 16K | 0.816 | 98.3 | 393.4 | 152.5 |
| sfmla | 16 | 16K | 1.172 | 136.9 | 547.7 | 212.3 |
| sfmla | 32 | 16K | 2.476 | 129.6 | 518.5 | 201.0 |
| sfmla | 1 | 32K | 1.295 | 15.5 | 61.9 | 24.0 |
| sfmla | 2 | 32K | 1.305 | 30.7 | 122.9 | 47.6 |
| sfmla | 4 | 32K | 1.289 | 62.2 | 248.8 | 96.4 |
| sfmla | 8 | 32K | 1.335 | 120.1 | 480.5 | 186.3 |
| sfmla | 16 | 32K | 2.034 | 157.7 | 631.0 | 244.6 |
| sfmla | 32 | 32K | 4.697 | 136.6 | 546.5 | 211.8 |
| sfmla | 1 | 64K | 2.336 | 17.2 | 68.7 | 26.6 |
| sfmla | 2 | 64K | 2.348 | 34.2 | 136.7 | 53.0 |
| sfmla | 4 | 64K | 2.370 | 67.7 | 270.8 | 105.0 |
| sfmla | 8 | 64K | 2.462 | 130.3 | 521.3 | 202.1 |
| sfmla | 16 | 64K | 3.933 | 163.2 | 652.7 | 253.0 |
| sfmla | 32 | 64K | 9.070 | 141.5 | 566.0 | 219.4 |

Suggested raw sweep command before rerunning this report:

```bash
python mla_flash_attention_dev/experiments/run_wh_mla_batch_seq_sweep.py \
  --methods flash_mla sfmla --modes decode \
  --batches 1 2 4 8 16 32 \
  --seq-lens 256 512 1024 2048 4096 8192 16384 32768 65536 \
  --warmup 2 --iters 10 --resume
```

Then run:

```bash
python mla_flash_attention_dev/experiments/characterization/compute_decode_effective_read_bandwidth.py
```

For actual NoC-event measured DRAM read bytes rather than byte-model estimates, extend or run `characterization/measure_dram_traffic.py` on selected cases.

