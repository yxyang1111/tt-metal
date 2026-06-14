# Decode preset top-1 方案摘要

| run | selected latency | selection | topology | layout | math | q_chunk | k_chunk | pipe | k_cb | v_cb | dual_noc | active cores | searched | measured |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---:|---:|
| analytical | 0.071 ms | analytical | tree | bandwidth_balanced | LoFi | 32 | 64 | 3 | 3 | 3 | true | 24 | 14688 | 0 |
| calibrated | 0.009 ms | analytical | independent | default | HiFi4 | 32 | 256 | 2 | 2 | 2 | false | 4 | 14688 | 0 |
| rerank | 0.069 ms | analytical | tree | bandwidth_balanced | LoFi | 32 | 128 | 3 | 3 | 3 | true | 24 | 14688 | 0 |
