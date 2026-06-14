# A-BH Reader/Writer Coupling Breakdown

| case | compute | reader total | writer total | reader reserve | writer cb_wait | dominant stage | dominant coupling |
|---|---:|---:|---:|---:|---:|---|---|
| decode_1k | 0.0064 ms | 0.0135 ms | 0.0112 ms | 0.0000 ms | 0.0043 ms | reader | writer_cb_wait_from_reader_ms |
| decode_4k | 0.0258 ms | 0.0490 ms | 0.0277 ms | 0.0028 ms | 0.0207 ms | reader | writer_cb_wait_from_reader_ms |
| decode_8k | 0.0516 ms | 0.0922 ms | 0.0491 ms | 0.0145 ms | 0.0420 ms | reader | writer_cb_wait_from_reader_ms |
| decode_16k | 0.1032 ms | 0.1826 ms | 0.0940 ms | 0.0395 ms | 0.0871 ms | reader | writer_cb_wait_from_reader_ms |
| decode_32k | 0.2063 ms | 0.3715 ms | 0.1882 ms | 0.0881 ms | 0.1813 ms | reader | writer_cb_wait_from_reader_ms |
