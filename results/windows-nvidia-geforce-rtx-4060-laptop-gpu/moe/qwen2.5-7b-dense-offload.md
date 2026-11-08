| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  29 |           pp512 |     1093.21 ± 107.47 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  29 |           tg128 |         44.03 ± 2.19 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  14 |           pp512 |       629.28 ± 11.52 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  14 |           tg128 |         13.72 ± 0.46 |

build: d81235049 (11429)
