| run | layers on GPU | checks | GPU compute MiB | RAM compute MiB | output MiB |
|---|---|---|---|---|---|
| Falcon-H1-0.5B-Instruct-IQ1_M-16k-q8_0 | 37/37 | 4 ok | 88.08 | 24.05 | 0.13 |
| Falcon-H1-0.5B-Instruct-IQ1_M-16k | 37/37 | 4 ok | 88.05 | 24.02 | 0.13 |
| Falcon-H1-0.5B-Instruct-IQ1_M-4k | 37/37 | 4 ok | 76.05 | 12.02 | 0.13 |
| Falcon-H1-0.5B-Instruct-IQ1_M-64k | 37/37 | 4 ok | 89.33 | 72.02 | 0.13 |
| gemma-2-2b-it-IQ3_M-16k-q8_0 | 27/27 | 3 ok | 539.05 | 39.06 | 0.98 |
| gemma-2-2b-it-IQ3_M-16k | 27/27 | 3 ok | 538.52 | 38.53 | 0.98 |
| gemma-2-2b-it-IQ3_M-4k | 27/27 | 3 ok | 526.02 | 26.03 | 0.98 |
| gemma-2-2b-it-IQ3_M-64k | 27/27 | 3 ok | 586.52 | 86.53 | 0.98 |
| gemma-3-1b-it-Q2_K-16k-q8_0 | 27/27 | 0 ok, **1 differ** | 538.56 | 26.56 | 1.00 |
| gemma-3-1b-it-Q2_K-16k | 27/27 | 0 ok, **1 differ** | 538.03 | 26.03 | 1.00 |
| gemma-3-1b-it-Q2_K-4k | 27/27 | 0 ok, **1 differ** | 526.03 | 14.03 | 1.00 |
| gemma-3-1b-it-Q2_K-64k | 27/27 | 0 ok, **1 differ** | 586.03 | 74.03 | 1.00 |
| gpt-oss-20b-MXFP4-16k-q8_0 | 25/25 | 3 ok | 416.04 | 39.34 | 0.77 |
| gpt-oss-20b-MXFP4-16k | 25/25 | 3 ok | 416.01 | 39.28 | 0.77 |
| gpt-oss-20b-MXFP4-4k | 25/25 | 3 ok | 416.01 | 27.28 | 0.77 |
| gpt-oss-20b-MXFP4-64k | 25/25 | 3 ok | 416.01 | 87.28 | 0.77 |
| granite-3.1-1b-a400m-instruct-Q2_K-16k-q8_0 | - | error: 5.372 D D:/a/llama.cpp/llama.cpp/ggml/src\ggml-impl.h:330: fatal error | 0.00 | 0.00 | 0.19 |
| granite-3.1-1b-a400m-instruct-Q2_K-16k | 25/25 | 3 ok | 120.02 | 24.02 | 0.19 |
| granite-3.1-1b-a400m-instruct-Q2_K-4k | 25/25 | 3 ok | 108.02 | 12.02 | 0.19 |
| granite-3.1-1b-a400m-instruct-Q2_K-64k | 25/25 | 3 ok | 168.02 | 72.02 | 0.19 |
| granite-4.0-h-350m-Q2_K-16k-q8_0 | 33/33 | 4 ok | 218.05 | 22.05 | 0.38 |
| granite-4.0-h-350m-Q2_K-16k | 33/33 | 4 ok | 218.01 | 22.02 | 0.38 |
| granite-4.0-h-350m-Q2_K-4k | 33/33 | 4 ok | 206.01 | 10.02 | 0.38 |
| granite-4.0-h-350m-Q2_K-64k | 33/33 | 4 ok | 266.01 | 70.02 | 0.38 |
| jamba-reasoning-3b-Q4_K_M-16k-q8_0 | 29/29 | 4 ok | 164.09 | 36.10 | 0.25 |
| jamba-reasoning-3b-Q4_K_M-16k | 29/29 | 4 ok | 164.01 | 36.02 | 0.25 |
| jamba-reasoning-3b-Q4_K_M-4k | 29/29 | 4 ok | 177.01 | 24.02 | 0.25 |
| jamba-reasoning-3b-Q4_K_M-64k | 29/29 | 4 ok | 212.01 | 84.02 | 0.25 |
| LFM2-350M-Q4_0-16k-q8_0 | 17/17 | 4 ok | 152.05 | 24.05 | 0.25 |
| LFM2-350M-Q4_0-16k | 17/17 | 4 ok | 152.02 | 24.02 | 0.25 |
| LFM2-350M-Q4_0-4k | 17/17 | 4 ok | 140.02 | 12.02 | 0.25 |
| LFM2-350M-Q4_0-64k | 17/17 | 4 ok | 200.02 | 72.02 | 0.25 |
| olmoe-1b-7b-0924-instruct-q2_k-16k-q8_0 | 17/17 | 3 ok | 164.09 | 32.10 | 0.19 |
| olmoe-1b-7b-0924-instruct-q2_k-16k | 17/17 | 3 ok | 130.27 | 32.02 | 0.19 |
| olmoe-1b-7b-0924-instruct-q2_k-4k | 17/17 | 3 ok | 118.27 | 20.02 | 0.19 |
| olmoe-1b-7b-0924-instruct-q2_k-64k | 11/17 | 1 ok | 600.00 | 80.02 | 0.19 |
