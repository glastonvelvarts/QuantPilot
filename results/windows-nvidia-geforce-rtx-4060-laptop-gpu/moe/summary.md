| model | flags | QuantPilot plans | prompt tok/s (512) | generate tok/s (128) |
|---|---|---|---|---|
| granite-3.1-1b-a400m-instruct-Q2_K.gguf | -ngl 99 -ncmoe 0 | --n-cpu-moe 0 | 4936.8 ± 220.9 | 144.4 ± 6.9 |
| granite-3.1-1b-a400m-instruct-Q2_K.gguf | -ngl 99 -ncmoe 6 | --n-cpu-moe 0 | 4712.4 ± 248.1 | 123.3 ± 5.9 |
| granite-3.1-1b-a400m-instruct-Q2_K.gguf | -ngl 99 -ncmoe 12 | --n-cpu-moe 0 | 3781.0 ± 477.1 | 100.9 ± 4.2 |
| granite-3.1-1b-a400m-instruct-Q2_K.gguf | -ngl 99 -ncmoe 24 | --n-cpu-moe 0 | 3288.2 ± 122.1 | 88.9 ± 1.0 |
| olmoe-1b-7b-0924-instruct-q2_k.gguf | -ngl 99 -ncmoe 0 | --n-cpu-moe 0 | 2115.5 ± 83.7 | 104.1 ± 4.7 |
| olmoe-1b-7b-0924-instruct-q2_k.gguf | -ngl 99 -ncmoe 4 | --n-cpu-moe 0 | 1573.8 ± 33.9 | 78.1 ± 4.1 |
| olmoe-1b-7b-0924-instruct-q2_k.gguf | -ngl 99 -ncmoe 8 | --n-cpu-moe 0 | 1368.3 ± 41.8 | 71.8 ± 0.8 |
| olmoe-1b-7b-0924-instruct-q2_k.gguf | -ngl 99 -ncmoe 16 | --n-cpu-moe 0 | 1071.2 ± 31.6 | 58.4 ± 3.8 |
| gpt-oss-20b-MXFP4.gguf | -ngl 99 -ncmoe 6 | --n-cpu-moe 6 | 250.1 ± 1.3 | 17.0 ± 0.1 |
| gpt-oss-20b-MXFP4.gguf | -ngl 99 -ncmoe 8 | --n-cpu-moe 6 | 249.1 ± 2.4 | 18.1 ± 0.2 |
| gpt-oss-20b-MXFP4.gguf | -ngl 99 -ncmoe 10 | --n-cpu-moe 6 | 544.9 ± 11.6 | 21.2 ± 0.3 |
| gpt-oss-20b-MXFP4.gguf | -ngl 99 -ncmoe 24 | --n-cpu-moe 6 | 362.1 ± 4.0 | 15.2 ± 1.8 |
| gpt-oss-20b-MXFP4.gguf | -ngl 6 | --n-cpu-moe 6 | 392.7 ± 6.0 | 18.2 ± 1.5 |
| gpt-oss-20b-MXFP4.gguf | -ngl 9 | --n-cpu-moe 6 | 430.8 ± 6.6 | 20.1 ± 1.4 |
| gpt-oss-20b-MXFP4.gguf | -ngl 12 | --n-cpu-moe 6 | 460.1 ± 4.7 | 15.7 ± 1.0 |
| gpt-oss-20b-MXFP4.gguf | -ngl 15 | --n-cpu-moe 6 | 526.3 ± 4.0 | 21.9 ± 0.3 |
| gpt-oss-20b-MXFP4.gguf | -ngl 18 | --n-cpu-moe 6 | 222.3 ± 2.0 | 15.0 ± 0.0 |
| gpt-oss-20b-MXFP4.gguf | -ngl 21 | --n-cpu-moe 6 | 248.6 ± 8.1 | 14.3 ± 0.0 |
