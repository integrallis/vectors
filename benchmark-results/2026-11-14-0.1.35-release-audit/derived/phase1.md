host: AMD EPYC 7R13 Processor / 2 / 8
java: openjdk version "25.0.5" 2026-10-20 LTS

JMH files: {'v0.1.34': 20, 'v0.1.35': 20}

| benchmark | mode | v0.1.34 median [min..max] | v0.1.35 median [min..max] | 0.1.35 / 0.1.34 | reading |
| --- | --- | --- | --- | --- | --- |
| AnnSearchBenchmark.searchFlat{dim=128,k=10,n=2000} | thrpt (ops/ms) | 21.951 [17.370..22.393] (n=20) | 22.262 [22.156..22.506] (n=20) | 1.014 | faster by 1.4% |
| AnnSearchBenchmark.searchHnsw{dim=128,k=10,n=2000} | thrpt (ops/ms) | 24.250 [23.296..25.570] (n=20) | 23.559 [22.761..23.920] (n=20) | 0.971 | slower by 2.9% |
| AnnSearchBenchmark.searchIvfFlat{dim=128,k=10,n=2000} | thrpt (ops/ms) | 23.096 [22.444..24.505] (n=20) | 23.654 [22.588..25.114] (n=20) | 1.024 | faster by 2.4% |
| AnnSearchBenchmark.searchVamana{dim=128,k=10,n=2000} | thrpt (ops/ms) | 19.145 [18.401..19.410] (n=20) | 21.595 [20.468..22.136] (n=20) | 1.128 | faster by 12.8% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=100,liveSize=100000} | avgt (us/op) | 14.186 [13.917..14.353] (n=20) | 14.095 [13.574..14.692] (n=20) | 0.994 | faster by 0.6% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=100,liveSize=10000} | avgt (us/op) | 10.185 [9.929..10.719] (n=20) | 10.063 [9.766..10.286] (n=20) | 0.988 | faster by 1.2% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=100,liveSize=1000} | avgt (us/op) | 8.263 [8.213..8.446] (n=20) | 8.354 [8.129..8.942] (n=20) | 1.011 | slower by 1.1% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=1000,liveSize=100000} | avgt (us/op) | 142.442 [138.326..148.435] (n=20) | 138.086 [132.857..147.649] (n=20) | 0.969 | faster by 3.1% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=1000,liveSize=10000} | avgt (us/op) | 97.049 [92.056..103.232] (n=20) | 96.114 [94.360..104.767] (n=20) | 0.990 | faster by 1.0% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=1000,liveSize=1000} | avgt (us/op) | 80.208 [77.707..84.011] (n=20) | 80.369 [77.932..83.524] (n=20) | 1.002 | slower by 0.2% |
| GgufBatchedMatmulBenchmark.batched{batchSize=1,cols=2048,rows=1024} | avgt (ms/op) | 0.089 [0.089..0.092] (n=20) | 0.089 [0.089..0.092] (n=20) | 1.001 | slower by 0.1% |
| GgufBatchedMatmulBenchmark.batched{batchSize=2,cols=2048,rows=1024} | avgt (ms/op) | 0.271 [0.216..0.274] (n=20) | 0.247 [0.216..0.275] (n=20) | 0.911 | faster by 8.9% |
| GgufBatchedMatmulBenchmark.batched{batchSize=24,cols=2048,rows=1024} | avgt (ms/op) | 1.525 [1.523..1.632] (n=20) | 1.526 [1.522..1.585] (n=20) | 1.001 | slower by 0.1% |
| GgufBatchedMatmulBenchmark.batched{batchSize=32,cols=2048,rows=1024} | avgt (ms/op) | 2.027 [2.021..2.239] (n=20) | 2.009 [2.002..2.085] (n=20) | 0.991 | faster by 0.9% |
| GgufBatchedMatmulBenchmark.batched{batchSize=4,cols=2048,rows=1024} | avgt (ms/op) | 0.300 [0.299..0.310] (n=20) | 0.300 [0.299..0.321] (n=20) | 1.000 | slower by 0.0% |
| GgufBatchedMatmulBenchmark.batched{batchSize=8,cols=2048,rows=1024} | avgt (ms/op) | 0.546 [0.543..0.584] (n=20) | 0.545 [0.543..0.547] (n=20) | 0.999 | faster by 0.1% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=1,cols=2048,rows=1024} | avgt (ms/op) | 0.107 [0.103..0.122] (n=20) | 0.105 [0.104..0.110] (n=20) | 0.982 | faster by 1.8% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=2,cols=2048,rows=1024} | avgt (ms/op) | 0.240 [0.239..0.251] (n=20) | 0.242 [0.239..0.279] (n=20) | 1.005 | slower by 0.5% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=32,cols=2048,rows=1024} | avgt (ms/op) | 2.417 [2.408..2.576] (n=20) | 2.432 [2.412..2.581] (n=20) | 1.006 | slower by 0.6% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=4,cols=2048,rows=1024} | avgt (ms/op) | 0.307 [0.305..0.317] (n=20) | 0.320 [0.307..0.334] (n=20) | 1.045 | slower by 4.5% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=8,cols=2048,rows=1024} | avgt (ms/op) | 0.614 [0.605..0.626] (n=20) | 0.611 [0.604..0.643] (n=20) | 0.995 | faster by 0.5% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=1,cols=2048,rows=1024} | avgt (ms/op) | 0.104 [0.103..0.114] (n=20) | 0.105 [0.104..0.121] (n=20) | 1.004 | slower by 0.4% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=2,cols=2048,rows=1024} | avgt (ms/op) | 0.208 [0.207..0.219] (n=20) | 0.209 [0.208..0.215] (n=20) | 1.001 | slower by 0.1% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=32,cols=2048,rows=1024} | avgt (ms/op) | 3.425 [3.329..3.868] (n=20) | 3.419 [3.322..3.448] (n=20) | 0.998 | faster by 0.2% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=4,cols=2048,rows=1024} | avgt (ms/op) | 0.418 [0.416..0.436] (n=20) | 0.427 [0.418..0.497] (n=20) | 1.021 | slower by 2.1% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=8,cols=2048,rows=1024} | avgt (ms/op) | 0.846 [0.838..1.303] (n=20) | 0.838 [0.828..0.858] (n=20) | 0.991 | faster by 0.9% |
| IndexBuildBenchmark.buildHnsw{dim=128,n=1000} | ss (ms/op) | 294.501 [287.027..302.203] (n=20) | 275.484 [269.298..294.647] (n=20) | 0.935 | faster by 6.5% |
| IndexBuildBenchmark.buildHnsw{dim=128,n=5000} | ss (ms/op) | 2699.921 [2638.292..2814.628] (n=20) | 2587.821 [2532.574..2627.190] (n=20) | 0.958 | faster by 4.2% |
| IndexBuildBenchmark.buildIvfFlat{dim=128,n=1000} | ss (ms/op) | 5.788 [5.167..6.407] (n=20) | 5.856 [5.261..8.465] (n=20) | 1.012 | slower by 1.2% |
| IndexBuildBenchmark.buildIvfFlat{dim=128,n=5000} | ss (ms/op) | 76.485 [75.210..79.696] (n=20) | 34.721 [33.854..37.400] (n=20) | 0.454 | faster by 54.6% |
| IndexBuildBenchmark.buildVamana{dim=128,n=1000} | ss (ms/op) | 1882.260 [1845.248..1932.869] (n=20) | 1865.335 [1826.777..1910.343] (n=20) | 0.991 | faster by 0.9% |
| IndexBuildBenchmark.buildVamana{dim=128,n=5000} | ss (ms/op) | 14323.668 [14151.466..14779.376] (n=20) | 14250.858 [14081.584..14561.041] (n=20) | 0.995 | faster by 0.5% |

| recall@10 row | committed baseline (2026-06, k-means) | v0.1.34 record | v0.1.35 record | 0.1.35 − 0.1.34 |
| --- | --- | --- | --- | --- |
| HNSW efSearch=128 | 0.9145 | 0.913 | 0.9135 | +0.0005 |
| HNSW efSearch=200 | 0.9635 | 0.9635 | 0.963 | -0.0005 |
| IVF_FLAT nprobe=8,oq=1.0 | 0.3615 | 0.3615 | 0.365 | +0.0035 |
| IVF_PQ nprobe=8,m=16,rescore=4,oq=1.0 | 0.305 | 0.305 | 0.307 | +0.0020 |
| IVF_PQ nprobe=8,m=16,rescore=4,oq=4.0 | 0.716 | 0.716 | 0.7335 | +0.0175 |
| VAMANA L=128 | 0.9705 | 0.9705 | 0.9705 | +0.0000 |
| VAMANA L=200 | 0.983 | 0.983 | 0.983 | +0.0000 |

recallGate verify on v0.1.35 against the committed baseline: PASSED
    RecallGateTest STANDARD_ERROR
    RecallGateTest > recallMatchesBaselineOrRecord() PASSED

SIFT-small recall floors v0.1.34: tests=8 failures=0  ['fullPrecision_beamWidth128_recall10_above95()', 'fullPrecision_beamWidth200_recall10_above97()', 'fullPrecision_beamWidth64_recall10_above90()', 'fullPrecision_ef128_recall10_above95()', 'fullPrecision_ef200_recall10_above98()', 'fullPrecision_ef64_recall10_above90()', 'twoPassSq8_beamWidth128_recall10_above93()', 'twoPassSq8_ef128_recall10_above93()']
SIFT-small recall floors v0.1.35: tests=8 failures=0  ['fullPrecision_beamWidth128_recall10_above95()', 'fullPrecision_beamWidth200_recall10_above97()', 'fullPrecision_beamWidth64_recall10_above90()', 'fullPrecision_ef128_recall10_above95()', 'fullPrecision_ef200_recall10_above98()', 'fullPrecision_ef64_recall10_above90()', 'twoPassSq8_beamWidth128_recall10_above93()', 'twoPassSq8_ef128_recall10_above93()']
