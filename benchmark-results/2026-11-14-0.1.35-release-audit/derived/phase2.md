| benchmark | mode | v0.1.34 median [min..max] | v0.1.35 median [min..max] | 0.1.35 / 0.1.34 | reading |
| --- | --- | --- | --- | --- | --- |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=16} | avgt (us/op) | 38.694 [37.564..39.797] (n=10) | 42.878 [41.299..44.208] (n=10) | 1.108 | slower by 10.8% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=1} | avgt (us/op) | 2.791 [2.753..2.828] (n=10) | 3.141 [3.073..3.202] (n=10) | 1.125 | slower by 12.5% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=2} | avgt (us/op) | 4.375 [4.230..4.841] (n=10) | 5.708 [5.581..5.875] (n=10) | 1.305 | slower by 30.5% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=32} | avgt (us/op) | 77.270 [72.000..82.600] (n=10) | 79.073 [75.067..82.848] (n=10) | 1.023 | slower by 2.3% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=4} | avgt (us/op) | 9.944 [9.905..10.631] (n=10) | 12.192 [11.606..12.381] (n=10) | 1.226 | slower by 22.6% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=8} | avgt (us/op) | 19.256 [18.548..20.646] (n=10) | 25.546 [24.008..26.724] (n=10) | 1.327 | slower by 32.7% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=16} | avgt (us/op) | 152.927 [149.611..155.645] (n=10) | 146.979 [138.520..155.572] (n=10) | 0.961 | faster by 3.9% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=1} | avgt (us/op) | 10.788 [10.424..11.178] (n=10) | 10.162 [10.149..10.416] (n=10) | 0.942 | faster by 5.8% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=2} | avgt (us/op) | 18.944 [18.358..19.505] (n=10) | 20.596 [19.356..21.882] (n=10) | 1.087 | slower by 8.7% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=32} | avgt (us/op) | 280.281 [264.509..294.911] (n=10) | 287.802 [270.395..307.002] (n=10) | 1.027 | slower by 2.7% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=4} | avgt (us/op) | 35.562 [33.446..37.691] (n=10) | 37.177 [35.415..39.369] (n=10) | 1.045 | slower by 4.5% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=8} | avgt (us/op) | 73.533 [68.109..77.801] (n=10) | 74.422 [70.631..78.275] (n=10) | 1.012 | slower by 1.2% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=16} | avgt (us/op) | 219.413 [218.466..220.409] (n=10) | 167.619 [167.279..168.158] (n=10) | 0.764 | faster by 23.6% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=1} | avgt (us/op) | 16.816 [16.281..17.046] (n=10) | 11.083 [10.081..12.190] (n=10) | 0.659 | faster by 34.1% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=2} | avgt (us/op) | 32.599 [32.495..32.773] (n=10) | 17.962 [16.346..19.498] (n=10) | 0.551 | faster by 44.9% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=32} | avgt (us/op) | 470.357 [466.187..474.256] (n=10) | 331.358 [329.832..335.721] (n=10) | 0.704 | faster by 29.6% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=4} | avgt (us/op) | 70.855 [68.973..72.838] (n=10) | 42.276 [42.039..42.493] (n=10) | 0.597 | faster by 40.3% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=8} | avgt (us/op) | 155.658 [142.534..169.203] (n=10) | 89.601 [80.912..97.358] (n=10) | 0.576 | faster by 42.4% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=16} | avgt (us/op) | 715.988 [711.258..721.609] (n=10) | 769.478 [759.250..780.096] (n=10) | 1.075 | slower by 7.5% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=1} | avgt (us/op) | 42.907 [37.904..45.944] (n=10) | 52.290 [48.020..56.714] (n=10) | 1.219 | slower by 21.9% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=2} | avgt (us/op) | 92.055 [91.697..92.340] (n=10) | 93.471 [91.698..96.422] (n=10) | 1.015 | slower by 1.5% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=32} | avgt (us/op) | 2187.297 [2163.554..2208.196] (n=10) | 2118.924 [2088.733..2152.516] (n=10) | 0.969 | faster by 3.1% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=4} | avgt (us/op) | 155.073 [153.520..156.480] (n=10) | 172.825 [161.399..184.629] (n=10) | 1.114 | slower by 11.4% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=8} | avgt (us/op) | 323.571 [322.751..324.790] (n=10) | 308.268 [306.088..313.107] (n=10) | 0.953 | faster by 4.7% |
| VectorDbBenchmark.searchFacadeHnsw{dim=128,n=100000} | avgt (us/op) | 118.398 [117.326..118.654] (n=10) | 117.692 [116.581..118.725] (n=10) | 0.994 | faster by 0.6% |
| VectorDbBenchmark.searchFacadeHnsw{dim=128,n=10000} | avgt (us/op) | 73.691 [72.739..74.175] (n=10) | 73.863 [70.729..77.756] (n=10) | 1.002 | slower by 0.2% |
| VectorDbBenchmark.searchFacadeHnsw{dim=768,n=100000} | avgt (us/op) | 323.950 [320.934..328.418] (n=10) | 266.556 [263.806..269.307] (n=10) | 0.823 | faster by 17.7% |
| VectorDbBenchmark.searchFacadeHnsw{dim=768,n=10000} | avgt (us/op) | 221.610 [207.237..236.024] (n=10) | 235.341 [234.083..237.335] (n=10) | 1.062 | slower by 6.2% |
| VectorDbBenchmark.searchRawHnsw{dim=128,n=100000} | avgt (us/op) | 114.136 [111.868..116.429] (n=10) | 123.321 [122.217..125.075] (n=10) | 1.080 | slower by 8.0% |
| VectorDbBenchmark.searchRawHnsw{dim=128,n=10000} | avgt (us/op) | 70.852 [67.230..74.405] (n=10) | 69.006 [67.026..71.295] (n=10) | 0.974 | faster by 2.6% |
| VectorDbBenchmark.searchRawHnsw{dim=768,n=100000} | avgt (us/op) | 302.278 [278.349..326.028] (n=10) | 301.066 [277.933..318.757] (n=10) | 0.996 | faster by 0.4% |
| VectorDbBenchmark.searchRawHnsw{dim=768,n=10000} | avgt (us/op) | 239.692 [234.984..243.879] (n=10) | 204.358 [202.857..206.232] (n=10) | 0.853 | faster by 14.7% |

