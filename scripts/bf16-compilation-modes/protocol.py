"""Frozen15-block compilation perturbation diagnosis; not performance qualification."""
import math
ARTIFACTS={'published':'21eeae4fcaa56fa26ed45d99610142f794cdec0e1d852894e637263f5bd0b40c','old':'13f0bd516ede8e4306541c72ef368e94c9bddddd80a99631ff5aa65992fdbba8','candidate':'9403f34da7b0713d3e9fb1c4d507a3f11d745fe1a7b3f4620d8ad24be4c82592'}
ARMS=['published-plain','old-plain','candidate-plain','candidate-log','candidate-assembly']
PROTOCOL={'blocks':15,'arms':ARMS,'host':'Neoverse-N2','benchmark':'mappedBfloat16MatVec','shape':'4864x896','order':'cyclicLatinSquare;eacharm occupies eachposition exactly3times','forksPerArm':15,'totalForks':75,'scope':'Causal diagnostic only. Allforks retained; no result exclusions or release claims; oldplain distinguishes artifact drift from instrumentation.'}
