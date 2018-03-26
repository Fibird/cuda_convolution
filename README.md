# Overview

Using cuda C to implement convolution operation.

# Build

```
nvcc -arch=sm_xx gpu_convolution_1D_vx.cu waveformat/waveformat.c -o bin/gpu_vx
```

# Run

```
./bin/gpu_vx audios/moz_noisy.wav
```




