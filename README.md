# Titled Matrix Multiply CUDA - with GP-GPU Sim

Simple overview of use/purpose.

## Description

- The objective of this assignment is to implement a tiled matrix multiplication kernel that can support arbitrary sized matrices
  - kernel.cu and main.cu complete the functionality of the matrix multiplication on the GPU.
  - Given matrices could be any size,
  - testing matrix size will not exceed 65,536 elements (for example, 256 x 256 input matrices).
  - This is purely a limitation for testing in a timely manner, however, code can handle larger sizes
- Main.cu supports 3 operations
  - No arguments: The application will create two randomly initialized matrices to multiply size (1000x1000). After the device multiplication is invoked, it will compute the correct solution matrix using the CPU, and compare that solution with the device-computed solution. If it matches (within a certain tolerance), if will print out "Test PASSED" to the screen before exiting.
  - One argument: The application will use the random initialization to create the input matrices (size mxm, where m is the argument. Start your testing with small matrices
  - Three arguments m, k, and n: The application will initialize the two input matrices with random values. A matrix will be of size m x k while the B matrix will be of size k x n, producing a C matrix of size m x n
- Utilizes Docker + GpGPUSIM:
  - We will analyze memory behavior of tiled matrix multiplication using GPGPU-Sim.
  - To aid in analyzing the microarchitectural properties of these programs, it may help to save the output of the GPGPU-Sim run into a file.
  - output can be saved by redirecting the printouts to a file using ./sgemm-tiled &> outfile.
  - The gpusim will be containerized via Docker image from UCR
  - Ther focus will surround the following metrics from GPUSIM:
    - gpgpu_n_load_insn  -- Number of global/local load instructions executed.
    - gpgpu_n_store_insn -- Number of global/local store instructions executed.
    - gpgpu_n_shmem_insn -- Number of shared memory instructions executed. 


### Dependencies

* Docker, CUDA C, GPGPUSIM

### Installing

```
* After you use docker run -w /root -it socalucr/gpgpu-sim /bin/bash you will be in the terminal session of the docker image
You only need to use docker run once to create your container image. Subsequent calls would result in multiple GPGPU-Sim images on your system.
* Run the following to compile GPGPU-sim
* cd ~/gpgpu-sim_distribution
* source setup_environment (You only need to do this once per terminal session)
* make clean
* make
```

### Executing program

* How to run the program
```
- We will make a test directory in your home mkdir ~/test
- Go into the test directory. cd ~/test
- Copy configuration files to test directory: cp -a ~/gpgpu-sim_distribution/configs/GTX480/* ~/test/
- Now run a binary. For example, we can run the RAY binary in ISPASS2009. ~/ispass2009-benchmarks/bin/release/RAY 4 4
- Note: If ~/ispass2009-benchmarks/bin/release/RAY does not exist, you will need to compile it first by doing the following: cd ~/ispass2009-benchmarks/RAY/;make clean;make;
```
