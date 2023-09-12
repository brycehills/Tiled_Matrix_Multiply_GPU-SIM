# Titled Matrix Multiply CUDA - with GPGPU Sim + Docker

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

### Conceptual Understanding/Answer the following questions:

* On Bender, compare the execution time of a 256 x 256 square matrix multiplication compared to a 1024 x 64 and 64 x 1024 rectangular matrix multiply. All input matricies have 65k entries. What do you observe? Which is faster? you explain the observed behavior? Tip: You may want to comment out the verify() function in main.cu when timing this question.
* Conceptual Question: For a 64 square tiled matrix multiplication, how many times is each element of the input matrices loaded from global memory? Assume 16x16 tiles.  
* Conceptual Question: For a 64 square non-tiled matrix multiplication, how many times is each element of the input matrices loaded from global memory?
* GPGPU-Sim related question: In this part, we will compare the execution of a 128x128 square tiled matrix multiplication across different tile sizes. Run ./sgemm-tiled 128 in GPGPU-Sim with TILE_SIZE of 8, 16 (default), and 32. Fill the following table:Anaylsis:
* Which tile size resulted in the least number of accesses to global memory? Which tile size resulted in the most number of accesses to global memory? What is the reasoning behind this observation?
* Which tile size performed the fastest, which tile size performed the slowest? Why do you think that is?

<img src="Bryce-Hills.pdf" alt="alt text" width="700" height="500">  

