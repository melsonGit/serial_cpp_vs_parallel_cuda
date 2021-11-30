# serial_cpp_vs_parallel_cuda
Dissertation work for completion of MSc ICT. This implementation, alongside my written thesis, achieved a 77% mark. I achieved a Distinction for my MSc overall.
File raw_data.xlsx details runtime data of both parallel and serial programs.

# Basic Overview

This project focuses on the GPU/GPGPU and CPU components, as well as the CUDA and C++ languages. I wanted to test and compare the computational power of both the GPU/GPGPU and CPU. To do this, I developed 8 programs: 4 programs were developed in traditional C++, and were to be executed solely by the CPU; 4 programs were developed in Nvidia's proprietary GPU/GPGPU CUDA/C++ language, and were to be executed by the GPU/GPGPU. 

The 4 arithmetic operations are: [Vector Addition](https://github.com/melsonGit/serial_cpp_vs_parallel_cuda#program-design-vector-addition), [Matrix Multiplication](https://github.com/melsonGit/serial_cpp_vs_parallel_cuda#program-design-matrix-multiplication), [1-D Convolution](https://github.com/melsonGit/serial_cpp_vs_parallel_cuda#program-design-1-d-convolution) and [2-D Convolution](https://github.com/melsonGit/serial_cpp_vs_parallel_cuda#program-design-2-d-convolution).

# Detailed Overview

I started this project to not only fulfill my MSc, but I also hold a keen interest in the roles of the GPU/GPGPU and CPU. I wanted to test the capabilities of both components in numerous environments that were computationally intensive enough to determine where the strengths of GPU/GPGPUs and CPUs lie anecdotally (this area is heavily documented). Combining that with my passion for coding; I went down the rabbit hole that is CUDA (compute unified device architecture; CUDA is Nvidia's propriety GPU programming language).  

My first hurdle was overcoming the technical issues that came with setting up CUDA inside the VS IDE. After **3** failed attempts and pain-stakingly configuring my workstation, I was able to get CUDA up and running. With documentation and a few online video tutorials to guide me, I was able to develop **8** mini-programs which tested the GPU/GPGPU and CPU using arithmetic operations (detailed below).

# Future of this project

With regards to the overall design of the code; being more acclimated to an OOP approach, I found myself steering more towards procedural programming as I became more familiar with CUDA. Perhaps in the future I will look at pushing a more OOP stance on the code, as well as collating all programs into a one single program. As it stands, I believe each program serves the purpose it was made for. In the near-future, I wish to try and refactor most of the code.

# Program Logic: Vector Addition

The below image displays the vector addition logic. This encompasses an addition operation between two populated input vectors into a vector sum. The resulting value is then placed into an output vector. Vector addition was selected as it demonstrates raw CPU and GPU processing speeds in a computationally competent operation while undertaking a substantial sample size. Additionally, the vector addition program implementation was used as an instrument in the initial attempt of homogenising serialising parallel code, thus creating a blueprint for forthcoming program code serialisation.

<p align="center">
  <img src="https://github.com/melsonGit/serial_cpp_vs_parallel_cuda/blob/main/img/add.png" alt="Vector Addition"/>
</p>

# Program Logic: Matrix Multiplication

Matrix multiplication involves a resultant square matrix through multiplication and summation of two squared arrays or vector matrices. As displayed below, the two matrices, matrix A and matrix B, multiply corresponding first elements and proceeding second elements. The summation of these multiplication operations are inputted into matrix C, whereby the loop iterates continuously following this pattern until all elements of matrix A and matrix B are processed. 

<p align="center">
  <img src="https://github.com/melsonGit/serial_cpp_vs_parallel_cuda/blob/main/img/multi.png" alt="Matrix Multiplication"/>
    <p align="center"><i>X denotes a subsequent matrix element</i></p>
</p>

Matrix multiplication is a commonly used operation in computer science and software development, specifically in image and digital sound. It incorporates two fundamental components of mathematical operations: matrices and linear algebra. Additionally, it encompasses core components of ML and AI through simultaneous iterative operations of element multiplication and addition, which in turn highlights certain CPU and GPU capabilities during mass and substantial calculations. Ultimately, the operation was an important inclusion for the project.

# Program Logic: 1-D Convolution

Convolution is an arithmetic operation commonly used in image filtering software to implement effects such as blur and sharpen. With numerous convolution algorithms existing which serve alternate image processing needs, the 1-D convolution operation used in the present project involves overlapping vectors; the first two vectors are input and output vectors, followed by a mask vector. Starting with the first value of the input vector, each value in the mask vector will multiply with the indexed adjacent vector input. The resulting multiplications are then summed together and inputted into the output vector. The mask vector then shifts up an index, and the process continues.

<p align="center">
  <img src="https://github.com/melsonGit/serial_cpp_vs_parallel_cuda/blob/main/img/1d.png" alt="1-D Convolution"/>
</p>

The above image displays this process; values 1, 4, 6 and 3 populate the input vector and values 3, 2 and 1 populate the mask vector. Each corresponding element of input and mask vectors will multiply (represented by orange arrows; e.g. 3x1, 2x4, 1x6). Summation of these multiplications then populate the output vector (for example 3 + 8 + 6 = 17). This operation iterates across the entirety of the input vector until all values are processed.


# Program Logic: 2-D Convolution

Presented below, 2-D convolution follows an identical operation (of 1-D convolution) albeit with a 2-D mask vector, increasing value population twofold through rows and columns. Once more, each corresponding element of input and mask vectors will multiply (represented by orange arrows; e.g. 3x1, 3x1, 2x4, 2x4, 1x6, 1x6.) and summation of this operation is pushed to the outpour vector (e.g. 3 + 3 + 8 + 8 + 6 + 6 = 34).

<p align="center">
  <img src="https://github.com/melsonGit/serial_cpp_vs_parallel_cuda/blob/main/img/2d.png" alt="2-D Convolution"/>
</p>

Previous literature highlights that convolution is an arithmetic operation that CPUs excel at executing, inherently a result of greater cache volumes than GPUs. However, 2-D convolutions (or any multi-dimensional convolution) introduces more parallel aspects due a larger mask vector, thus incurring non-sequential data access. This additional dimension favours GPUs over CPUs. As a result, this justified the inclusion of both 1-D and 2-D convolution arithmetic operations for the present project.

<p align="center">
    <p align="center"><i>Above images sourced from thesis.</i></p>
    <p align="center"><i>Credit to Nick (CoffeeBeforeArch) for creating tutorials that provided the foundation of this project.</i></p>
</p>
