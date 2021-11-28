# serial_cpp_vs_parallel_cuda
Dissertation work for completion of MSc ICT. This implementation, alongside my written thesis, achieved a 77% mark. I achieved a Distinction for my MSc overall.
File raw_data.xlsx details runtime data of both parallel and serial programs.

# Basic Overview

This project focuses on the GPU/GPGPU and CPU components, as well as the CUDA and C++ languages. I wanted to test and compare the computational power of both the GPU/GPGPU and CPU. To do this, I developed 8 programs: 4 programs were developed in traditional C++, and were to be executed solely by the CPU; 4 programs were developed in Nvidia's proprietary GPU/GPGPU CUDA/C++ language, and were to be executed by the GPU/GPGPU. 

The 4 arithmetic operations are: [Vector Addition](https://github.com/melsonGit/serial_cpp_vs_parallel_cuda#program-design-vector-addition), [Matrix Multiplication](https://github.com/melsonGit/serial_cpp_vs_parallel_cuda#program-design-matrix-multiplication), [1-D Convolution](https://github.com/melsonGit/serial_cpp_vs_parallel_cuda#program-design-1-d-convolution) and [2-D Convolution](https://github.com/melsonGit/serial_cpp_vs_parallel_cuda#program-design-2-d-convolution).

# Detailed Overview

I started this project to not only fulfill my MSc, but I also hold a keen interest in the roles of the GPU/GPGPU (graphics processing unit/ general-purpose graphics processing unit) and CPU (central processing unit). I wanted to test the capabilities of both components in numerous environments that were computationally intensive enough to determine where the strengths of GPU/GPGPUs and CPUs lie anecdotally (this area is heavily documented). Combining that with my passion for coding; I went down the rabbit hole that is CUDA (compute unified device architecture). 

My first hurdle was overcoming the technical issues that came with setting up CUDA inside the VS (visual studio) environment. After **3** failed attempts and pain-stakingly configuring my workstation, I was able to get CUDA up and running. With documentation and a few online video tutorials to guide me, I was able to develop **8** mini-programs which tested the GPU/GPGPU and CPU using arithmetic operations (detailed below). 

# Future of this project

With regards to the overall design of the code; being more acclimated to an OOP approach, I found myself steering more towards procedural programming as I became more familiar with CUDA. Perhaps in the future I will look at pushing a more OOP stance on the code, as well as collating all programs into a one single program. As it stands, I believe each program serves the purpose it was made for. In the near-future, I wish to try and refactor most of the code.

# Program Design: Vector Addition
WIP

<p align="center">
  <img src="https://github.com/melsonGit/serial_cpp_vs_parallel_cuda/blob/main/img/add.png" alt="Vector Addition"/>
</p>


# Program Design: Matrix Multiplication
WIP

<p align="center">
  <img src="https://github.com/melsonGit/serial_cpp_vs_parallel_cuda/blob/main/img/multi.png" alt="Matrix Multiplication"/>
</p>


# Program Design: 1-D Convolution
WIP

<p align="center">
  <img src="https://github.com/melsonGit/serial_cpp_vs_parallel_cuda/blob/main/img/1d.png" alt="1-D Convolution"/>
</p>


# Program Design: 2-D Convolution
WIP

<p align="center">
  <img src="https://github.com/melsonGit/serial_cpp_vs_parallel_cuda/blob/main/img/2d.png" alt="2-D Convolution"/>
</p>

