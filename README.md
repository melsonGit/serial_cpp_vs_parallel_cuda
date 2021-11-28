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

With regards to the overall design of the code; being more acclimated to an OOP approach, I found myself steering more towards procedural programming as I became more familiar with CUDA. Perhaps in the future I will look at pushing a more OOP stance on the code, as well as compiling all programs into one single program. As it stands, I believe each program serves the purpose it was made for. 

# Program Design: Vector Addition
WIP

![image](https://user-images.githubusercontent.com/50531920/143775297-518d75d8-155f-4b65-af49-84203804684b.png)
# Program Design: Matrix Multiplication
WIP

![image](https://user-images.githubusercontent.com/50531920/143775312-2c57ea4a-67f1-4767-b885-53c245922dfa.png)
# Program Design: 1-D Convolution
WIP

![image](https://user-images.githubusercontent.com/50531920/143775280-c8060745-af36-4295-9d2b-496ed4adb974.png)
# Program Design: 2-D Convolution
WIP

![image](https://user-images.githubusercontent.com/50531920/143775286-eb21b914-e63e-408a-8fc9-05370b7a645e.png)

