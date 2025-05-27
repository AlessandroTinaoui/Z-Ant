# Z-Ant

<div align="left">
  <img src="https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-tests.yml/badge.svg" alt="Zig Tests" />
  <img src="https://github.com/ZantFoundation/Z-Ant/actions/workflows/zant-benchmarks.yml/badge.svg" alt="Zig Benchamrk Tests" />
  <img src="https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-codegen-tests.yml/badge.svg" alt="Zig Codegen Tests" />
</div>

![image](https://github.com/user-attachments/assets/6a5346e5-58ec-4069-8143-c3b7b03586f3)

# 📖 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Why Z-Ant?](#why-z-ant)
- [Use Cases](#use-cases)
- [Project Status & Achievements](#-project-status--achievements)
- [Roadmap to Best-in-Class TinyML Engine](#-roadmap-to-best-in-class-tinyml-engine)
- [Getting Started for Contributors](#-getting-started-for-contributors)
- [Development Workflow](#️-development-workflow)
- [ODHack14](#odhack14)
- [Using Z-Ant](#-using-z-ant)
- [Build System](#️-build-system-buildzig)
- [Contributing](#-contributing)
- [License](#-license)

---

## Project Overview

**Z-Ant** (Zig-Ant) is a comprehensive, open-source neural network framework specifically designed for deploying optimized AI models on microcontrollers and edge devices. Built with Zig, Z-Ant provides end-to-end tools for model optimization, code generation, and real-time inference on resource-constrained hardware.

## Key Features

### 🚀 **Comprehensive Model Deployment**
- **ONNX Model Support**: Full compatibility with ONNX format models
- **Cross-platform Compilation**: ARM Cortex-M, RISC-V, x86, and more
- **Static Library Generation**: Generate optimized static libraries for any target architecture
- **Real-time Inference**: Microsecond-level prediction times on microcontrollers

### 🛠 **Advanced Optimization Engine**
- **Quantization**: Automatic model quantization with dynamic and static options
- **Pruning**: Neural network pruning for reduced model size
- **Buffer Optimization**: Memory-efficient tensor operations
- **Flash vs RAM Execution**: Configurable execution strategies

### 🖥 **GUI Interface**
Z-Ant includes an experimental cross-platform GUI built with SDL for basic model selection and code generation. Note that the GUI is currently unstable and under active development - we recommend using the command-line interface for production workflows.

### 📷 **ImageToTensor Processing**
- **JPEG Decoding**: Complete JPEG image processing pipeline
- **Multiple Color Spaces**: RGB, YUV, Grayscale support
- **Hardware Optimization**: SIMD and platform-specific optimizations
- **Preprocessing Pipeline**: Normalization, resizing, and format conversion

### 🔧 **Extensive ONNX Support**
- **30+ Operators**: Comprehensive coverage of neural network operations
- **Multiple Data Types**: Float32, Int64, Bool, and more
- **Dynamic Shapes**: Support for variable input dimensions
- **Custom Operators**: Extensible operator framework

## Why Z-Ant?

- **🚫 Lack of DL Support**: Devices like TI Sitara, Raspberry Pi Pico, or ARM Cortex-M lack comprehensive DL libraries
- **🌍 Open-source**: Complete end-to-end NN deployment and optimization solution
- **🎓 Research-Inspired**: Implements cutting-edge optimization techniques inspired by MIT's Han Lab research
- **🏛 Academic Collaboration**: Developed in collaboration with institutions like Politecnico di Milano
- **⚡ Performance First**: Designed for real-time inference with minimal resource usage
- **🔧 Developer Friendly**: Clear APIs, extensive documentation, and practical examples

## Use Cases

- **🏭 Edge AI**: Real-time anomaly detection, predictive maintenance
- **🤖 IoT & Autonomous Systems**: Lightweight AI models for drones, robots, vehicles, IoT devices
- **📱 Mobile Applications**: On-device inference for privacy-preserving AI
- **🏥 Medical Devices**: Real-time health monitoring and diagnostics
- **🎮 Gaming**: AI-powered gameplay enhancement on embedded systems

---

## 🎯 Project Status & Achievements

### ✅ **Completed Features** (Current State - May 2025)

- **📷 im2tensor**: Complete JPEG image processing pipeline with multiple color space support
- **🚀 Enhanced Code Generation**: Advanced code generation with flash vs RAM execution strategies
- **🔧 Expanded ONNX Compatibility**: 30+ operators with comprehensive neural network coverage
- **📊 Shape Tracker**: Dynamic tensor shape management and optimization
- **🧪 Comprehensive Testing Suite**: Automated testing for all major components
- **📚 Static Library Generation**: Cross-platform compilation for ARM Cortex-M, RISC-V, x86

### 🚧 **Work in Progress** (Long-term goals actively being developed)

- **🔬 Advanced Pruning & Quantization**: Research-grade optimization techniques
- **📱 Expanded Microcontroller Support**: Additional hardware platforms
- **⚡ Real-time Benchmarking Tools**: Performance analysis and profiling suite
- **🔄 Model Execution Optimization**: Further inference speed improvements

### 🎯 **Upcoming Milestones**

- **Q4 2025**: Efficient YOLO deployment on edge devices

---

## 🚀 Roadmap to Best-in-Class TinyML Engine

To establish Z-Ant as the premier tinyML inference engine, we are pursuing several key improvements:

### 🔥 **Performance Optimizations**

#### **Ultra-Low Latency Inference**
- **Custom Memory Allocators**: Zero-allocation inference with pre-allocated memory pools
- **In-Place Operations**: Minimize memory copies through tensor operation fusion
- **SIMD Vectorization**: ARM NEON, RISC-V Vector extensions, and x86 AVX optimizations
- **Assembly Kernels**: Hand-optimized assembly for critical operations (matrix multiplication, convolution)
- **Cache-Aware Algorithms**: Memory access patterns optimized for L1/L2 cache efficiency

#### **Advanced Model Optimization**
- **Dynamic Quantization**: Runtime precision adjustment based on input characteristics
- **Structured Pruning**: Channel and block-level pruning for hardware-friendly sparsity
- **Knowledge Distillation**: Automatic teacher-student model compression pipeline
- **Neural Architecture Search (NAS)**: Hardware-aware model architecture optimization
- **Binary/Ternary Networks**: Extreme quantization for ultra-low power inference

### ⚡ **Hardware Acceleration**

#### **Microcontroller-Specific Optimizations**
- **DSP Instruction Utilization**: Leverage ARM Cortex-M DSP instructions and RISC-V packed SIMD
- **DMA-Accelerated Operations**: Offload data movement to DMA controllers
- **Flash Execution Strategies**: XIP (Execute-in-Place) optimization for flash-resident models
- **Low-Power Modes**: Dynamic frequency scaling and sleep mode integration
- **Hardware Security Modules**: Secure model storage and execution

#### **Emerging Hardware Support**
- **NPU Integration**: Support for dedicated neural processing units (e.g., Arm Ethos, Intel Movidius)
- **FPGA Acceleration**: Custom hardware generation for ultra-performance inference
- **GPU Compute**: OpenCL/CUDA kernels for edge GPU acceleration
- **Neuromorphic Computing**: Spike-based neural network execution

### 🧠 **Advanced AI Capabilities**

#### **Model Compression & Acceleration**
- **Lottery Ticket Hypothesis**: Sparse subnetwork discovery and training
- **Progressive Quantization**: Gradual precision reduction during training/deployment
- **Magnitude-Based Pruning**: Automatic weight importance analysis
- **Channel Shuffling**: Network reorganization for efficient inference
- **Tensor Decomposition**: Low-rank approximation for parameter reduction

#### **Adaptive Inference**
- **Early Exit Networks**: Conditional computation based on input complexity
- **Dynamic Model Selection**: Runtime model switching based on resource availability
- **Cascaded Inference**: Multi-stage models with progressive complexity
- **Attention Mechanism Optimization**: Efficient transformer and attention implementations

### 🔧 **Developer Experience & Tooling**

#### **Advanced Profiling & Analysis**
- **Hardware Performance Counters**: Cycle-accurate performance measurement
- **Energy Profiling**: Power consumption analysis per operation
- **Memory Footprint Analysis**: Detailed RAM/Flash usage breakdown
- **Thermal Analysis**: Temperature impact on inference performance
- **Real-Time Visualization**: Live performance monitoring dashboards

#### **Automated Optimization Pipeline**
- **AutoML Integration**: Automated hyperparameter tuning for target hardware
- **Benchmark-Driven Optimization**: Continuous performance regression testing
- **Hardware-in-the-Loop Testing**: Automated testing on real hardware platforms
- **Model Validation**: Accuracy preservation verification throughout optimization
- **Deploy-to-Production Pipeline**: One-click deployment to embedded systems

### 🌐 **Ecosystem & Integration**

#### **Framework Interoperability**
- **TensorFlow Lite Compatibility**: Seamless migration from TFLite models
- **PyTorch Mobile Integration**: Direct PyTorch model deployment pipeline
- **ONNX Runtime Parity**: Feature-complete ONNX runtime alternative
- **MLflow Integration**: Model versioning and experiment tracking
- **Edge Impulse Compatibility**: Integration with popular edge ML platforms

#### **Production Deployment**
- **OTA Model Updates**: Over-the-air model deployment and versioning
- **A/B Testing Framework**: Safe model rollout with performance comparison
- **Federated Learning Support**: Distributed training on edge devices
- **Model Encryption**: Secure model storage and execution
- **Compliance Tools**: GDPR, HIPAA, and safety-critical certifications

### 📊 **Benchmarking & Validation**

#### **Industry-Standard Benchmarks**
- **MLPerf Tiny**: Competitive performance on standard benchmarks
- **EEMBC MLMark**: Energy efficiency measurements
- **Custom TinyML Benchmarks**: Domain-specific performance evaluation
- **Real-World Workload Testing**: Production-representative model validation
- **Cross-Platform Consistency**: Identical results across all supported hardware

#### **Quality Assurance**
- **Fuzzing Infrastructure**: Automated testing with random inputs
- **Formal Verification**: Mathematical proof of correctness for critical operations
- **Hardware Stress Testing**: Extended operation under extreme conditions
- **Regression Test Suite**: Comprehensive backward compatibility testing
- **Performance Monitoring**: Continuous integration with performance tracking

---

## 🚀 Getting Started for Contributors

### Prerequisites

- **Zig Compiler**: Install the latest [Zig compiler](https://ziglang.org/learn/getting-started/)
- **Git**: For version control and collaboration
- **Basic Zig Knowledge**: Improve Zig proficiency via [Ziglings](https://codeberg.org/ziglings/exercises)

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ZIGTinyBook/Z-Ant.git
   cd Z-Ant
   ```

2. **Run tests to verify setup:**
   ```bash
   zig build test --summary all
   ```

3. **Generate code for a model:**
   ```bash
   zig build codegen -Dmodel=mnist-1
   ```

### First Time Contributors

**Start here if you're new to Z-Ant:**

1. **Run existing tests**: Use `zig build test --summary all` to understand the codebase
2. **Try code generation**: Use `zig build codegen -Dmodel=mnist-1` to see the workflow
3. **Read the documentation**: Check `/docs/` folder for detailed guides
4. **Review the Hackathon Guide**: For specific guidance on the rendering and lowering pipeline, refer to the [HackathonGuide.md](./HackathonGuide.md).

### Project Architecture

```
Z-Ant/
├── src/                    # Core source code
│   ├── Core/              # Neural network core functionality
│   ├── CodeGen/           # Code generation engine
│   ├── ImageToTensor/     # Image preprocessing pipeline
│   ├── onnx/              # ONNX model parsing
│   └── Utils/             # Utilities and helpers
├── tests/                 # Comprehensive test suite
├── datasets/              # Sample models and test data
├── generated/             # Generated code output
├── examples/              # Arduino and microcontroller examples
└── docs/                  # Documentation and guides
```

---

## 🛠️ Development Workflow

### Quick Start Commands

```bash
# Run comprehensive tests
zig build test --summary all

# Generate code for a specific model
zig build codegen -Dmodel=mnist-1

# Test generated code
zig build test-codegen -Dmodel=mnist-1

# Compile static library for deployment
zig build lib -Dmodel=mnist-1 -Dtarget=thumb-freestanding -Dcpu=cortex_m33
```

### Git Branching Strategy

We follow a structured branching strategy to ensure code quality and smooth collaboration:

#### Branch Types

- **`main`**: Stable, production-ready code for releases
- **`feature/<feature-name>`**: New features under development
- **`fix/<issue-description>`**: Bug fixes and patches
- **`docs/<documentation-topic>`**: Documentation improvements
- **`test/<test-improvements>`**: Test suite enhancements

#### Best Practices for Contributors

- **Test Before Committing**: Run `zig build test --summary all` before every commit
- **Document Your Code**: Follow Zig's doc-comments standard
- **Small, Focused PRs**: Keep pull requests small and focused on a single feature/fix
- **Use Conventional Commits**: Follow commit message conventions (feat:, fix:, docs:, etc.)

---

## ODHack14

All the Hackaton-related issues are available with the tag **ODHack14**! Join us on telegram for the hackaton [https://t.me/ODHackZant](https://t.me/ODHackZant)


## 🔧 Using Z-Ant

### Development Requirements

- Install the latest [Zig compiler](https://ziglang.org/learn/getting-started/)
- Improve Zig proficiency via [Ziglings](https://codeberg.org/ziglings/exercises)

### Running Tests

Add tests to `build.zig/test_list`.

- **Regular tests:**
  ```bash
  zig build test --summary all
  ```
- **Heavy computational tests:**
  ```bash
  zig build test -Dheavy --summary all
  ```

### Generating Code for Models

```bash
zig build codegen -Dmodel=model_name [-Dlog -Duser_tests=user_tests.json]
```

Generated code will be placed in:

```
generated/model_name/
├── lib_{model_name}.zig
├── test_{model_name}.zig
└── user_tests.json
```

### Testing Generated Models

```bash
zig build test-codegen -Dmodel=model_name
```

### Integrating into Your Project

Build the static library:

```bash
zig build lib -Dmodel=model_name -Dtarget={arch} -Dcpu={cpu}
```

Linking with CMake:

```cmake
target_link_libraries(your_project PUBLIC path/to/libzant.a)
```

### Logging (Optional)

To set a custom log function from your C code:

```c
extern void setLogFunction(void (*log_function)(uint8_t *string));
```

---

## 🏗️ Build System (`build.zig`)

### Available Build Commands

#### **Core Commands**

- **Standard build:**
  ```bash
  zig build                                    # Build all targets
  ```

- **Run unit tests:**
  ```bash
  zig build test --summary all                # Run all unit tests
  ```

- **Code generation:**
  ```bash
  zig build codegen -Dmodel=model_name        # Generate code for specified model
  ```

- **Static library compilation:**
  ```bash
  zig build lib -Dmodel=model_name            # Compile static library for deployment
  ```

#### **Testing Commands**

- **Test generated library:**
  ```bash
  zig build test-generated-lib -Dmodel=model_name    # Test specific generated model library
  ```

- **OneOp model testing:**
  ```bash
  zig build test-codegen-gen                   # Generate oneOperation test models
  zig build test-codegen                       # Test all generated oneOperation models
  ```

- **ONNX parser testing:**
  ```bash
  zig build onnx-parser                        # Test ONNX parser functionality
  ```

#### **Profiling & Performance**

- **Build main executable for profiling:**
  ```bash
  zig build build-main -Dmodel=model_name      # Build profiling target executable
  ```

### Command-Line Options

#### **Target & Architecture Options**
- `-Dtarget=<arch>`: Target architecture (e.g., `thumb-freestanding`, `native`)
- `-Dcpu=<cpu>`: CPU model (e.g., `cortex_m33`, `cortex_m4`)

#### **Model & Path Options**
- `-Dmodel=<name>`: Model name (default: `mnist-8`)
- `-Dmodel_path=<path>`: Custom ONNX model path
- `-Dgenerated_path=<path>`: Output directory for generated code
- `-Doutput_path=<path>`: Output directory for compiled library

#### **Code Generation Options**
- `-Dlog=true|false`: Enable detailed logging during code generation
- `-Duser_tests=<path>`: Specify custom user tests JSON file
- `-Dshape=<shape>`: Input tensor shape
- `-Dtype=<type>`: Input data type (default: `f32`)
- `-Dcomm=true|false`: Generate code with comments
- `-Ddynamic=true|false`: Enable dynamic memory allocation

#### **Testing Options**
- `-Dheavy=true|false`: Run heavy computational tests
- `-Dtest_name=<name>`: Run specific test by name

#### **Debug & Profiling Options**
- `-Dtrace_allocator=true|false`: Use tracing allocator for debugging (default: `true`)
- `-Dallocator=<type>`: Allocator type to use (default: `raw_c_allocator`)

### Common Usage Examples

```bash
# Generate code for MNIST model with logging
zig build codegen -Dmodel=mnist-1 -Dlog=true

# Build static library for ARM Cortex-M33
zig build lib -Dmodel=mnist-1 -Dtarget=thumb-freestanding -Dcpu=cortex_m33

# Test with heavy computational tests enabled
zig build test -Dheavy=true --summary all

# Generate code with custom paths and comments
zig build codegen -Dmodel=custom_model -Dmodel_path=my_models/custom.onnx -Dgenerated_path=output/ -Dcomm=true

# Build library with custom output location
zig build lib -Dmodel=mnist-1 -Doutput_path=/path/to/deployment/

# Run specific test
zig build test -Dtest_name=tensor_math_test

# Build profiling executable for performance analysis
zig build build-main -Dmodel=mnist-1 -Dtarget=native
```

---

## 🤝 Contributing

We welcome contributions from developers of all skill levels! Here's how to get involved:

### Getting Started
1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your work
4. **Make your changes** following our coding standards
5. **Run tests** to ensure everything works
6. **Submit a pull request** for review

### Ways to Contribute
- **🐛 Bug Reports**: Found an issue? Let us know!
- **✨ Feature Requests**: Have an idea? Share it with us!
- **💻 Code Contributions**: Improve the codebase or add new features
- **📚 Documentation**: Help make the project easier to understand
- **🧪 Testing**: Write tests or improve test coverage

### Community Guidelines
- Follow our [Code of Conduct](docs/CODE_OF_CONDUCT.md)
- Check out the [Contributing Guide](docs/CONTRIBUTING.md) for detailed guidelines
- Join discussions on GitHub Issues and Discussions

### Recognition
All contributors are recognized in our [Contributors list](https://github.com/ZIGTinyBook/Z-Ant/contributors). Thank you for helping shape the future of tinyML!

---

## 📄 License

This project is licensed under the [LICENSE](LICENSE) file in the repository.

---

<div align="center">

**Join us in revolutionizing AI on edge devices! 🚀**

[GitHub](https://github.com/ZIGTinyBook/Z-Ant) • [Documentation](docs/) • [Examples](examples/) • [Community](https://github.com/ZIGTinyBook/Z-Ant/discussions)

</div>
