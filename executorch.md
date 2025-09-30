---
layout: default
title: ExecuTorch
permalink: /executorch/
background-class: executorch-background
body-class: executorch-page
---

<div class="jumbotron jumbotron-fluid">
  <div class="container">
    <h1>
      <img src="{{ site.baseurl }}/assets/images/executorch-chip-logo.svg" alt="ExecuTorch" class="executorch-logo">
      ExecuTorch
    </h1>
    <p class="lead">Deploy PyTorch models directly to edge devices. Text, vision, and audio AI with privacy-preserving, real-time inference—no cloud required.</p>

    <div class="executorch-stats">
      <div class="executorch-stat-card">
        <div class="stat-number">12+</div>
        <div class="stat-label">hardware backends supported</div>
      </div>
      <div class="executorch-stat-card">
        <div class="stat-number">Billions</div>
        <div class="stat-label">users in production at Meta</div>
      </div>
      <div class="executorch-stat-card">
        <div class="stat-number">50KB</div>
        <div class="stat-label">base runtime footprint</div>
      </div>
    </div>

    <div class="executorch-cta-buttons">
      <a href="https://pytorch.org/executorch/stable/getting-started.html" class="btn btn-lg btn-orange">Get Started</a>
      <a href="https://github.com/pytorch/executorch" class="btn btn-lg btn-white">View on GitHub</a>
    </div>
  </div>
</div>

<div class="main-content-wrapper">
  <div class="main-content">
    <div class="container">

<div class="executorch-section">
  <h2 class="section-title">Why On-Device AI <span class="executorch-highlight">Matters</span></h2>
  <p class="section-subtitle">The future of AI is at the edge, where privacy meets performance</p>

  <div class="row">
    <div class="col-md-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">🔒</div>
          <h3 class="card-title">Enhanced Privacy</h3>
          <p class="card-text">Data never leaves the device. Process personal content, conversations, and media locally without cloud exposure.</p>
        </div>
      </div>
    </div>

    <div class="col-md-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">⚡</div>
          <h3 class="card-title">Real-Time Response</h3>
          <p class="card-text">Instant inference with no network round-trips. Perfect for AR/VR experiences, multimodal AI interactions, and responsive conversational agents.</p>
        </div>
      </div>
    </div>

    <div class="col-md-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">🌐</div>
          <h3 class="card-title">Offline & Low-Bandwidth Ready</h3>
          <p class="card-text">Zero network dependency for inference. Works seamlessly in low-bandwidth regions, remote areas, or completely offline.</p>
        </div>
      </div>
    </div>

    <div class="col-md-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">💰</div>
          <h3 class="card-title">Cost Efficient</h3>
          <p class="card-text">No cloud compute bills. No API rate limits. Scale to billions of users without infrastructure costs growing linearly.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="executorch-section alt-background">
  <h2 class="section-title">Models Are Getting <span class="executorch-highlight">Smaller & Smarter</span></h2>
  <p class="section-subtitle">The convergence of efficient architectures and edge hardware creates new opportunities</p>

  <div class="row text-center">
    <div class="col-md-4 mb-3">
      <div class="feature-box">
        <h4>Dramatically Smaller</h4>
        <p>Modern LLMs achieve high quality at a fraction of historical sizes</p>
      </div>
    </div>
    <div class="col-md-4 mb-3">
      <div class="feature-box">
        <h4>Edge-Ready Performance</h4>
        <p>Real-time inference on consumer smartphones</p>
      </div>
    </div>
    <div class="col-md-4 mb-3">
      <div class="feature-box">
        <h4>Quantization Benefits</h4>
        <p>Significant size reduction while preserving accuracy</p>
      </div>
    </div>
  </div>

  <p class="text-center lead mt-4">
    <strong>The opportunity is now:</strong> Foundation models have crossed the efficiency threshold. Deploy sophisticated AI directly where data lives.
  </p>
</div>

<div class="executorch-section">
  <h2 class="section-title">Why On-Device AI Was <span class="executorch-highlight">Hard</span></h2>
  <p class="section-subtitle">The technical challenges that made edge deployment complex... until now</p>

  <div class="row">
    <div class="col-md-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">🔋</div>
          <h3 class="card-title">Power Constraints</h3>
          <p class="card-text">From battery-powered phones to energy-harvesting sensors, edge devices have strict power budgets. Microcontrollers may run on milliwatts, requiring extreme efficiency.</p>
        </div>
      </div>
    </div>

    <div class="col-md-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">🌡️</div>
          <h3 class="card-title">Thermal Management</h3>
          <p class="card-text">Sustained inference generates heat without active cooling. From smartphones to industrial IoT devices, thermal throttling limits continuous AI workloads.</p>
        </div>
      </div>
    </div>

    <div class="col-md-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">💾</div>
          <h3 class="card-title">Memory Limitations</h3>
          <p class="card-text">Edge devices range from high-end phones to tiny microcontrollers. Beyond capacity, limited memory bandwidth creates bottlenecks when moving tensors between compute units.</p>
        </div>
      </div>
    </div>

    <div class="col-md-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">🔧</div>
          <h3 class="card-title">Hardware Heterogeneity</h3>
          <p class="card-text">From microcontrollers to smartphone NPUs to embedded GPUs. Each architecture demands unique optimizations, making broad deployment across diverse form factors extremely challenging.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="executorch-section alt-background">
  <h2 class="section-title">PyTorch Powers <span class="executorch-highlight">92%</span> of AI Research</h2>
  <p class="section-subtitle">But deploying PyTorch models to edge devices meant losing everything that made PyTorch great</p>

  <div class="row align-items-center mb-4">
    <div class="col-md-5">
      <div class="card border-success h-100">
        <div class="card-body">
          <h3 class="mb-3">Research & Training</h3>
          <p class="mb-0">PyTorch's intuitive APIs and eager execution power breakthrough research</p>
        </div>
      </div>
    </div>

    <div class="col-md-2 text-center">
      <div class="arrow">→</div>
    </div>

    <div class="col-md-5">
      <div class="card border-danger h-100">
        <div class="card-body">
          <h3 class="mb-3">The Conversion Nightmare</h3>
          <p class="mb-0">Multiple intermediate formats, custom runtimes, C++ rewrites</p>
        </div>
      </div>
    </div>
  </div>

  <div class="card my-5">
    <div class="card-body">
      <h3 class="text-center mb-4">The Hidden Costs of Conversion (Status Quo)</h3>
      <div class="row">
        <div class="col-md-6 mb-3">
          <div class="d-flex align-items-start">
            <span class="issue-icon me-2">❌</span>
            <div>
              <strong class="d-block">Lost Semantics</strong>
              <p class="mb-0">PyTorch operations don't map 1:1 to other formats</p>
            </div>
          </div>
        </div>
        <div class="col-md-6 mb-3">
          <div class="d-flex align-items-start">
            <span class="issue-icon me-2">❌</span>
            <div>
              <strong class="d-block">Debugging Nightmare</strong>
              <p class="mb-0">Can't trace errors back to original PyTorch code</p>
            </div>
          </div>
        </div>
        <div class="col-md-6 mb-3">
          <div class="d-flex align-items-start">
            <span class="issue-icon me-2">❌</span>
            <div>
              <strong class="d-block">Vendor-Specific Formats</strong>
              <p class="mb-0">Locked into proprietary formats with limited operator support</p>
            </div>
          </div>
        </div>
        <div class="col-md-6 mb-3">
          <div class="d-flex align-items-start">
            <span class="issue-icon me-2">❌</span>
            <div>
              <strong class="d-block">Language Barriers</strong>
              <p class="mb-0">Teams spend months rewriting Python models in C++ for production</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="executorch-section">
  <h2 class="section-title">ExecuTorch<br><span class="executorch-highlight small">PyTorch's On-Device AI Framework</span></h2>

  <div class="row">
    <div class="col-md-4 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">🎯</div>
          <h3 class="card-title">No Conversions</h3>
          <p class="card-text">Direct export from PyTorch to edge. Core ATen operators preserved. No intermediate formats, no vendor lock-in.</p>
        </div>
      </div>
    </div>

    <div class="col-md-4 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">⚙️</div>
          <h3 class="card-title">Ahead-of-Time Compilation</h3>
          <p class="card-text">Optimize models offline for target device capabilities. Hardware-specific performance tuning before deployment.</p>
        </div>
      </div>
    </div>

    <div class="col-md-4 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">🔧</div>
          <h3 class="card-title">Modular by Design</h3>
          <p class="card-text">Pick and choose optimization steps. Composable at both compile-time and runtime for maximum flexibility.</p>
        </div>
      </div>
    </div>

    <div class="col-md-4 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">🚀</div>
          <h3 class="card-title">Hardware Ecosystem</h3>
          <p class="card-text">Fully open source with hardware partner contributions. Built on PyTorch's standardized IR and operator set.</p>
        </div>
      </div>
    </div>

    <div class="col-md-4 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">💾</div>
          <h3 class="card-title">Embedded-Friendly Runtime</h3>
          <p class="card-text">Portable C++ runtime runs on microcontrollers to smartphones.</p>
        </div>
      </div>
    </div>

    <div class="col-md-4 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="card-icon">🔗</div>
          <h3 class="card-title">PyTorch Ecosystem</h3>
          <p class="card-text">Native integration with PyTorch ecosystem, including torchao for quantization. Stay in familiar tools throughout.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="executorch-section alt-background">
  <h2 class="section-title">Simple as <span class="executorch-highlight">1-2-3</span></h2>
  <p class="section-subtitle">Export, optimize, and run PyTorch models on edge devices with just a few lines of code</p>

  <div class="executorch-code-section">
    <div class="code-step">
      <h3 class="code-step-title">1. Export Your PyTorch Model</h3>
      <pre><code class="language-python">import torch
from torch.export import export

# Your existing PyTorch model
model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# Creates semantically equivalent graph representation
exported_program = export(model, example_inputs)</code></pre>
    </div>

    <div class="code-step">
      <h3 class="code-step-title">2. Optimize for Target Hardware</h3>
      <p class="code-instruction">Switch between backends with a single line change</p>

      <div class="backend-switcher">
        <div class="backend-tabs">
          <button class="backend-tab active" onclick="switchBackend('cpu')">
            <div class="tab-title">CPU Optimization</div>
            <div class="tab-desc">XNNPACK with KleidiAI</div>
          </button>
          <button class="backend-tab" onclick="switchBackend('apple')">
            <div class="tab-title">Apple Devices</div>
            <div class="tab-desc">Core ML partitioner</div>
          </button>
          <button class="backend-tab" onclick="switchBackend('qualcomm')">
            <div class="tab-title">Qualcomm Chips</div>
            <div class="tab-desc">Hexagon NPU support</div>
          </button>
          <button class="backend-tab more-backends">
            <div class="tab-title">+ 9 More</div>
            <div class="tab-desc">Vulkan, MediaTek, Samsung...</div>
          </button>
        </div>

        <div id="cpu" class="backend-content active">
          <pre><code class="language-python">from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack import XnnpackPartitioner

program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]
).to_executorch()</code></pre>
        </div>

        <div id="apple" class="backend-content">
          <pre><code class="language-python">from executorch.exir import to_edge_transform_and_lower
from executorch.backends.apple import CoreMLPartitioner

program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[CoreMLPartitioner()]
).to_executorch()</code></pre>
        </div>

        <div id="qualcomm" class="backend-content">
          <pre><code class="language-python">from executorch.exir import to_edge_transform_and_lower
from executorch.backends.qualcomm import QnnPartitioner

program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[QnnPartitioner()]
).to_executorch()</code></pre>
        </div>
      </div>

      <pre><code class="language-python"># Save to .pte file
with open("model.pte", "wb") as f:
    f.write(program.buffer)</code></pre>
    </div>

    <div class="code-step">
      <h3 class="code-step-title">3. Run on Any Platform</h3>

      <div class="platform-switcher">
        <div class="platform-tabs">
          <button class="platform-tab active" onclick="switchPlatform('cpp')">C++</button>
          <button class="platform-tab" onclick="switchPlatform('swift')">Swift</button>
          <button class="platform-tab" onclick="switchPlatform('kotlin')">Kotlin</button>
          <button class="platform-tab" onclick="switchPlatform('objc')">Objective-C</button>
          <button class="platform-tab" onclick="switchPlatform('wasm')">WebAssembly</button>
        </div>

        <div id="cpp" class="platform-content active">
          <pre><code class="language-cpp">// Load and execute model
auto module = Module("model.pte");
auto method = module.load_method("forward");
auto outputs = method.execute({input_tensor});
// Access result tensors
auto result = outputs[0].toTensor();</code></pre>
        </div>

        <div id="swift" class="platform-content">
          <pre><code class="language-swift">// Initialize ExecuTorch module
let module = try ETModule(path: "model.pte")
// Run inference with tensors
let outputs = try module.forward([inputTensor])
// Process results
let result = outputs[0]</code></pre>
        </div>

        <div id="kotlin" class="platform-content">
          <pre><code class="language-kotlin">// Load model from assets
val module = Module.load(assetFilePath("model.pte"))
// Execute with tensor input
val outputs = module.forward(inputTensor)
// Extract prediction results
val prediction = outputs[0].dataAsFloatArray</code></pre>
        </div>

        <div id="objc" class="platform-content">
          <pre><code class="language-objc">// Initialize ExecuTorch module
ETModule *module = [[ETModule alloc] initWithPath:@"model.pte" error:nil];
// Run inference with tensors
NSArray *outputs = [module forwardWithInputs:@[inputTensor] error:nil];
// Process results
ETTensor *result = outputs[0];</code></pre>
        </div>

        <div id="wasm" class="platform-content">
          <pre><code class="language-javascript">// Load model from ArrayBuffer
const module = et.Module.load(buffer);
// Create input tensor from data
const inputTensor = et.Tensor.fromIter(tensorData, shape);
// Run inference
const output = module.forward(inputTensor);</code></pre>
        </div>
      </div>

      <p class="text-center mt-3 small text-muted">
        Available on Android, iOS, Linux, Windows, macOS, and embedded microcontrollers (e.g., DSP and Cortex-M processors)
      </p>
    </div>

    <div class="text-center mt-5">
      <p class="small mb-3 font-italic text-muted">
        Need advanced features? ExecuTorch supports memory planning, quantization, profiling, and custom compiler passes.
      </p>
      <a href="https://pytorch.org/executorch/stable/getting-started.html" class="btn btn-lg btn-orange">
        Try the Full Tutorial →
      </a>
    </div>
  </div>
</div>

<div class="executorch-section alt-background">
  <h2 class="section-title">High-Level <span class="executorch-highlight">Multimodal APIs</span></h2>
  <p class="section-subtitle">Run complex multimodal LLMs with simplified C++ interfaces</p>

  <div class="executorch-code-section">
    <div class="code-step">
      <h3 class="code-step-title">Multimodal Runner - Text + Vision + Audio in One API</h3>
      <p class="text-center text-muted mb-3 small">Choose your platform to see the multimodal API supporting text, images, and audio:</p>

      <div class="platform-switcher">
        <div class="platform-tabs">
          <button class="platform-tab active" onclick="switchMultimodalPlatform('cpp-mm')">C++<br><small>Cross-platform</small></button>
          <button class="platform-tab" onclick="switchMultimodalPlatform('swift-mm')">Swift<br><small>iOS native</small></button>
          <button class="platform-tab" onclick="switchMultimodalPlatform('kotlin-mm')">Kotlin<br><small>Android native</small></button>
        </div>

        <div id="cpp-mm" class="platform-content active">
          <pre><code class="language-cpp">#include "executorch/extension/llm/runner/multimodal_runner.h"

// Initialize multimodal model (e.g., Voxtral, LLaVA)
auto runner = MultimodalRunner::create(
    "model.pte",     // Text model
    "vision.pte",   // Vision encoder
    "audio.pte",    // Audio encoder
    tokenizer_path,
    temperature
);

// Run inference with audio + image + text
auto result = runner->generate_multimodal(
    "Describe what you hear and see",
    audio_tensor,   // Audio input
    image_tensor,   // Image input
    max_tokens
);

// Stream response tokens
for (const auto& token : result.tokens) {
    std::cout << token << std::flush;
}</code></pre>
        </div>

        <div id="swift-mm" class="platform-content">
          <pre><code class="language-swift">import ExecuTorch
import AVFoundation

// Initialize multimodal runner with audio support
let runner = try MultimodalRunner(
    modelPath: "model.pte",
    visionPath: "vision.pte",
    audioPath: "audio.pte",
    tokenizerPath: tokenizerPath,
    temperature: 0.7
)

// Process audio and image inputs
let audioTensor = AudioProcessor.preprocess(audioURL)
let imageTensor = ImageProcessor.preprocess(uiImage)

// Generate with audio + vision + text
let result = try runner.generateMultimodal(
    prompt: "Describe what you hear and see",
    audio: audioTensor,
    image: imageTensor,
    maxTokens: 512
)

// Stream tokens to UI
result.tokens.forEach { token in
    DispatchQueue.main.async {
        responseText += token
    }
}</code></pre>
        </div>

        <div id="kotlin-mm" class="platform-content">
          <pre><code class="language-kotlin">import org.pytorch.executorch.MultimodalRunner
import android.media.MediaRecorder

// Initialize multimodal runner with audio
val runner = MultimodalRunner.create(
    modelPath = "model.pte",
    visionPath = "vision.pte",
    audioPath = "audio.pte",
    tokenizerPath = tokenizerPath,
    temperature = 0.7f
)

// Process audio and image inputs
val audioTensor = AudioProcessor.preprocess(audioFile)
val imageTensor = ImageProcessor.preprocess(bitmap)

// Generate with audio + vision + text
val result = runner.generateMultimodal(
    prompt = "Describe what you hear and see",
    audio = audioTensor,
    image = imageTensor,
    maxTokens = 512
)

// Display streaming response
result.tokens.forEach { token ->
    runOnUiThread {
        responseView.append(token)
    }
}</code></pre>
        </div>
      </div>

      <p class="text-center mt-4 small text-muted font-italic">
        High-level APIs abstract away model complexity - just load, prompt, and get results
      </p>
      <div class="text-center mt-3">
        <a href="https://pytorch.org/executorch/main/llm/getting-started.html" class="btn btn-lg btn-orange">
          Explore LLM APIs →
        </a>
      </div>
    </div>
  </div>
</div>

<div class="executorch-section alt-background">
  <div class="container">
    <h2 class="section-title">Universal <span class="executorch-highlight">AI Runtime</span></h2>
    <div style="text-align: center; margin: 2rem 0;">
      <div class="domain-slider">
        <div class="domain-track">
          <span>💬 LLMs</span>
          <span>👁️ Computer Vision</span>
          <span>🎤 Speech AI</span>
          <span>🎯 Recommendations</span>
          <span>🧠 Multimodal</span>
          <span>⚡ Any PyTorch Model</span>
          <!-- Duplicate for seamless loop -->
          <span>💬 LLMs</span>
          <span>👁️ Computer Vision</span>
          <span>🎤 Speech AI</span>
          <span>🎯 Recommendations</span>
          <span>🧠 Multimodal</span>
          <span>⚡ Any PyTorch Model</span>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="executorch-section">
  <h2 class="section-title">Comprehensive Hardware <span class="executorch-highlight">Ecosystem</span></h2>
  <p class="section-subtitle">Hardware acceleration contributed by industry partners via open source</p>

  <div class="row">
    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">XNNPACK with KleidiAI</h3>
          <p class="card-text">CPU acceleration across ARM and x86 architectures</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">Apple Core ML</h3>
          <p class="card-text">Neural Engine and Apple Silicon optimization</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">Qualcomm Snapdragon</h3>
          <p class="card-text">Hexagon NPU support</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">ARM Ethos-U</h3>
          <p class="card-text">Microcontroller NPU for ultra-low power</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">Vulkan GPU</h3>
          <p class="card-text">Cross-platform graphics acceleration</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">Intel OpenVINO</h3>
          <p class="card-text">x86 CPU and integrated GPU optimization</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">MediaTek NPU</h3>
          <p class="card-text">Dimensity chipset acceleration</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">Samsung Exynos</h3>
          <p class="card-text">Integrated NPU optimization</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">NXP Neutron</h3>
          <p class="card-text">Automotive and IoT acceleration</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">Apple MPS</h3>
          <p class="card-text">Metal Performance Shaders for GPU acceleration</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">ARM VGF</h3>
          <p class="card-text">Versatile graphics framework support</p>
        </div>
      </div>
    </div>

    <div class="col-md-3 col-sm-6 mb-4">
      <div class="card h-100">
        <div class="card-body">
          <h3 class="card-title">Cadence DSP</h3>
          <p class="card-text">Digital signal processor optimization</p>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="executorch-section alt-background">
  <h2 class="section-title">Success <span class="executorch-highlight">Stories</span></h2>
  <p class="section-subtitle">Production deployments and strategic partnerships accelerating edge AI</p>

  <div class="mb-5">
    <h3 class="h4 mb-3">Adoption</h3>
    <ul class="executorch-list">
      <li><strong><a href="https://engineering.fb.com/2025/07/28/android/executorch-on-device-ml-meta-family-of-apps/">Meta Family of Apps</a>:</strong> Production deployment across Instagram, Facebook, and WhatsApp</li>
      <li><strong>Meta Reality Labs:</strong> Powers Quest 3 VR and Ray-Ban Meta Smart Glasses AI</li>
    </ul>
  </div>

  <div class="mb-5">
    <h3 class="h4 mb-3">Ecosystem Integration</h3>
    <ul class="executorch-list">
      <li><strong>Hugging Face:</strong> Optimum-ExecuTorch for direct transformer model deployment</li>
      <li><strong>LiquidAI:</strong> Next-generation Liquid Foundation Models optimized for edge deployment</li>
      <li><strong>Software Mansion:</strong> React Native ExecuTorch bringing edge AI to mobile apps</li>
    </ul>
  </div>

  <div class="mb-5">
    <h3 class="h4 mb-3">Demos</h3>
    <ul class="executorch-list">
      <li><strong><a href="https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md">Llama</a>:</strong> Complete LLM implementation with quantization, KV caching, and mobile deployment</li>
      <li><strong><a href="https://github.com/pytorch/executorch/blob/main/examples/models/voxtral/README.md">Voxtral</a>:</strong> Multimodal AI combining text, vision, and audio processing in one model</li>
    </ul>
  </div>
</div>

<div class="executorch-cta-section">
  <h2>Ready to Deploy AI at the Edge?</h2>
  <p>Join thousands of developers using ExecuTorch in production</p>
  <a href="https://pytorch.org/executorch/stable/getting-started.html" class="btn btn-lg btn-white">Get Started Today</a>
</div>

<script>
// Backend switcher
function switchBackend(backend) {
  document.querySelectorAll('.backend-content').forEach(content => {
    content.classList.remove('active');
  });
  document.querySelectorAll('.backend-tab').forEach(tab => {
    tab.classList.remove('active');
  });
  document.getElementById(backend).classList.add('active');
  event.currentTarget.classList.add('active');
}

// Platform switcher
function switchPlatform(platform) {
  document.querySelectorAll('.platform-content').forEach(content => {
    content.classList.remove('active');
  });
  document.querySelectorAll('.platform-tab').forEach(tab => {
    tab.classList.remove('active');
  });
  document.getElementById(platform).classList.add('active');
  event.currentTarget.classList.add('active');
}

// Multimodal platform switcher
function switchMultimodalPlatform(platform) {
  document.querySelectorAll('.platform-content').forEach(content => {
    content.classList.remove('active');
  });
  document.querySelectorAll('.platform-tab').forEach(tab => {
    tab.classList.remove('active');
  });
  document.getElementById(platform).classList.add('active');
  event.currentTarget.classList.add('active');
}
</script>

    </div>
  </div>
</div>
