# tinytorch.rs

> 🤖 **AI Disclaimer**: This `README.md` was generated with the assistance of **Gemini 3.8 Flash**.

A lightweight, modular deep learning library implemented entirely from first principles in safe, modern **Rust**—with zero external machine learning dependencies.

`tinytorch.rs` demonstrates how modern deep learning frameworks (such as PyTorch) operate under the hood: from contiguous 2D tensor memory management, cache-efficient matrix multiplications, and analytical autograd backward passes, to numerically stable activation functions, cross-entropy loss, the Adam optimizer, and an end-to-end computer vision inference engine.

---

## 🌟 Highlights & Achievements

- **Built from Scratch**: No LibTorch, no ONNX, no BLAS bindings, no high-level ML frameworks. Every matrix multiplication, forward pass, gradient calculation, and optimizer update is handwritten in safe Rust.
- **Verified with Numerical Differentiation**: Analytical backpropagation gradients across linear layers and activation functions are mathematically verified against finite-difference numerical approximations with relative errors $< 10^{-3}$.
- **97.68% Test Accuracy on MNIST**: An MLP trained on raw MNIST binary IDX data achieves over $97.6\%$ generalization accuracy on 10,000 unseen test images in just 5 epochs.
- **Sub-Millisecond Model Serialization**: Raw binary weight persistence (`save_weights` & `load_weights`) serializes all parameters into compact binary files that load in $< 1\text{ ms}$.
- **Real-World Image Inference Engine**: Custom CLI predictor utilizing Yann LeCun's 1998 MNIST preprocessing pipeline (aspect-preserving bounding-box cropping, center-of-mass alignment, background polarity detection) with an inference latency of **$\approx 300\ \mu\text{s}$** per image.

---

## 🧠 What's Implemented

### 1. 2D Tensor Substrate
- **Memory Layout**: Contiguous 1D `Vec<f32>` buffer mapped to `(rows, cols)` row-major shape.
- **Linear Algebra**:
  - Cache-friendly matrix multiplication ($O(M \cdot N \cdot K)$).
  - Matrix transpositions and axis-0 reduction/summation.
  - Broadcast row-vector bias addition.
  - Element-wise Hadamard arithmetic (addition, subtraction, multiplication, scalar scaling).
- **Weight Initializations**:
  - **Xavier / Glorot Uniform**: Uniform sampling scaled by $\sqrt{6 / (d_{\text{in}} + d_{\text{out}})}$, ideal for Sigmoid/Softmax.
  - **He / Kaiming Normal**: Normal distribution scaled by $\sqrt{2 / d_{\text{in}}}$, essential for avoiding vanishing gradients in ReLU networks.
- **Slicing & Extraction**: Row-range slicing (`slice_rows`) and arbitrary row gathering (`get_rows`).

### 2. Modular Layers & Autograd
All layers implement a uniform, composable `Layer` trait:
- **`Linear`**: Fully connected dense layer ($Y = XW + b$). Supports analytical backward passes with gradient accumulation:
  $$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T, \quad \frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}, \quad \frac{\partial L}{\partial b} = \sum_{\text{rows}} \frac{\partial L}{\partial Y}$$
- **`ReLU`**: Rectified Linear Unit activation ($f(x) = \max(0, x)$) with boolean mask backpropagation.
- **`Sigmoid`**: Logistic activation ($f(x) = \frac{1}{1 + e^{-x}}$) with analytical gradient $f'(x) = f(x)(1 - f(x))$.
- **`Softmax`**: Numerically stabilized multi-class probability distribution using the **Log-Sum-Exp max-subtraction trick** ($z_i - \max(z)$) to prevent floating-point overflow.

### 3. Loss Functions
- **`MSELoss`**: Mean Squared Error ($L = \frac{1}{N} \sum (y - \hat{y})^2$) with gradient $\frac{2}{N}(\hat{y} - y)$.
- **`BCELoss`**: Binary Cross-Entropy with numerical $\varepsilon$-clamping to prevent $\ln(0) = -\infty$.
- **`CrossEntropyLoss`**: Categorical Cross-Entropy with analytical gradient for Softmax outputs:
  $$\frac{\partial L}{\partial z} = \frac{\hat{y} - y}{N}$$

### 4. Optimizers
- **`SGD`**: Stochastic Gradient Descent with configurable learning rate.
- **`Adam`**: Adaptive Moment Estimation maintaining per-parameter running first moments ($m_t$) and second moments ($v_t$) with bias-correction:
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_t \leftarrow \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t$$

### 5. Model Container & Data Pipelines
- **`Sequential`**: Dynamically holds boxed layers, chaining forward passes via `.fold()` and propagating error gradients backwards in reverse order.
- **`DataLoader`**: Mini-batch generator supporting deterministic or in-place Fisher-Yates row shuffling.
- **Binary IDX Parser**: Direct big-endian binary decoder for MNIST `train-images-idx3-ubyte` and `train-labels-idx1-ubyte` files.
- **ASCII Terminal Visualizer**: Renders 28x28 normalized float tensors as ASCII art directly in the terminal.

### 6. Real-World Image Ingestion
A robust preprocessing pipeline that allows classifying real-world images (e.g. photos, paint drawings) against MNIST-trained weights:
1. Converts arbitrary image formats (PNG, JPEG, BMP, WebP) to 8-bit grayscale.
2. Automatically detects light paper vs. dark background via corner sampling and inverts polarity when necessary.
3. Computes the bounding box of active ink strokes ($> 30$ intensity threshold).
4. Proportionally scales the bounding box into a $20 \times 20$ box preserving aspect ratio.
5. Positions the digit onto a $28 \times 28$ canvas using its **Center of Mass** $(\bar{x}, \bar{y})$ (matching the exact 1998 LeCun specification).

---

## 🚀 Getting Started

### Prerequisites
- Install the Rust toolchain (Rust 1.85+ recommended):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### Running Tests
Execute the comprehensive test suite (47 automated unit tests, including mathematical finite-difference gradient checks):
```bash
cargo test
```

---

## 📖 Usage Examples

### 1. Constructing and Training a Model
```rust
use selfnn::model::Sequential;
use selfnn::layer::{Linear, ReLU, Softmax};
use selfnn::loss::{CrossEntropyLoss, Loss};
use selfnn::optim::Adam;

// Define architecture: 784 -> 128 -> 64 -> 10 -> Softmax
let mut model = Sequential::new(vec![
    Box::new(Linear::new(784, 128)),
    Box::new(ReLU::new()),
    Box::new(Linear::new(128, 64)),
    Box::new(ReLU::new()),
    Box::new(Linear::new(64, 10)),
    Box::new(Softmax::new()),
]);

let loss_fn = CrossEntropyLoss::new();
let optimizer = Adam::new(0.001);

// Forward pass
let predictions = model.forward(&batch_x);
let loss = loss_fn.forward(&predictions, &batch_y);

// Backpropagation & optimization
let grad = loss_fn.backward(&predictions, &batch_y);
model.backward(&grad);
optimizer.step(&mut model);
model.zero_grad();
```

### 2. Saving and Loading Weights
```rust
// Save trained parameters to disk
model.save_weights("data/mnist_weights.bin").expect("Failed to save weights");

// Instantaneous weight restoration (< 1 ms)
let mut loaded_model = Sequential::new(/* same architecture */);
loaded_model.load_weights("data/mnist_weights.bin").expect("Failed to load weights");
```

### 3. Training on MNIST
Place the uncompressed MNIST IDX binary files in `data/`:
- `data/train-images-idx3-ubyte`
- `data/train-labels-idx1-ubyte`
- `data/t10k-images-idx3-ubyte`
- `data/t10k-labels-idx1-ubyte`

Run the training pipeline:
```bash
cargo run --release --example train_mnist
```

Output:
```
Starting training (5 epochs, batch_size = 64)...
Epoch [1/5] - Avg Loss: 0.2707 - Time: 19.82s
Epoch [2/5] - Avg Loss: 0.1105 - Time: 20.97s
Epoch [3/5] - Avg Loss: 0.0753 - Time: 21.87s
Epoch [4/5] - Avg Loss: 0.0591 - Time: 21.23s
Epoch [5/5] - Avg Loss: 0.0447 - Time: 21.93s

Evaluating on 10,000 unseen test images...
Test Accuracy: 97.68%!
Saved trained weights to: data/mnist_weights.bin
```

### 4. Real-World Image Prediction
Pass any PNG, JPG, or BMP file (hand-drawn digit or photo):
```bash
cargo run --release --example predict -- path/to/my_digit.png
```

Sample output:
```
============================================================
        MNIST Real-World Digit Classifier on selfnn         
============================================================
Weights loaded in 820.62µs

Preprocessing 'data/8.png'...
Saved preprocessed 28x28 canvas to: data/preprocessed_input.png

╔══════════════════════════════════════════════════════╗
║  PREDICTED DIGIT: 8        Confidence:  59.4%        ║
╚══════════════════════════════════════════════════════╝

Confidence Distribution:
   [0]   6.9% | ███
   [1]   2.8% | █
   [2]   3.6% | █
   [3]   6.0% | ██
   [4]   3.5% | █
   [5]   8.6% | ███
   [6]   3.7% | █
   [7]   0.8% | 
👉 [8]  59.4% | ████████████████████████
   [9]   4.7% | ██

Inference latency: 324.46µs
```

---

## 🤝 Contributing

Contributions, experiments, and discussions are warmly welcome! Whether you are exploring how neural networks function mathematically or looking to enhance performance, here are a few exciting directions:

- **SIMD / Vectorization**: Accelerating tensor matrix multiplications using AVX2/AVX-512 or Rayon multithreading.
- **Convolutional Layers**: Adding `Conv2d` and `MaxPool2d` to form a full LeNet-5 architecture.
- **Dynamic Computational Graph**: Expanding from `Sequential` to a node-based Autograd engine (similar to micrograd).
- **New Optimizers & Losses**: Implementing RMSprop, AdaGrad, Huber Loss, or Cosine Annealing learning rate schedulers.
- **ONNX Export / Import**: Facilitating weight exchange with PyTorch or external runtimes.

### How to Contribute
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/conv2d`).
3. Ensure all tests pass and follow Clippy conventions:
   ```bash
   cargo test
   cargo clippy --all-targets
   ```
4. Commit your changes and submit a Pull Request!

---

## 📜 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
