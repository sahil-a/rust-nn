# Rust ML Experimentation

I made a basic feedforward neural network ~~from scratch~~ with minimal dependencies. Also see https://github.com/sahil-a/metal-rs-matmul, which inspired this.
```bash
ag "dependencies" -G Cargo.toml -A 10
```
gives us:

```toml
basic_nn/Cargo.toml
8:[dependencies]
9-csv = "1.3.0"
10-half = "2.4.1"
11-dataframe = { path = "../dataframe" }
12-model = { path = "../model" }
13-compute = { path = "../compute" }
14-optimizer = { path = "../optimizer" }
15-

dataframe/Cargo.toml
8:[dependencies]
9-csv = "1.3.0"
10-half = "2.4.1"
11-memmap = "0.7.0"
12-rand = "0.8.5"
13-

optimizer/Cargo.toml
8:[dependencies]
9-model = { path = "../model" }
10-compute = { path = "../compute" }
11-half = "2.4.1"
12-

compute/Cargo.toml
9:[dependencies]
10-half = "2.4.1"
11-metal = "0.30.0"
12-objc = "0.2.7"
13-

model/Cargo.toml
8:[dependencies]
9-half = "2.4.1"
10-metal = "0.30.0"
11-compute = { path = "../compute" }
12-rand = "0.8.5"
13-
```

## Crates

### Dataframe

Handles data loading from a CSV to a memory efficient row major format of half precision floats. Memory maps a file backed dataframe and provides an interface for train/val/test split and batching.

### Compute

Provides elementary operations implemented in Metal as compute kernels. This includes matrix multiplication, addition, dot products, relu, softmax, etc. Also provides a GPU buffer wrapper so that other crates can keep all weights/gradients on GPU. I've added unit tests, because I obviously can't write kernels correctly on the first attempt.

### Model

Defines a model with variable layers and a loss function that is capable of inference and backpropagation. Layer and Loss function are defined as traits but the only provided implementations are `FullyConnectedLayer` and `CrossEntropyLoss`.

### Optimizer

Provides a very basic optimizer that accumulates gradients in a batch and applies the gradient update step. In the future, this would be the place to implement l2 regularization, momentum, and more advanced optimization strategies.

## Results

I've included an example dataset and training loop in `basic_nn/src/main.rs`. Training a very small model gives us these very modest results:

```
epoch 100: val accuracy 76.89, train accuracy 85.71
test accuracy 74.79
```

Try it yourself! - `git clone https://github.com/sahil-a/rust-nn.git && cd rust-nn && cargo run --release`.
