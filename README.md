# Rust ML Experimentation

I try to code a neural network ~~from scratch~~ with minimal dependencies. Also see https://github.com/sahil-a/metal-rs-matmul.

```
➜ ag "dependencies" -G Cargo.toml -A 10
basic_nn/Cargo.toml
8:[dependencies]
9-csv = "1.3.0"
10-half = "2.4.1"
11-dataframe = { path = "../dataframe" }
12-model = { path = "../model" }
13-

dataframe/Cargo.toml
8:[dependencies]
9-csv = "1.3.0"
10-half = "2.4.1"
11-memmap = "0.7.0"
12-

compute/Cargo.toml
9:[dependencies]
10-half = "2.4.1"
11-metal = "0.30.0"
12-objc = "0.2.7"
13-

model/Cargo.toml
8:[dependencies]
9-
```
