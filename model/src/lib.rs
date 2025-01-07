use compute::{get_metal_context, GPUBuffer};
use half::f16;

pub struct Model {
    loss_fn: Box<dyn LossFn>,
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    fn layers(&mut self) -> &mut Vec<Box<dyn Layer>> {
        &mut self.layers
    }

    fn inference(&mut self, input: &GPUBuffer, output: &GPUBuffer) {
        let first = &self.layers[0];
        get_metal_context().copy(input, first.input());
        for i in 0..self.layers.len() - 1 {
            self.layers[i].forward(self.layers[i + 1].input())
        }
        let last = &self.layers[self.layers.len() - 1];
        last.forward(output)
    }

    fn forward_backward(
        &mut self,
        input: &GPUBuffer,
        output: &GPUBuffer,
        gradients: Vec<&GPUBuffer>,
    ) {
        // TODO
    }
}

pub struct ModelBuilder {
    input_size: usize,
    output_size: usize,
    layers: Vec<Box<dyn Layer>>,
}

impl ModelBuilder {
    pub fn input_size(input_size: usize) -> Self {
        Self {
            input_size,
            output_size: 0,
            layers: vec![],
        }
    }
    pub fn layer(self, layer: Box<dyn Layer>) -> Result<Self, String> {
        if !self.layers.is_empty() && layer.input_size() != self.output_size {
            return Err(format!(
                "Layer input size {} does not match previous layer output size {}",
                layer.input_size(),
                self.output_size
            ));
        }

        let output_size = layer.output_size();
        Ok(Self {
            input_size: self.input_size,
            output_size,
            layers: {
                let mut layers = self.layers;
                layers.push(layer);
                layers
            },
        })
    }

    pub fn loss_fn(self, loss_fn: Box<dyn LossFn>) -> Result<Model, String> {
        if self.layers.is_empty() {
            return Err("Model must have at least one layer".to_string());
        }
        Ok(Model {
            loss_fn,
            layers: self.layers,
        })
    }
}

pub trait Layer {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn input(&self) -> &GPUBuffer; // layers should own a copy of their last input
    fn forward(&self, output: &GPUBuffer);
    // computes gradient wrt input and gradient wrt weights
    fn backward(
        &self,
        gradient_wrt_output: &GPUBuffer,
        gradient_wrt_input: &GPUBuffer,
        gradient_wrt_weights: &GPUBuffer,
    );
    fn weights(&mut self) -> &mut GPUBuffer;
    fn mask(&mut self) -> &mut GPUBuffer;
}

pub trait LossFn {
    // computes gradient wrt input and returns loss
    fn loss(self, input: GPUBuffer, target: &GPUBuffer, gradient: &GPUBuffer) -> f16;
}

pub struct FullyConnectedLayer {
    weights: GPUBuffer,
    mask: GPUBuffer,
    buffer: GPUBuffer,
}

// TODO
impl Layer for FullyConnectedLayer {
    fn input_size(&self) -> usize {
        0
    }
    fn input(&self) -> &GPUBuffer {
        &self.buffer
    }
    fn output_size(&self) -> usize {
        0
    }
    fn forward(&self, output: &GPUBuffer) {}
    // computes gradient wrt input and gradient wrt weights
    fn backward(
        &self,
        gradient_wrt_output: &GPUBuffer,
        gradient_wrt_input: &GPUBuffer,
        gradient_wrt_weights: &GPUBuffer,
    ) {
    }
    fn weights(&mut self) -> &mut GPUBuffer {
        &mut self.weights
    }
    fn mask(&mut self) -> &mut GPUBuffer {
        &mut self.mask
    }
}

pub struct CrossEntropyLoss {}

impl LossFn for CrossEntropyLoss {
    fn loss(self, input: GPUBuffer, target: &GPUBuffer, gradient: &GPUBuffer) -> f16 {
        // TODO
        f16::ZERO
    }
}
