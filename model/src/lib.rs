use compute::{get_metal_context, GPUBuffer};
use half::f16;

use rand::Rng;

// backprop calculations are from https://princeton-introml.github.io/files/ch11.pdf
pub mod layers;
pub mod loss_functions;

pub struct Model {
    loss_fn: Box<dyn LossFn>,
    layers: Vec<Box<dyn Layer>>,
}

pub trait Layer {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn input(&self) -> &GPUBuffer; // layers should own a copy of their last input
    fn gradient(&self) -> &GPUBuffer; // layers should own a copy of their last gradient
    fn forward(&self, output: &GPUBuffer);
    // stores gradient wrt weights and returns gradient wrt input
    fn backward(&self, gradient_wrt_output: &GPUBuffer, output: &GPUBuffer) -> &GPUBuffer;
    fn weights(&self) -> &GPUBuffer;
}

pub trait LossFn {
    // computes gradient wrt input and loss
    fn loss(&self, input: &GPUBuffer, target: &GPUBuffer) -> (f16, &GPUBuffer);
}

impl Model {
    pub fn layers(&self) -> &Vec<Box<dyn Layer>> {
        &self.layers
    }

    pub fn inference(&self, input: &GPUBuffer, output: &GPUBuffer) {
        let first = &self.layers[0];
        get_metal_context().copy(input, first.input());
        for i in 0..self.layers.len() - 1 {
            self.layers[i].forward(self.layers[i + 1].input())
        }
        let last = &self.layers[self.layers.len() - 1];
        last.forward(output);
    }

    pub fn train_step(&self, input: &GPUBuffer, output: &GPUBuffer, target: &GPUBuffer) -> f16 {
        let first = &self.layers[0];
        get_metal_context().copy(input, first.input());
        for i in 0..self.layers.len() - 1 {
            self.layers[i].forward(self.layers[i + 1].input())
        }
        let last = &self.layers[self.layers.len() - 1];
        last.forward(output);

        let (loss, mut gradient) = self.loss_fn.loss(output, target);
        let mut curr_output = output;
        for i in (0..self.layers.len()).rev() {
            gradient = self.layers[i].backward(gradient, curr_output);
            curr_output = self.layers[i].input();
        }

        loss
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
