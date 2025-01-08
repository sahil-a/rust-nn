use compute::{get_metal_context, GPUBuffer};
use half::f16;

use rand::Rng;

// backprop calculations are from https://princeton-introml.github.io/files/ch11.pdf

pub struct Model {
    loss_fn: Box<dyn LossFn>,
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn layers(&self) -> &Vec<Box<dyn Layer>> {
        &self.layers
    }

    pub fn inference(&mut self, input: &GPUBuffer, output: &GPUBuffer) {
        let first = &self.layers[0];
        get_metal_context().copy(input, first.input());
        for i in 0..self.layers.len() - 1 {
            self.layers[i].forward(self.layers[i + 1].input())
        }
        let last = &self.layers[self.layers.len() - 1];
        last.forward(output);
    }

    pub fn train_step(
        &mut self,
        input: &GPUBuffer,
        output: &GPUBuffer,
        target: &GPUBuffer,
        gradients: Vec<&GPUBuffer>,
    ) -> f16 {
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
            gradient = self.layers[i].backward(gradient, curr_output, gradients[i]);
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

pub trait Layer {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn input(&self) -> &GPUBuffer; // layers should own a copy of their last input
    fn forward(&self, output: &GPUBuffer);
    // computes gradient gradient wrt weights and returns gradient wrt input
    fn backward(
        &self,
        gradient_wrt_output: &GPUBuffer,
        output: &GPUBuffer,
        gradient_wrt_weights: &GPUBuffer,
    ) -> &GPUBuffer;
    fn weights(&self) -> &GPUBuffer;
}

pub trait LossFn {
    // computes gradient wrt input and loss
    fn loss(&self, input: &GPUBuffer, target: &GPUBuffer) -> (f16, &GPUBuffer);
}

fn init_weights(num_inputs: usize, num_outputs: usize) -> GPUBuffer {
    let mut rng = rand::thread_rng();

    let boundary = f32::sqrt(6 as f32 / (num_inputs as f32));

    GPUBuffer::from_vec(
        num_outputs,
        num_inputs,
        &(0..num_inputs * num_outputs)
            .map(|_| f16::from_f32(rng.gen_range(-boundary..boundary)))
            .collect(),
    )
}

pub struct FullyConnectedLayer {
    weights: GPUBuffer,
    input: GPUBuffer,
    gradient_wrt_input: GPUBuffer,
    has_relu: bool,
}

impl FullyConnectedLayer {
    pub fn new(input_size: usize, output_size: usize, has_relu: bool) -> Self {
        let weights = init_weights(input_size, output_size);
        let input = GPUBuffer::new(input_size, 1);
        let gradient_wrt_input = GPUBuffer::new(1, input_size);

        Self {
            weights,
            input,
            gradient_wrt_input,
            has_relu,
        }
    }
}

impl Layer for FullyConnectedLayer {
    fn input_size(&self) -> usize {
        self.weights.cols
    }
    fn input(&self) -> &GPUBuffer {
        &self.input
    }
    fn output_size(&self) -> usize {
        self.weights.rows
    }

    fn forward(&self, output: &GPUBuffer) {
        let compute = get_metal_context();
        compute.matrix_multiply(&self.weights, &self.input, output, false, false);
        if self.has_relu {
            compute.relu(output);
        }
    }

    // computes gradient wrt input and gradient wrt weights
    fn backward(
        &self,
        gradient_wrt_output: &GPUBuffer,
        output: &GPUBuffer,
        gradient_wrt_weights: &GPUBuffer,
    ) -> &GPUBuffer {
        let compute = get_metal_context();
        // 1. gradient_wrt_output
        if self.has_relu {
            // hacky - it's okay to spoil outputs as the layer in front of us is done
            compute.positive_indicator(output);
            // use gradient wrt weights as a tmp buffer (it's the right size!)
            compute.matrix_multiply_rowwise(&self.weights, output, gradient_wrt_weights);
            compute.matrix_multiply(
                gradient_wrt_output,
                gradient_wrt_weights,
                output,
                false,
                false,
            );
        } else {
            compute.matrix_multiply(
                gradient_wrt_output,
                &self.weights,
                &self.gradient_wrt_input,
                false,
                false,
            )
        }

        // 2. gradient_wrt_weights
        compute.matrix_multiply(
            gradient_wrt_output,
            &self.input,
            gradient_wrt_weights,
            true,
            true,
        );
        if self.has_relu {
            // output should already be in positive indicator form from above
            compute.matrix_multiply_rowwise_in_place(gradient_wrt_weights, output);
        }

        &self.gradient_wrt_input
    }
    fn weights(&self) -> &GPUBuffer {
        &self.weights
    }
}

pub struct CrossEntropyLoss {
    gradient: GPUBuffer,
    softmax: GPUBuffer,
}

impl CrossEntropyLoss {
    pub fn new(num_classes: usize) -> Self {
        Self {
            gradient: GPUBuffer::new(1, num_classes),
            softmax: GPUBuffer::new(num_classes, 1),
        }
    }
}

impl LossFn for CrossEntropyLoss {
    fn loss(&self, input: &GPUBuffer, target: &GPUBuffer) -> (f16, &GPUBuffer) {
        let compute = get_metal_context();
        compute.softmax(input, &self.softmax);

        // gradient = (softmax(input) - target)^T
        compute.matrix_addition(
            &self.softmax,
            target,
            &self.gradient,
            f16::ONE,
            f16::from_f32(-1.0),
        );

        // loss = -log(softmax(input) * target)
        let loss = f16::from_f32(-f32::ln(compute.dot_product(&self.softmax, target)));

        (loss, &self.gradient)
    }
}
