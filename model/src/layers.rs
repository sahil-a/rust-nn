use crate::*;
use compute::{get_metal_context, GPUBuffer};

pub struct FullyConnectedLayer {
    weights: GPUBuffer,
    input: GPUBuffer,
    gradient: GPUBuffer,
    gradient_wrt_input: GPUBuffer,
    has_relu: bool,
}

impl FullyConnectedLayer {
    pub fn new(input_size: usize, output_size: usize, has_relu: bool) -> Self {
        let weights = init_weights(input_size, output_size);
        let gradient = GPUBuffer::new(output_size, input_size);
        let input = GPUBuffer::new(input_size, 1);
        let gradient_wrt_input = GPUBuffer::new(1, input_size);

        Self {
            weights,
            input,
            gradient,
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

    fn gradient(&self) -> &GPUBuffer {
        &self.gradient
    }

    fn forward(&self, output: &GPUBuffer) {
        let compute = get_metal_context();
        compute.matrix_multiply(&self.weights, &self.input, output, false, false);
        if self.has_relu {
            compute.relu(output);
        }
    }

    // computes gradient wrt input and gradient wrt weights
    fn backward(&self, gradient_wrt_output: &GPUBuffer, output: &GPUBuffer) -> &GPUBuffer {
        let compute = get_metal_context();
        // 1. gradient_wrt_output
        if self.has_relu {
            // hacky - it's okay to spoil outputs as the layer in front of us is done
            compute.positive_indicator(output);
            // use gradient wrt weights as a tmp buffer (it's the right size!)
            compute.matrix_multiply_rowwise(&self.weights, output, &self.gradient);
            compute.matrix_multiply(
                gradient_wrt_output,
                &self.gradient,
                &self.gradient_wrt_input,
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
        compute.matrix_multiply(gradient_wrt_output, &self.input, &self.gradient, true, true);
        if self.has_relu {
            // output should already be in positive indicator form from above
            compute.matrix_multiply_rowwise(&self.gradient, output, &self.gradient);
        }

        &self.gradient_wrt_input
    }
    fn weights(&self) -> &GPUBuffer {
        &self.weights
    }
}

fn init_weights(num_inputs: usize, num_outputs: usize) -> GPUBuffer {
    let mut rng = rand::thread_rng();

    let boundary = f32::sqrt(2.0 / (num_inputs as f32));

    GPUBuffer::from_vec(
        num_outputs,
        num_inputs,
        &(0..num_inputs * num_outputs)
            .map(|_| f16::from_f32(rng.gen_range(-boundary..boundary)))
            .collect::<Vec<f16>>(),
    )
}
