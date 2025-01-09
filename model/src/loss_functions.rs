use crate::*;
use compute::{get_metal_context, GPUBuffer};
use half::f16;

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
