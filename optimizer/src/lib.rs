use compute::*;
use half::f16;
use model::*;

pub trait Optimizer {
    fn process_step(&mut self);
    // batch was processed, update model gradients
    fn update_gradients(&mut self);
}

pub struct FixedLearningRateOptimizer<'a> {
    model: &'a Model,
    accumulated_gradients: Vec<GPUBuffer>,
    steps: usize,
    learning_rate: f16,
}

impl<'a> FixedLearningRateOptimizer<'a> {
    pub fn new(model: &'a Model, learning_rate: f16) -> Self {
        let accumulated_gradients = model
            .layers()
            .iter()
            .map(|layer| {
                let weights = layer.weights();
                GPUBuffer::new(weights.rows, weights.cols)
            })
            .collect();

        Self {
            model,
            accumulated_gradients,
            steps: 0,
            learning_rate,
        }
    }
}

impl<'a> Optimizer for FixedLearningRateOptimizer<'a> {
    fn process_step(&mut self) {
        for (acc_grad, layer) in self.accumulated_gradients.iter().zip(self.model.layers()) {
            let compute = get_metal_context();
            compute.matrix_addition(acc_grad, layer.gradient(), acc_grad, f16::ONE, f16::ONE);
        }
        self.steps += 1;
    }

    fn update_gradients(&mut self) {
        let compute = get_metal_context();
        let steps = f16::from_f32(self.steps as f32);

        for (acc_grad, layer) in self.accumulated_gradients.iter().zip(self.model.layers()) {
            compute.matrix_multiply_constant(acc_grad, acc_grad, f16::ONE / steps);

            // w = w - learning_rate * gradient
            compute.matrix_addition(
                layer.weights(),
                acc_grad,
                layer.weights(),
                f16::ONE,
                -self.learning_rate,
            );

            compute.matrix_multiply_constant(acc_grad, acc_grad, f16::ZERO);
        }

        self.steps = 0;
    }
}
