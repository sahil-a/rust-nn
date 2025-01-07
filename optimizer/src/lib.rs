use compute::*;
use model::*;

// TODO
trait Optimizer {
    fn process_item(&mut self, item: Vec<&GPUBuffer>);
    // batch was processed, update model gradients
    fn update(&self, model: Model);
}
