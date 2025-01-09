use half::f16;
use metal::*;
use std::sync::Once;

pub mod operations;
pub mod tests;

static mut GLOBAL_CONTEXT: Option<MetalContext> = None;
static INIT: Once = Once::new();

pub fn initialize_metal_context() {
    INIT.call_once(|| unsafe {
        GLOBAL_CONTEXT = Some(MetalContext::new("compute-kernel.metallib"));
    });
}

pub fn initialize_metal_context_from(path: &str) {
    INIT.call_once(|| unsafe {
        GLOBAL_CONTEXT = Some(MetalContext::new(path));
    });
}

pub fn get_metal_context() -> &'static MetalContext {
    unsafe {
        GLOBAL_CONTEXT
            .as_ref()
            .expect("Metal context not initialized! Call initialize_metal_context() first")
    }
}

/// A reusable context holding Metal device, command queue, and precompiled pipelines.
pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    dot_product_pipeline: ComputePipelineState,
    matrix_multiply_pipeline: ComputePipelineState,
    matrix_multiply_constant_pipeline: ComputePipelineState,
    matrix_multiply_rowwise_pipeline: ComputePipelineState,
    matrix_addition_pipeline: ComputePipelineState,
    relu_pipeline: ComputePipelineState,
    vector_multiply_pipeline: ComputePipelineState,
    softmax_sum_pipeline: ComputePipelineState,
    softmax_output_pipeline: ComputePipelineState,
    positive_indicator_pipeline: ComputePipelineState,
}

pub struct GPUBuffer {
    pub rows: usize,
    pub cols: usize,
    buffer: metal::Buffer,
}

impl GPUBuffer {
    pub fn new(rows: usize, cols: usize) -> Self {
        let buffer = get_metal_context().device.new_buffer(
            (rows * cols * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        Self { rows, cols, buffer }
    }

    pub fn from_vec(rows: usize, cols: usize, vec: &[f16]) -> Self {
        Self {
            rows,
            cols,
            buffer: create_buffer(&get_metal_context().device, vec),
        }
    }

    pub fn to_cpu_vec(&self) -> Vec<f16> {
        let array_len = self.rows * self.cols;
        let ptr = self.buffer.contents() as *const f16;
        unsafe { std::slice::from_raw_parts(ptr, array_len).to_vec() }
    }
}

fn create_buffer<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let size = (data.len() * std::mem::size_of::<T>()) as u64;
    let raw_ptr = data.as_ptr() as *const std::ffi::c_void;
    device.new_buffer_with_data(raw_ptr, size, MTLResourceOptions::CPUCacheModeDefaultCache)
}
