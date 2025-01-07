use half::f16;
use metal::*;
use objc::rc::autoreleasepool;
use std::mem;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Once;

static mut GLOBAL_CONTEXT: Option<MetalContext> = None;
static INIT: Once = Once::new();

pub fn initialize_metal_context(library_path: &str) {
    INIT.call_once(|| unsafe {
        GLOBAL_CONTEXT = Some(MetalContext::new(library_path));
    });
}

pub fn get_metal_context() -> &'static MetalContext {
    unsafe {
        GLOBAL_CONTEXT
            .as_ref()
            .expect("Metal context not initialized! Call initialize_metal_context() first")
    }
}

pub struct GPUBuffer {
    rows: usize,
    cols: usize,
    buffer: metal::Buffer,
}

impl GPUBuffer {
    pub fn new(device: &Device, rows: usize, cols: usize) -> Self {
        let buffer = device.new_buffer(
            (rows * cols * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        Self { rows, cols, buffer }
    }

    pub fn from_vec(device: &Device, rows: usize, cols: usize, vec: &Vec<f16>) -> Self {
        Self {
            rows,
            cols,
            buffer: create_buffer(device, vec),
        }
    }

    pub fn to_cpu_vec(&self) -> Vec<f16> {
        let array_len = self.rows * self.cols;
        let ptr = self.buffer.contents() as *const f16;
        unsafe { std::slice::from_raw_parts(ptr, array_len).to_vec() }
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

impl MetalContext {
    /// Create a new context from a `.metallib` file.
    pub fn new<P: AsRef<Path>>(library_path: P) -> Self {
        autoreleasepool(|| {
            let device = Device::system_default().expect("No Metal-capable device found!");
            let command_queue = device.new_command_queue();
            let library_path = PathBuf::from(library_path.as_ref());
            let library = device
                .new_library_with_file(library_path)
                .expect("Failed to load metallib");

            let dot_kernel = library.get_function("dot_product", None).unwrap();
            let dot_product_pipeline = device
                .new_compute_pipeline_state_with_function(&dot_kernel)
                .unwrap();

            let mat_kernel = library.get_function("matrix_multiply", None).unwrap();
            let matrix_multiply_pipeline = device
                .new_compute_pipeline_state_with_function(&mat_kernel)
                .unwrap();

            let relu_kernel = library.get_function("relu", None).unwrap();
            let relu_pipeline = device
                .new_compute_pipeline_state_with_function(&relu_kernel)
                .unwrap();

            let vector_multiply_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &library
                        .get_function("vector_pairwise_multiply", None)
                        .unwrap(),
                )
                .unwrap();

            let matrix_multiply_constant_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &library
                        .get_function("matrix_multiply_constant", None)
                        .unwrap(),
                )
                .unwrap();

            let matrix_multiply_rowwise_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &library
                        .get_function("matrix_multiply_rowwise", None)
                        .unwrap(),
                )
                .unwrap();

            let matrix_addition_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &library.get_function("matrix_addition", None).unwrap(),
                )
                .unwrap();

            let softmax_sum_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &library.get_function("softmax_sum", None).unwrap(),
                )
                .unwrap();

            let softmax_output_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &library.get_function("softmax_output", None).unwrap(),
                )
                .unwrap();

            let positive_indicator_pipeline = device
                .new_compute_pipeline_state_with_function(
                    &library.get_function("positive_indicator", None).unwrap(),
                )
                .unwrap();

            Self {
                device,
                command_queue,
                dot_product_pipeline,
                matrix_multiply_pipeline,
                matrix_multiply_constant_pipeline,
                matrix_multiply_rowwise_pipeline,
                matrix_addition_pipeline,
                relu_pipeline,
                vector_multiply_pipeline,
                softmax_sum_pipeline,
                softmax_output_pipeline,
                positive_indicator_pipeline,
            }
        })
    }

    /// Add two matrices with constant factors (c_a*A + c_b*B)
    pub fn matrix_addition(
        &self,
        input_a: &GPUBuffer,
        input_b: &GPUBuffer,
        output: &GPUBuffer,
        c_a: f16,
        c_b: f16,
    ) {
        assert_eq!(input_a.rows, input_b.rows);
        assert_eq!(input_a.cols, input_b.cols);
        assert_eq!(input_a.rows, output.rows);
        assert_eq!(input_a.cols, output.cols);

        let row_len = input_a.rows as u32;
        let col_len = input_a.cols as u32;

        autoreleasepool(|| {
            let c_a_buffer = create_buffer(&self.device, &[c_a]);
            let c_b_buffer = create_buffer(&self.device, &[c_b]);
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.matrix_addition_pipeline);
            encoder.set_buffer(0, Some(&input_a.buffer), 0);
            encoder.set_buffer(1, Some(&c_a_buffer), 0);
            encoder.set_buffer(2, Some(&input_b.buffer), 0);
            encoder.set_buffer(3, Some(&c_b_buffer), 0);
            encoder.set_buffer(4, Some(&output.buffer), 0);
            encoder.set_buffer(5, Some(&row_len_buffer), 0);
            encoder.set_buffer(6, Some(&col_len_buffer), 0);

            let threadgroup_size = MTLSize {
                width: self.matrix_addition_pipeline.thread_execution_width(),
                height: 1,
                depth: 1,
            };
            let col_threads = (col_len as u64 + 3) / 4;
            let threadgroup_count = MTLSize {
                width: ((row_len as u64 + threadgroup_size.width - 1) / threadgroup_size.width)
                    as u64,
                height: ((col_threads + threadgroup_size.width - 1) / threadgroup_size.width)
                    as u64,
                depth: 1,
            };

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        })
    }

    /// Multiply each row of a matrix by the corresponding scalar in a vector.
    pub fn matrix_multiply_rowwise(
        &self,
        input: &GPUBuffer,
        row_factors: &GPUBuffer,
        output: &GPUBuffer,
        mat_transposed: bool,
    ) {
        assert_eq!(input.rows, output.rows);
        assert_eq!(input.cols, output.cols);
        assert_eq!(row_factors.rows * row_factors.cols, input.rows);

        let row_len = input.rows as u32;
        let col_len = input.cols as u32;

        autoreleasepool(|| {
            let transposed_buffer = create_buffer(&self.device, &[mat_transposed]);
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.matrix_multiply_rowwise_pipeline);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&row_factors.buffer), 0);
            encoder.set_buffer(2, Some(&transposed_buffer), 0);
            encoder.set_buffer(3, Some(&output.buffer), 0);
            encoder.set_buffer(4, Some(&row_len_buffer), 0);
            encoder.set_buffer(5, Some(&col_len_buffer), 0);

            let threadgroup_size = MTLSize {
                width: self
                    .matrix_multiply_rowwise_pipeline
                    .thread_execution_width(),
                height: 1,
                depth: 1,
            };
            let col_threads = (col_len as u64 + 3) / 4;
            let threadgroup_count = MTLSize {
                width: ((row_len as u64 + threadgroup_size.width - 1) / threadgroup_size.width)
                    as u64,
                height: ((col_threads + threadgroup_size.width - 1) / threadgroup_size.width)
                    as u64,
                depth: 1,
            };

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        })
    }

    pub fn copy(&self, input: &GPUBuffer, output: &GPUBuffer) {
        self.matrix_multiply_constant(input, output, f16::ONE);
    }

    /// Multiply a matrix by a constant scalar value.
    pub fn matrix_multiply_constant(&self, input: &GPUBuffer, output: &GPUBuffer, constant: f16) {
        assert_eq!(input.rows, output.rows);
        assert_eq!(input.cols, output.cols);

        let row_len = input.rows as u32;
        let col_len = input.cols as u32;

        autoreleasepool(|| {
            let constant_buffer = create_buffer(&self.device, &[constant]);
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.matrix_multiply_constant_pipeline);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&constant_buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            encoder.set_buffer(3, Some(&row_len_buffer), 0);
            encoder.set_buffer(4, Some(&col_len_buffer), 0);

            let threadgroup_size = MTLSize {
                width: self
                    .matrix_multiply_constant_pipeline
                    .thread_execution_width(),
                height: 1,
                depth: 1,
            };
            let col_threads = (col_len as u64 + 3) / 4;
            let threadgroup_count = MTLSize {
                width: ((row_len as u64 + threadgroup_size.width - 1) / threadgroup_size.width)
                    as u64,
                height: ((col_threads + threadgroup_size.width - 1) / threadgroup_size.width)
                    as u64,
                depth: 1,
            };

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        })
    }

    /// Compute the dot product of two half-precision vectors on the GPU.
    /// Returns the resulting sum as a `u32`.
    pub fn dot_product(&self, a: &GPUBuffer, b: &GPUBuffer) -> u32 {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);

        let array_len = (a.rows * a.cols) as u32;

        autoreleasepool(|| {
            let output_buffer = create_buffer(&self.device, &[0u32]);
            let arraylen_buffer = create_buffer(&self.device, &[array_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.dot_product_pipeline);
            encoder.set_buffer(0, Some(&a.buffer), 0);
            encoder.set_buffer(1, Some(&b.buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&arraylen_buffer), 0);

            let num_threads = self.dot_product_pipeline.thread_execution_width();
            let threadgroup_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };
            let threadgroup_count = MTLSize {
                width: ((array_len as u64 + num_threads - 1) / num_threads) as u64,
                height: 1,
                depth: 1,
            };

            encoder.set_threadgroup_memory_length(
                0,
                threadgroup_size.width * (mem::size_of::<f16>() as u64),
            );

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            let ptr = output_buffer.contents() as *mut u32;
            unsafe { *ptr }
        })
    }

    /// Compute softmax on the contents of `input` and write to `output`.
    pub fn softmax(&self, input: &GPUBuffer, output: &GPUBuffer) {
        assert_eq!(input.rows, output.rows);
        assert_eq!(input.cols, output.cols);

        let array_len = (input.rows * input.cols) as u32;

        autoreleasepool(|| {
            let sum_buffer = create_buffer(&self.device, &[0.0f32]);
            let array_len_buffer = create_buffer(&self.device, &[array_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.softmax_sum_pipeline);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&sum_buffer), 0);
            encoder.set_buffer(2, Some(&array_len_buffer), 0);

            let num_threads = ((array_len + 3) / 4) as u64;
            let threadgroup_size = MTLSize {
                width: self.softmax_sum_pipeline.thread_execution_width(),
                height: 1,
                depth: 1,
            };
            let threadgroup_count = MTLSize {
                width: (num_threads + threadgroup_size.width - 1) / threadgroup_size.width,
                height: 1,
                depth: 1,
            };

            encoder.set_threadgroup_memory_length(
                0,
                threadgroup_size.width * (mem::size_of::<f32>() as u64),
            );

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.softmax_output_pipeline);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&sum_buffer), 0);
            encoder.set_buffer(2, Some(&array_len_buffer), 0);
            encoder.set_buffer(3, Some(&output.buffer), 0);

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        })
    }

    /// Multiply two vectors element-wise
    pub fn vector_multiply(&self, a: &GPUBuffer, b: &GPUBuffer, output: &GPUBuffer) {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        assert_eq!(a.rows, output.rows);
        assert_eq!(a.cols, output.cols);

        let array_len = (a.rows * a.cols) as u32;

        autoreleasepool(|| {
            let array_len_buffer = create_buffer(&self.device, &[array_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.vector_multiply_pipeline);
            encoder.set_buffer(0, Some(&a.buffer), 0);
            encoder.set_buffer(1, Some(&b.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            encoder.set_buffer(3, Some(&array_len_buffer), 0);

            let num_threads = ((array_len + 3) / 4) as u64;
            let threadgroup_size = MTLSize {
                width: self.vector_multiply_pipeline.thread_execution_width(),
                height: 1,
                depth: 1,
            };
            let threadgroup_count = MTLSize {
                width: (num_threads + threadgroup_size.width - 1) / threadgroup_size.width,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        })
    }

    /// Returns a new vector with ReLU applied (max(0, x) for each element).
    pub fn relu(&self, input: &GPUBuffer, output: &GPUBuffer) {
        assert_eq!(input.rows, output.rows);
        assert_eq!(input.cols, output.cols);

        let array_len = (input.rows * input.cols) as u32;

        autoreleasepool(|| {
            let array_len_buffer = create_buffer(&self.device, &[array_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.relu_pipeline);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            encoder.set_buffer(2, Some(&array_len_buffer), 0);

            let num_threads = ((array_len + 3) / 4) as u64;
            let threadgroup_size = MTLSize {
                width: self.relu_pipeline.thread_execution_width(),
                height: 1,
                depth: 1,
            };
            let threadgroup_count = MTLSize {
                width: (num_threads + threadgroup_size.width - 1) / threadgroup_size.width,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        })
    }

    /// Returns a new vector with 1.0 for positive values and 0.0 for non-positive values.
    pub fn positive_indicator(&self, input: &GPUBuffer, output: &GPUBuffer) {
        assert_eq!(input.rows, output.rows);
        assert_eq!(input.cols, output.cols);

        let array_len = (input.rows * input.cols) as u32;

        autoreleasepool(|| {
            let array_len_buffer = create_buffer(&self.device, &[array_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.positive_indicator_pipeline);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            encoder.set_buffer(2, Some(&array_len_buffer), 0);

            let num_threads = ((array_len + 3) / 4) as u64;
            let threadgroup_size = MTLSize {
                width: self.positive_indicator_pipeline.thread_execution_width(),
                height: 1,
                depth: 1,
            };
            let threadgroup_count = MTLSize {
                width: (num_threads + threadgroup_size.width - 1) / threadgroup_size.width,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        })
    }

    /// Multiply two matrices (A of size row_len x inner_len, and B of size inner_len x col_len)
    /// stored in row-major order.
    pub fn matrix_multiply(
        &self,
        a: &GPUBuffer,
        b: &GPUBuffer,
        output: &GPUBuffer,
        a_transposed: bool,
        b_transposed: bool,
        thread_count: u32,
    ) {
        let row_len = a.rows as u32;
        let inner_len = a.cols as u32;
        let col_len = b.cols as u32;
        assert_eq!(b.rows as u32, inner_len);
        assert_eq!(output.rows as u32, row_len);
        assert_eq!(output.cols as u32, col_len);

        autoreleasepool(|| {
            let tile_size = thread_count;
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let inner_len_buffer = create_buffer(&self.device, &[inner_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);
            let tile_size_buffer = create_buffer(&self.device, &[tile_size]);
            let a_transposed_buffer = create_buffer(&self.device, &[a_transposed]);
            let b_transposed_buffer = create_buffer(&self.device, &[b_transposed]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.matrix_multiply_pipeline);
            encoder.set_buffer(0, Some(&a.buffer), 0);
            encoder.set_buffer(1, Some(&b.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            encoder.set_buffer(3, Some(&row_len_buffer), 0);
            encoder.set_buffer(4, Some(&inner_len_buffer), 0);
            encoder.set_buffer(5, Some(&col_len_buffer), 0);
            encoder.set_buffer(6, Some(&a_transposed_buffer), 0);
            encoder.set_buffer(7, Some(&b_transposed_buffer), 0);
            encoder.set_buffer(8, Some(&tile_size_buffer), 0);

            let threadgroup_count = MTLSize {
                width: ((row_len + tile_size - 1) / tile_size) as u64,
                height: ((col_len + tile_size - 1) / tile_size) as u64,
                depth: 1,
            };
            let threadgroup_size = MTLSize {
                width: tile_size as u64,
                height: tile_size as u64,
                depth: 1,
            };

            encoder.set_threadgroup_memory_length(
                0,
                (tile_size * tile_size) as u64 * (mem::size_of::<f16>() as u64),
            );
            encoder.set_threadgroup_memory_length(
                1,
                (tile_size * tile_size) as u64 * (mem::size_of::<f16>() as u64),
            );

            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        })
    }
}

fn create_buffer<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let size = (data.len() * std::mem::size_of::<T>()) as u64;
    let raw_ptr = data.as_ptr() as *const std::ffi::c_void;
    device.new_buffer_with_data(raw_ptr, size, MTLResourceOptions::CPUCacheModeDefaultCache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn cpu_matrix_multiply(
        a: &[f16],
        b: &[f16],
        row_len: usize,
        inner_len: usize,
        col_len: usize,
        a_transposed: bool,
        b_transposed: bool,
    ) -> Vec<f16> {
        let mut result = vec![f16::from_f32(0.0); row_len * col_len];
        for i in 0..row_len {
            for j in 0..col_len {
                let mut sum = f16::from_f32(0.0);
                for k in 0..inner_len {
                    let a_idx = if a_transposed {
                        k * row_len + i
                    } else {
                        i * inner_len + k
                    };
                    let b_idx = if b_transposed {
                        (j * inner_len + k) as usize
                    } else {
                        k * col_len + j
                    };
                    sum = sum + a[a_idx] * b[b_idx];
                }
                result[i * col_len + j] = sum;
            }
        }
        result
    }

    #[test]
    fn matrix_multiply() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let row_len = 258;
        let inner_len = 256;
        let col_len = 259;
        let mat_a = vec![f16::from_f32(2.0); (row_len * inner_len) as usize];
        let mat_b = vec![f16::from_f32(4.0); (inner_len * col_len) as usize];

        let gpu_a = GPUBuffer::from_vec(
            &context.device,
            row_len as usize,
            inner_len as usize,
            &mat_a,
        );
        let gpu_b = GPUBuffer::from_vec(
            &context.device,
            inner_len as usize,
            col_len as usize,
            &mat_b,
        );
        let gpu_out = GPUBuffer::new(&context.device, row_len as usize, col_len as usize);

        context.matrix_multiply(
            &gpu_a,
            &gpu_b,
            &gpu_out,
            false,
            false,
            context.matrix_multiply_pipeline.thread_execution_width() as u32,
        );
        let result = gpu_out.to_cpu_vec();

        let cpu_result =
            cpu_matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len, false, false);

        assert_eq!(result.len(), cpu_result.len());
        for (gpu_val, cpu_val) in result.iter().zip(cpu_result.iter()) {
            assert!(
                (gpu_val.to_f32() - cpu_val.to_f32()).abs() < 1e-3,
                "Mismatch in matrix multiply"
            );
        }
    }

    #[test]
    fn matrix_multiply_transposed() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let row_len = 5;
        let inner_len = 5;
        let col_len = 5;

        let mat_a = (0..(row_len * inner_len))
            .map(|i| f16::from_f32((i % 5) as f32))
            .collect::<Vec<_>>();
        let mat_b = (0..(inner_len * col_len))
            .map(|i| f16::from_f32((i % 5) as f32))
            .collect::<Vec<_>>();

        let gpu_a = GPUBuffer::from_vec(
            &context.device,
            row_len as usize,
            inner_len as usize,
            &mat_a,
        );
        let gpu_b = GPUBuffer::from_vec(
            &context.device,
            inner_len as usize,
            col_len as usize,
            &mat_b,
        );
        let gpu_out = GPUBuffer::new(&context.device, row_len as usize, col_len as usize);

        context.matrix_multiply(
            &gpu_a,
            &gpu_b,
            &gpu_out,
            false,
            true,
            context.matrix_multiply_pipeline.thread_execution_width() as u32,
        );
        let result = gpu_out.to_cpu_vec();

        let cpu_a = (0..(row_len * inner_len))
            .map(|i| f16::from_f32((i % 5) as f32))
            .collect::<Vec<_>>();
        let cpu_b = (0..(inner_len * col_len))
            .map(|i| f16::from_f32((i % 5) as f32))
            .collect::<Vec<_>>();
        let cpu_result =
            cpu_matrix_multiply(&cpu_a, &cpu_b, row_len, inner_len, col_len, false, true);

        assert_eq!(result.len(), cpu_result.len());
        for (gpu_val, cpu_val) in result.iter().zip(cpu_result.iter()) {
            assert!(
                (gpu_val.to_f32() - cpu_val.to_f32()).abs() < 1e-3,
                "Mismatch in matrix multiply (transposed)"
            );
        }
    }

    #[test]
    fn matrix_addition() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let row_len = 3;
        let col_len = 3;
        let a: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b: Vec<f16> = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_a = GPUBuffer::from_vec(&context.device, row_len, col_len, &a);
        let gpu_b = GPUBuffer::from_vec(&context.device, row_len, col_len, &b);
        let gpu_out = GPUBuffer::new(&context.device, row_len, col_len);

        let c_a = f16::from_f32(2.0);
        let c_b = f16::from_f32(3.0);

        context.matrix_addition(&gpu_a, &gpu_b, &gpu_out, c_a, c_b);
        let result = gpu_out.to_cpu_vec();

        let expected: Vec<f16> = vec![
            2.0 * 1.0 + 3.0 * 9.0,
            2.0 * 2.0 + 3.0 * 8.0,
            2.0 * 3.0 + 3.0 * 7.0,
            2.0 * 4.0 + 3.0 * 6.0,
            2.0 * 5.0 + 3.0 * 5.0,
            2.0 * 6.0 + 3.0 * 4.0,
            2.0 * 7.0 + 3.0 * 3.0,
            2.0 * 8.0 + 3.0 * 2.0,
            2.0 * 9.0 + 3.0 * 1.0,
        ]
        .into_iter()
        .map(f16::from_f32)
        .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got.to_f32() - want.to_f32()).abs() < 1e-3);
        }
    }

    #[test]
    fn matrix_multiply_rowwise() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let row_len = 3;
        let col_len = 4;
        let input: Vec<f16> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]
        .into_iter()
        .map(f16::from_f32)
        .collect();
        let row_multipliers: Vec<f16> =
            vec![2.0, 3.0, 4.0].into_iter().map(f16::from_f32).collect();

        let gpu_input = GPUBuffer::from_vec(&context.device, row_len, col_len, &input);
        let gpu_out = GPUBuffer::new(&context.device, row_len, col_len);
        let gpu_row_factors =
            GPUBuffer::from_vec(&context.device, row_multipliers.len(), 1, &row_multipliers);

        context.matrix_multiply_rowwise(&gpu_input, &gpu_row_factors, &gpu_out, false);
        let result = gpu_out.to_cpu_vec();

        let expected: Vec<f16> = vec![
            2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0, 36.0, 40.0, 44.0, 48.0,
        ]
        .into_iter()
        .map(f16::from_f32)
        .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got.to_f32() - want.to_f32()).abs() < 1e-3);
        }

        let gpu_input_t = GPUBuffer::from_vec(&context.device, row_len, col_len, {
            &vec![
                1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
            ]
            .into_iter()
            .map(f16::from_f32)
            .collect()
        });
        let gpu_out_t = GPUBuffer::new(&context.device, row_len, col_len);
        context.matrix_multiply_rowwise(&gpu_input_t, &gpu_row_factors, &gpu_out_t, true);
        let result_t = gpu_out_t.to_cpu_vec();

        assert_eq!(result_t.len(), expected.len());
        for (got, want) in result_t.iter().zip(expected.iter()) {
            assert!((got.to_f32() - want.to_f32()).abs() < 1e-3);
        }
    }

    #[test]
    fn matrix_multiply_constant() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let row_len = 3;
        let col_len = 3;
        let input: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_input = GPUBuffer::from_vec(&context.device, row_len, col_len, &input);
        let gpu_out = GPUBuffer::new(&context.device, row_len, col_len);
        let constant = f16::from_f32(2.0);

        context.matrix_multiply_constant(&gpu_input, &gpu_out, constant);
        let result = gpu_out.to_cpu_vec();

        let expected: Vec<f16> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got.to_f32() - want.to_f32()).abs() < 1e-3);
        }
    }

    fn cpu_softmax(input: &[f16]) -> Vec<f16> {
        let exp_values: Vec<f32> = input.iter().map(|x| x.to_f32().exp()).collect();
        let sum: f32 = exp_values.iter().sum();
        exp_values.iter().map(|x| f16::from_f32(x / sum)).collect()
    }

    #[test]
    fn softmax() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let input: Vec<f16> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_in = GPUBuffer::from_vec(&context.device, 1, input.len(), &input);
        let gpu_out = GPUBuffer::new(&context.device, 1, input.len());
        context.softmax(&gpu_in, &gpu_out);
        let result = gpu_out.to_cpu_vec();
        let expected = cpu_softmax(&input);

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got.to_f32() - want.to_f32()).abs() < 1e-3);
        }
        let sum: f32 = result.iter().map(|x| x.to_f32()).sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn positive_indicator() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let input: Vec<f16> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_in = GPUBuffer::from_vec(&context.device, 1, input.len(), &input);
        let gpu_out = GPUBuffer::new(&context.device, 1, 5);
        context.positive_indicator(&gpu_in, &gpu_out);
        let result = gpu_out.to_cpu_vec();

        let expected: Vec<f16> = vec![0.0, 0.0, 0.0, 1.0, 1.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got.to_f32() - want.to_f32()).abs() < 1e-3);
        }
    }

    #[test]
    fn vector_multiply() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let a: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b: Vec<f16> = vec![2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_a = GPUBuffer::from_vec(&context.device, 1, a.len(), &a);
        let gpu_b = GPUBuffer::from_vec(&context.device, 1, b.len(), &b);
        let gpu_out = GPUBuffer::new(&context.device, 1, 5);

        context.vector_multiply(&gpu_a, &gpu_b, &gpu_out);
        let result = gpu_out.to_cpu_vec();

        let expected: Vec<f16> = vec![2.0, 6.0, 12.0, 20.0, 30.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got.to_f32() - want.to_f32()).abs() < 1e-3);
        }
    }

    #[test]
    fn relu() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let input: Vec<f16> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_in = GPUBuffer::from_vec(&context.device, 1, 5, &input);
        let gpu_out = GPUBuffer::new(&context.device, 1, 5);
        context.relu(&gpu_in, &gpu_out);
        let result = gpu_out.to_cpu_vec();

        let expected: Vec<f16> = vec![0.0, 0.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got.to_f32() - want.to_f32()).abs() < 1e-3);
        }
    }

    #[test]
    fn benchmark_matrix_multiply() {
        initialize_metal_context("compute-kernel.metallib");
        let context = get_metal_context();
        let row_len = 1024;
        let inner_len = 1024;
        let col_len = 1024;
        let mat_a = vec![f16::from_f32(2.0); (row_len * inner_len) as usize];
        let mat_b = vec![f16::from_f32(4.0); (inner_len * col_len) as usize];
        let total_ops = 2.0 * row_len as f64 * col_len as f64 * inner_len as f64;

        let gpu_a = GPUBuffer::from_vec(&context.device, row_len, inner_len, &mat_a);
        let gpu_b = GPUBuffer::from_vec(&context.device, inner_len, col_len, &mat_b);
        let gpu_out = GPUBuffer::new(&context.device, row_len, col_len);

        for _ in 0..2 {
            context.matrix_multiply(
                &gpu_a,
                &gpu_b,
                &gpu_out,
                false,
                false,
                context.matrix_multiply_pipeline.thread_execution_width() as u32,
            );
        }

        let iterations = 5;
        let gpu_start = Instant::now();
        for _ in 0..iterations {
            context.matrix_multiply(
                &gpu_a,
                &gpu_b,
                &gpu_out,
                false,
                false,
                context.matrix_multiply_pipeline.thread_execution_width() as u32,
            );
        }
        let gpu_total_time = gpu_start.elapsed();
        let gpu_avg_time = gpu_total_time / iterations;
        let gpu_avg_time_s = gpu_avg_time.as_secs_f64();
        let gpu_gflops = (total_ops / gpu_avg_time_s) / 1e9;

        println!(
            "GPU: ran {iterations} multiplies in {:#?} total; ~{:#?} each => ~{:.2} GFLOPS",
            gpu_total_time, gpu_avg_time, gpu_gflops
        );

        assert!(gpu_gflops > 100.0);

        let cpu_start = Instant::now();
        let _ = cpu_matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len, false, false);
        let cpu_total_time = cpu_start.elapsed();
        let cpu_avg_time_s = cpu_total_time.as_secs_f64();
        let cpu_gflops = (total_ops / cpu_avg_time_s) / 1e9;

        println!(
            "CPU: ran 1 multiply in {:#?} => ~{:.2} GFLOPS",
            cpu_total_time, cpu_gflops
        );

        let speedup = cpu_avg_time_s / gpu_avg_time_s;
        println!("GPU is about {:.2}x faster than CPU.", speedup);
    }
}
