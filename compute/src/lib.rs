use half::f16;
use metal::*;
use objc::rc::autoreleasepool;
use std::mem;
use std::path::Path;
use std::path::PathBuf;

/// A reusable context holding Metal device, command queue, and precompiled pipelines.
pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    dot_product_pipeline: ComputePipelineState,
    matrix_multiply_pipeline: ComputePipelineState,
    matrix_multiply_constant_pipeline: ComputePipelineState,
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
            // 1. Get the default system GPU device
            let device = Device::system_default().expect("No Metal-capable device found!");

            // 2. Create a command queue
            let command_queue = device.new_command_queue();

            // 3. Load the `.metallib` file
            let library_path = PathBuf::from(library_path.as_ref());
            let library = device
                .new_library_with_file(library_path)
                .expect("Failed to load metallib");

            // 4. Create pipeline for dot_product kernel
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
    /// Input matrices are row_len x col_len in row-major order.
    /// Returns a new matrix of the same dimensions.
    pub fn matrix_addition(
        &self,
        a: &[f16],
        b: &[f16],
        c_a: f16,
        c_b: f16,
        row_len: u32,
        col_len: u32,
    ) -> Vec<f16> {
        let array_len = row_len * col_len;
        assert_eq!(
            a.len() as u32,
            array_len,
            "Matrix A dimensions don't match input length"
        );
        assert_eq!(
            b.len() as u32,
            array_len,
            "Matrix B dimensions don't match input length"
        );

        autoreleasepool(|| {
            // 1. Create buffers
            let input_buffer_a = create_buffer(&self.device, a);
            let input_buffer_b = create_buffer(&self.device, b);
            let c_a_buffer = create_buffer(&self.device, &[c_a]);
            let c_b_buffer = create_buffer(&self.device, &[c_b]);
            let output_buffer = create_buffer(
                &self.device,
                vec![f16::from_f32(0.0); array_len as usize].as_slice(),
            );
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);

            // 2. Create command buffer & encoder
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers
            encoder.set_compute_pipeline_state(&self.matrix_addition_pipeline);
            encoder.set_buffer(0, Some(&input_buffer_a), 0);
            encoder.set_buffer(1, Some(&c_a_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer_b), 0);
            encoder.set_buffer(3, Some(&c_b_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);
            encoder.set_buffer(5, Some(&row_len_buffer), 0);
            encoder.set_buffer(6, Some(&col_len_buffer), 0);

            // 4. Determine thread layout (process 4 elements per thread in y dimension)
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

            // 5. Encode & dispatch
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 6. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 7. Read results
            let ptr = output_buffer.contents() as *const f16;
            let output_slice = unsafe { std::slice::from_raw_parts(ptr, array_len as usize) };
            output_slice.to_vec()
        })
    }

    /// Multiply a matrix by a constant scalar value.
    /// Input matrix is row_len x col_len in row-major order.
    /// Returns a new matrix of the same dimensions.
    pub fn matrix_multiply_constant(
        &self,
        mat: &[f16],
        constant: f16,
        row_len: u32,
        col_len: u32,
    ) -> Vec<f16> {
        let array_len = row_len * col_len;
        assert_eq!(
            mat.len() as u32,
            array_len,
            "Matrix dimensions don't match input length"
        );

        autoreleasepool(|| {
            // 1. Create buffers
            let input_buffer = create_buffer(&self.device, mat);
            let constant_buffer = create_buffer(&self.device, &[constant]);
            let output_buffer = create_buffer(
                &self.device,
                vec![f16::from_f32(0.0); array_len as usize].as_slice(),
            );
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);

            // 2. Create command buffer & encoder
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers
            encoder.set_compute_pipeline_state(&self.matrix_multiply_constant_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&constant_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&row_len_buffer), 0);
            encoder.set_buffer(4, Some(&col_len_buffer), 0);

            // 4. Determine thread layout (process 4 elements per thread in y dimension)
            let threadgroup_size = MTLSize {
                width: self
                    .matrix_multiply_constant_pipeline
                    .thread_execution_width(),
                height: 1,
                depth: 1,
            };
            let col_threads = (col_len as u64 as u64 + 3) / 4;
            let threadgroup_count = MTLSize {
                width: ((row_len as u64 + threadgroup_size.width - 1) / threadgroup_size.width)
                    as u64,
                height: ((col_threads + threadgroup_size.width - 1) / threadgroup_size.width)
                    as u64,
                depth: 1,
            };

            // 5. Encode & dispatch
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 6. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 7. Read results
            let ptr = output_buffer.contents() as *const f16;
            let output_slice = unsafe { std::slice::from_raw_parts(ptr, array_len as usize) };
            output_slice.to_vec()
        })
    }

    /// Compute the dot product of two half-precision vectors on the GPU.
    ///
    /// Returns the resulting sum as a `u32`.
    pub fn dot_product(&self, a: &[f16], b: &[f16]) -> u32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length!");
        let array_len = a.len() as u32;

        autoreleasepool(|| {
            // 1. Create buffers for input/output
            let input_buffer_a = create_buffer(&self.device, a);
            let input_buffer_b = create_buffer(&self.device, b);
            let output_buffer = create_buffer(&self.device, &[0u32]);
            let arraylen_buffer = create_buffer(&self.device, &[array_len]);

            // 2. Create command buffer & encoder
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers
            encoder.set_compute_pipeline_state(&self.dot_product_pipeline);
            encoder.set_buffer(0, Some(&input_buffer_a), 0);
            encoder.set_buffer(1, Some(&input_buffer_b), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&arraylen_buffer), 0);

            // 4. Determine thread layout
            let num_threads = self.dot_product_pipeline.thread_execution_width();
            let threadgroup_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };
            // Round up to cover all elements
            let threadgroup_count = MTLSize {
                width: ((array_len as u64 + num_threads - 1) / num_threads) as u64,
                height: 1,
                depth: 1,
            };

            // 5. Allocate threadgroup memory (for reduction)
            encoder.set_threadgroup_memory_length(
                0,
                threadgroup_size.width * (mem::size_of::<f16>() as u64),
            );

            // 6. Encode & dispatch
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 7. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 8. Read result
            let ptr = output_buffer.contents() as *mut u32;
            unsafe { *ptr }
        })
    }

    /// Returns a `row_len * col_len` vector of `f16`.
    /// Compute softmax function on input vector: exp(x_i)/sum(exp(x_j))
    /// Returns a new vector with softmax applied.
    pub fn softmax(&self, input: &[f16]) -> Vec<f16> {
        let array_len = input.len() as u32;

        autoreleasepool(|| {
            // 1. Create buffers
            let input_buffer = create_buffer(&self.device, input);
            let sum_buffer = create_buffer(&self.device, &[0.0f32]);
            let array_len_buffer = create_buffer(&self.device, &[array_len]);
            let output_buffer = create_buffer(
                // todo: this should be the number of thread groups
                &self.device,
                vec![f16::from_f32(0.0); array_len as usize].as_slice(),
            );

            // 2. Create command buffer & encoder for sum computation
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers for sum
            encoder.set_compute_pipeline_state(&self.softmax_sum_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&sum_buffer), 0);
            encoder.set_buffer(2, Some(&array_len_buffer), 0);

            // 4. Determine thread layout for sum (process 4 elements per thread)
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

            // 5. Allocate threadgroup memory for reduction
            encoder.set_threadgroup_memory_length(
                0,
                threadgroup_size.width * (mem::size_of::<f32>() as u64),
            );

            // 6. Encode & dispatch sum computation
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // Start a new encoder for the output computation
            let encoder = command_buffer.new_compute_command_encoder();

            // Set pipeline & buffers for output
            encoder.set_compute_pipeline_state(&self.softmax_output_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&sum_buffer), 0);
            encoder.set_buffer(2, Some(&array_len_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);

            // Use same thread layout for output computation
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 7. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 8. Read results
            let ptr = output_buffer.contents() as *const f16;
            let output_slice = unsafe { std::slice::from_raw_parts(ptr, array_len as usize) };
            output_slice.to_vec()
        })
    }

    /// Returns a new vector with ReLU applied (max(0, x) for each element).
    /// Multiply two vectors element-wise
    pub fn vector_multiply(&self, a: &[f16], b: &[f16]) -> Vec<f16> {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length!");
        let array_len = a.len() as u32;

        autoreleasepool(|| {
            // 1. Create buffers
            let input_buffer_a = create_buffer(&self.device, a);
            let input_buffer_b = create_buffer(&self.device, b);
            let output_buffer = create_buffer(
                &self.device,
                vec![f16::from_f32(0.0); array_len as usize].as_slice(),
            );
            let array_len_buffer = create_buffer(&self.device, &[array_len]);

            // 2. Create command buffer & encoder
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers
            encoder.set_compute_pipeline_state(&self.vector_multiply_pipeline);
            encoder.set_buffer(0, Some(&input_buffer_a), 0);
            encoder.set_buffer(1, Some(&input_buffer_b), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&array_len_buffer), 0);

            // 4. Determine thread layout (process 4 elements per thread)
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

            // 5. Encode & dispatch
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 6. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 7. Read results
            let ptr = output_buffer.contents() as *const f16;
            let output_slice = unsafe { std::slice::from_raw_parts(ptr, array_len as usize) };
            output_slice.to_vec()
        })
    }

    pub fn relu(&self, input: &[f16]) -> Vec<f16> {
        let array_len = input.len() as u32;

        autoreleasepool(|| {
            // 1. Create buffers
            let input_buffer = create_buffer(&self.device, input);
            let output_buffer = create_buffer(
                &self.device,
                vec![f16::from_f32(0.0); array_len as usize].as_slice(),
            );
            let array_len_buffer = create_buffer(&self.device, &[array_len]);

            // 2. Create command buffer & encoder
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers
            encoder.set_compute_pipeline_state(&self.relu_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_buffer(2, Some(&array_len_buffer), 0);

            // 4. Determine thread layout (process 4 elements per thread)
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

            // 5. Encode & dispatch
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 6. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 7. Read results
            let ptr = output_buffer.contents() as *const f16;
            let output_slice = unsafe { std::slice::from_raw_parts(ptr, array_len as usize) };
            output_slice.to_vec()
        })
    }

    /// Returns a new vector with 1.0 for positive values and 0.0 for non-positive values.
    pub fn positive_indicator(&self, input: &[f16]) -> Vec<f16> {
        let array_len = input.len() as u32;

        autoreleasepool(|| {
            // 1. Create buffers
            let input_buffer = create_buffer(&self.device, input);
            let output_buffer = create_buffer(
                &self.device,
                vec![f16::from_f32(0.0); array_len as usize].as_slice(),
            );
            let array_len_buffer = create_buffer(&self.device, &[array_len]);

            // 2. Create command buffer & encoder
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers
            encoder.set_compute_pipeline_state(&self.positive_indicator_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_buffer(2, Some(&array_len_buffer), 0);

            // 4. Determine thread layout (process 4 elements per thread)
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

            // 5. Encode & dispatch
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 6. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 7. Read results
            let ptr = output_buffer.contents() as *const f16;
            let output_slice = unsafe { std::slice::from_raw_parts(ptr, array_len as usize) };
            output_slice.to_vec()
        })
    }

    /// Multiply two matrices (A of size row_len x inner_len, and B of size inner_len x col_len)
    /// stored in row-major order. Both inputs are `Vec<f16>`; output is `Vec<f16>`
    pub fn matrix_multiply(
        &self,
        a: &[f16],
        b: &[f16],
        row_len: u32,
        inner_len: u32,
        col_len: u32,
        a_transposed: bool,
        b_transposed: bool,
        thread_count: u32, // usually, 32 is the right cfg here
    ) -> Vec<f16> {
        // Sanity checks
        assert_eq!(
            a.len() as u32,
            row_len * inner_len,
            "Dimensions of A are incorrect."
        );
        assert_eq!(
            b.len() as u32,
            inner_len * col_len,
            "Dimensions of B are incorrect."
        );

        let out_len = row_len * col_len;

        autoreleasepool(|| {
            // 1. Create buffers
            let tile_size = thread_count;
            let input_buffer_a = create_buffer(&self.device, a);
            let input_buffer_b = create_buffer(&self.device, b);
            let output_buffer = create_buffer(
                &self.device,
                vec![f16::from_f32(0.0); out_len as usize].as_slice(),
            );
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let inner_len_buffer = create_buffer(&self.device, &[inner_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);
            let tile_size_buffer = create_buffer(&self.device, &[tile_size]);
            let a_transposed_buffer = create_buffer(&self.device, &[a_transposed]);
            let b_transposed_buffer = create_buffer(&self.device, &[b_transposed]);

            // 2. Create command buffer & encoder
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // 3. Set pipeline & buffers
            encoder.set_compute_pipeline_state(&self.matrix_multiply_pipeline);
            encoder.set_buffer(0, Some(&input_buffer_a), 0);
            encoder.set_buffer(1, Some(&input_buffer_b), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&row_len_buffer), 0);
            encoder.set_buffer(4, Some(&inner_len_buffer), 0);
            encoder.set_buffer(5, Some(&col_len_buffer), 0);
            encoder.set_buffer(6, Some(&a_transposed_buffer), 0);
            encoder.set_buffer(7, Some(&b_transposed_buffer), 0);
            encoder.set_buffer(8, Some(&tile_size_buffer), 0);

            // 4. Determine thread layout
            //    We'll dispatch (row_len x col_len) threadgroups, each having 'inner_len' threads.
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
            // 5. Allocate threadgroup memory
            //    First buffer: Tile of matrix A (tile_size rows x inner_len columns)
            encoder.set_threadgroup_memory_length(
                0,
                (tile_size * tile_size) as u64 * (mem::size_of::<f16>() as u64),
            );
            //    Second buffer: Tile of matrix B (inner_len rows x tile_size columns)
            encoder.set_threadgroup_memory_length(
                1,
                (tile_size * tile_size) as u64 * (mem::size_of::<f16>() as u64),
            );

            // 6. Encode & dispatch
            encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            encoder.end_encoding();

            // 7. Commit & wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 8. Read results
            let ptr = output_buffer.contents() as *const f16;
            let output_slice = unsafe { std::slice::from_raw_parts(ptr, out_len as usize) };
            output_slice.to_vec()
        })
    }
}

/// A helper to create a Metal buffer from a slice of data.
/// Uses the correct byte size for `T`.
fn create_buffer<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let size = (data.len() * std::mem::size_of::<T>()) as u64;
    let raw_ptr = data.as_ptr() as *const std::ffi::c_void;
    device.new_buffer_with_data(raw_ptr, size, MTLResourceOptions::CPUCacheModeDefaultCache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// CPU implementation of matrix multiplication for comparison.
    /// Single core
    fn cpu_matrix_multiply(
        a: &[f16],
        b: &[f16],
        row_len: u32,
        inner_len: u32,
        col_len: u32,
        a_transposed: bool,
        b_transposed: bool,
    ) -> Vec<f16> {
        let mut result = vec![f16::from_f32(0.0); (row_len * col_len) as usize];

        for i in 0..row_len {
            for j in 0..col_len {
                let mut sum = f16::from_f32(0.0);
                for k in 0..inner_len {
                    let a_idx = if a_transposed {
                        (k * row_len + i) as usize
                    } else {
                        (i * inner_len + k) as usize
                    };
                    let b_idx = if b_transposed {
                        (j * inner_len + k) as usize
                    } else {
                        (k * col_len + j) as usize
                    };
                    sum = sum + a[a_idx] * b[b_idx];
                }
                result[(i * col_len + j) as usize] = sum;
            }
        }

        result
    }

    #[test]
    fn matrix_multiply() {
        // 1) Create the shared Metal context
        let context = MetalContext::new("compute-kernel.metallib");

        // 2) Configure some matrix sizes that you want to test
        let row_len = 258;
        let inner_len = 256;
        let col_len = 259;

        // 3) Create some test data
        let mat_a = vec![f16::from_f32(2.0); (row_len * inner_len) as usize];
        let mat_b = vec![f16::from_f32(4.0); (inner_len * col_len) as usize];

        // 4) Run GPU computation and verify accuracy
        let gpu_result = context.matrix_multiply(
            &mat_a,
            &mat_b,
            row_len,
            inner_len,
            col_len,
            false,
            false,
            context.matrix_multiply_pipeline.thread_execution_width() as u32,
        );
        // Verify GPU result against CPU result
        let cpu_result =
            cpu_matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len, false, false);
        assert_eq!(
            gpu_result.len(),
            cpu_result.len(),
            "GPU and CPU results have different lengths"
        );

        // Allow small floating-point differences due to precision
        for (gpu_val, cpu_val) in gpu_result.iter().zip(cpu_result.iter()) {
            assert!(
                (gpu_val.to_f32() - cpu_val.to_f32()).abs() < 1e-3,
                "GPU and CPU results differ significantly\nGPU result (incorrect): {:?}\nCPU result (correct): {:?}",
                gpu_result,
                cpu_result
            );
        }
    }

    #[test]
    fn matrix_multiply_transposed() {
        // 1) Create the shared Metal context
        let context = MetalContext::new("compute-kernel.metallib");

        // 2) Configure some matrix sizes that you want to test
        let row_len = 5;
        let inner_len = 5;
        let col_len = 5;

        // 3) Create some test data
        let mat_a = (0..(row_len * inner_len))
            .map(|i| f16::from_f32((i % 5) as f32))
            .collect::<Vec<_>>();
        let mat_b = (0..(inner_len * col_len))
            .map(|i| f16::from_f32((i % 5) as f32))
            .collect::<Vec<_>>();

        // 4) Run GPU computation and verify accuracy
        let gpu_result = context.matrix_multiply(
            &mat_a,
            &mat_b,
            row_len,
            inner_len,
            col_len,
            false,
            true,
            context.matrix_multiply_pipeline.thread_execution_width() as u32,
        );
        // Verify GPU result against CPU result
        let cpu_result =
            cpu_matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len, false, true);
        assert_eq!(
            gpu_result.len(),
            cpu_result.len(),
            "GPU and CPU results have different lengths"
        );

        // Allow small floating-point differences due to precision
        for (gpu_val, cpu_val) in gpu_result.iter().zip(cpu_result.iter()) {
            assert!(
                (gpu_val.to_f32() - cpu_val.to_f32()).abs() < 1e-3,
                "GPU and CPU results differ significantly\nGPU result (incorrect): {:?}\nCPU result (correct): {:?}",
                gpu_result,
                cpu_result
            );
        }
    }

    #[test]
    fn matrix_addition() {
        let context = MetalContext::new("compute-kernel.metallib");

        // Test matrix 3x3
        let row_len = 3;
        let col_len = 3;
        let a: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();
        let b: Vec<f16> = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        let c_a = f16::from_f32(2.0); // multiply first matrix by 2
        let c_b = f16::from_f32(3.0); // multiply second matrix by 3

        let result = context.matrix_addition(&a, &b, c_a, c_b, row_len, col_len);

        // Expected results: 2*A + 3*B
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
        .map(|x| f16::from_f32(x))
        .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(
                (got.to_f32() - want.to_f32()).abs() < 1e-3,
                "Matrix addition results differ: got {:?}, want {:?}",
                got,
                want
            );
        }
    }

    #[test]
    fn matrix_multiply_constant() {
        let context = MetalContext::new("compute-kernel.metallib");

        // Test matrix 3x3
        let row_len = 3;
        let col_len = 3;
        let input: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        let constant = f16::from_f32(2.0);

        let result = context.matrix_multiply_constant(&input, constant, row_len, col_len);

        // Expected results after multiplication by 2
        let expected: Vec<f16> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(
                (got.to_f32() - want.to_f32()).abs() < 1e-3,
                "Matrix multiply constant results differ: got {:?}, want {:?}",
                got,
                want
            );
        }
    }

    /// CPU implementation of softmax for testing
    fn cpu_softmax(input: &[f16]) -> Vec<f16> {
        // First pass: compute exp and sum
        let exp_values: Vec<f32> = input.iter().map(|x| x.to_f32().exp()).collect();
        let sum: f32 = exp_values.iter().sum();

        // Second pass: divide by sum
        exp_values.iter().map(|x| f16::from_f32(x / sum)).collect()
    }

    #[test]
    fn softmax() {
        let context = MetalContext::new("compute-kernel.metallib");

        // Test data with a mix of positive and negative values
        let input: Vec<f16> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        let result = context.softmax(&input);
        let expected = cpu_softmax(&input);

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(
                (got.to_f32() - want.to_f32()).abs() < 1e-3,
                "Softmax results differ significantly: got {:?}, want {:?}",
                got,
                want
            );
        }

        // Verify the sum of probabilities is approximately 1
        let sum: f32 = result.iter().map(|x| x.to_f32()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "Sum of softmax probabilities should be 1, got {}",
            sum
        );
    }

    #[test]
    fn positive_indicator() {
        let context = MetalContext::new("compute-kernel.metallib");

        // Test data with positive and negative values
        let input: Vec<f16> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        let result = context.positive_indicator(&input);

        // Expected results: 1.0 for positive values, 0.0 for non-positive values
        let expected: Vec<f16> = vec![0.0, 0.0, 0.0, 1.0, 1.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(
                (got.to_f32() - want.to_f32()).abs() < 1e-3,
                "Positive indicator results differ: got {:?}, want {:?}",
                got,
                want
            );
        }
    }

    #[test]
    fn vector_multiply() {
        let context = MetalContext::new("compute-kernel.metallib");

        // Test vectors with various values
        let a: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();
        let b: Vec<f16> = vec![2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        let result = context.vector_multiply(&a, &b);

        // Expected results after element-wise multiplication
        let expected: Vec<f16> = vec![2.0, 6.0, 12.0, 20.0, 30.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(
                (got.to_f32() - want.to_f32()).abs() < 1e-3,
                "Vector multiply results differ: got {:?}, want {:?}",
                got,
                want
            );
        }
    }

    #[test]
    fn relu() {
        let context = MetalContext::new("compute-kernel.metallib");

        // Test data with positive and negative values
        let input: Vec<f16> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        let result = context.relu(&input);

        // Expected results after ReLU
        let expected: Vec<f16> = vec![0.0, 0.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();

        assert_eq!(result.len(), expected.len());
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(
                (got.to_f32() - want.to_f32()).abs() < 1e-3,
                "ReLU results differ: got {:?}, want {:?}",
                got,
                want
            );
        }
    }

    #[test]
    fn benchmark_matrix_multiply() {
        // 1) Create the shared Metal context
        let context = MetalContext::new("compute-kernel.metallib");

        // 2) Configure some matrix sizes that you want to test
        let row_len = 1024;
        let inner_len = 1024;
        let col_len = 1024;

        // 3) Create some test data
        let mat_a = vec![f16::from_f32(2.0); (row_len * inner_len) as usize];
        let mat_b = vec![f16::from_f32(4.0); (inner_len * col_len) as usize];

        // For the GFLOPS calculation, each output element requires `inner_len` multiply-add pairs
        // => 2 * inner_len ops per output element.
        let total_ops = 2.0 * row_len as f64 * col_len as f64 * inner_len as f64;

        // ensure GPU is "warmed up"
        for _ in 0..2 {
            let _ = context.matrix_multiply(
                &mat_a,
                &mat_b,
                row_len,
                inner_len,
                col_len,
                false,
                false,
                context.matrix_multiply_pipeline.thread_execution_width() as u32,
            );
        }

        let iterations = 5;

        let gpu_start = Instant::now();
        for _ in 0..iterations {
            //let _ = context.matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len);
            let _ = context.matrix_multiply(
                &mat_a,
                &mat_b,
                row_len,
                inner_len,
                col_len,
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
        "GPU: ran {iterations} multiplies of size {row_len}x{inner_len} * {inner_len}x{col_len} \
         in {:#?} total; ~{:#?} each => approx. {:.2} GFLOPS",
        gpu_total_time, gpu_avg_time, gpu_gflops
        );

        // assert GPU GFLOPs >100
        assert!(
            gpu_gflops > 100.0,
            "GPU performance should exceed 100 GFLOPS"
        );

        // 6) CPU Benchmark
        let cpu_start = Instant::now();
        let _ = cpu_matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len, false, false);
        let cpu_total_time = cpu_start.elapsed();
        let cpu_avg_time_s = cpu_start.elapsed().as_secs_f64();
        let cpu_gflops = (total_ops / cpu_avg_time_s) / 1e9;

        println!(
            "CPU: ran 1 multiplies of size {row_len}x{inner_len} * {inner_len}x{col_len} \
         in {:#?} total; ~{:#?} each => approx. {:.2} GFLOPS",
            cpu_total_time, cpu_total_time, cpu_gflops
        );

        // 7) Print speedup (how many times faster GPU is than CPU)
        let speedup = cpu_avg_time_s / gpu_avg_time_s;
        println!(
            "GPU is about {:.2}x faster than CPU for this problem size.",
            speedup
        );
    }
}
