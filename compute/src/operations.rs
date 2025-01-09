use crate::*;
use half::f16;
use objc::rc::autoreleasepool;
use std::mem;
use std::path::Path;
use std::path::PathBuf;

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
    ) {
        assert_eq!(input.rows, output.rows);
        assert_eq!(input.cols, output.cols);
        assert_eq!(row_factors.rows * row_factors.cols, input.rows);

        let row_len = input.rows as u32;
        let col_len = input.cols as u32;

        autoreleasepool(|| {
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.matrix_multiply_rowwise_pipeline);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&row_factors.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            encoder.set_buffer(3, Some(&row_len_buffer), 0);
            encoder.set_buffer(4, Some(&col_len_buffer), 0);

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
    pub fn dot_product(&self, a: &GPUBuffer, b: &GPUBuffer) -> f32 {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);

        let array_len = (a.rows * a.cols) as u32;

        autoreleasepool(|| {
            let output_buffer = create_buffer(&self.device, &[0.0f32]);
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

            let ptr = output_buffer.contents() as *mut f32;
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
    pub fn relu(&self, input: &GPUBuffer) {
        let array_len = (input.rows * input.cols) as u32;

        autoreleasepool(|| {
            let array_len_buffer = create_buffer(&self.device, &[array_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.relu_pipeline);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&array_len_buffer), 0);

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
    pub fn positive_indicator(&self, input: &GPUBuffer) {
        let array_len = (input.rows * input.cols) as u32;

        autoreleasepool(|| {
            let array_len_buffer = create_buffer(&self.device, &[array_len]);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.positive_indicator_pipeline);
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&array_len_buffer), 0);

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
    ) {
        let row_len = if a_transposed { a.cols } else { a.rows } as u32;
        let inner_len = if a_transposed { a.rows } else { a.cols } as u32;
        let col_len = if b_transposed { b.rows } else { b.cols } as u32;
        let a_stride = a.cols as u32;
        let b_stride = b.cols as u32;
        assert_eq!(output.rows as u32, row_len);
        assert_eq!(output.cols as u32, col_len);

        autoreleasepool(|| {
            let tile_size = self.matrix_multiply_pipeline.thread_execution_width() as u32;
            let row_len_buffer = create_buffer(&self.device, &[row_len]);
            let inner_len_buffer = create_buffer(&self.device, &[inner_len]);
            let col_len_buffer = create_buffer(&self.device, &[col_len]);
            let a_stride_buffer = create_buffer(&self.device, &[a_stride]);
            let b_stride_buffer = create_buffer(&self.device, &[b_stride]);
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
            encoder.set_buffer(6, Some(&a_stride_buffer), 0);
            encoder.set_buffer(7, Some(&b_stride_buffer), 0);
            encoder.set_buffer(8, Some(&a_transposed_buffer), 0);
            encoder.set_buffer(9, Some(&b_transposed_buffer), 0);
            encoder.set_buffer(10, Some(&tile_size_buffer), 0);

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
