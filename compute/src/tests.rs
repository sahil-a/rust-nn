#[cfg(test)]
mod tests {
    use crate::*;
    use half::f16;
    use std::time::Instant;

    fn cpu_matrix_multiply(
        a: &[f16],
        b: &[f16],
        row_len: usize,
        inner_len: usize,
        col_len: usize,
    ) -> Vec<f16> {
        let mut result = vec![f16::from_f32(0.0); row_len * col_len];
        for i in 0..row_len {
            for j in 0..col_len {
                let mut sum = f16::from_f32(0.0);
                for k in 0..inner_len {
                    let a_idx = i * inner_len + k;
                    let b_idx = k * col_len + j;
                    sum = sum + a[a_idx] * b[b_idx];
                }
                result[i * col_len + j] = sum;
            }
        }
        result
    }

    #[test]
    fn dot_product() {
        initialize_metal_context();
        let context = get_metal_context();

        let a: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b: Vec<f16> = vec![2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_a = GPUBuffer::from_vec(1, a.len(), &a);
        let gpu_b = GPUBuffer::from_vec(1, b.len(), &b);

        let result = context.dot_product(&gpu_a, &gpu_b);

        // Calculate expected result: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 70
        let expected = 70.0f32;

        assert!(
            (result - expected).abs() < 1e-3,
            "Dot product result {} did not match expected {}",
            result,
            expected
        );
    }

    #[test]
    fn matrix_multiply() {
        initialize_metal_context();
        let context = get_metal_context();
        let row_len = 258;
        let inner_len = 256;
        let col_len = 259;
        let mat_a = vec![f16::from_f32(2.0); (row_len * inner_len) as usize];
        let mat_b = vec![f16::from_f32(4.0); (inner_len * col_len) as usize];

        let gpu_a = GPUBuffer::from_vec(row_len as usize, inner_len as usize, &mat_a);
        let gpu_b = GPUBuffer::from_vec(inner_len as usize, col_len as usize, &mat_b);
        let gpu_out = GPUBuffer::new(row_len as usize, col_len as usize);

        context.matrix_multiply(&gpu_a, &gpu_b, &gpu_out, false, false);
        let result = gpu_out.to_cpu_vec();

        let cpu_result = cpu_matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len);

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
        initialize_metal_context();
        let context = get_metal_context();
        let row_len = 25;
        let inner_len = 26;
        let col_len = 30;

        let mat_a = (0..(row_len * inner_len))
            .map(|i| f16::from_f32((i % 5) as f32))
            .collect::<Vec<_>>();
        let mat_b = (0..(inner_len * col_len))
            .map(|i| f16::from_f32((i % 5) as f32))
            .collect::<Vec<_>>();

        let gpu_a = GPUBuffer::from_vec(inner_len as usize, row_len as usize, &mat_a);
        let gpu_b = GPUBuffer::from_vec(col_len as usize, inner_len as usize, &mat_b);
        let gpu_out = GPUBuffer::new(row_len as usize, col_len as usize);

        context.matrix_multiply(&gpu_a, &gpu_b, &gpu_out, true, true);
        let result = gpu_out.to_cpu_vec();

        let mut mat_a_t = vec![f16::from_f32(0.0); row_len * inner_len];
        let mut mat_b_t = vec![f16::from_f32(0.0); inner_len * col_len];

        for i in 0..inner_len {
            for j in 0..row_len {
                mat_a_t[j * inner_len + i] = mat_a[i * row_len + j];
            }
        }
        for i in 0..col_len {
            for j in 0..inner_len {
                mat_b_t[j * col_len + i] = mat_b[i * inner_len + j];
            }
        }

        let cpu_result = cpu_matrix_multiply(&mat_a_t, &mat_b_t, row_len, inner_len, col_len);

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
        initialize_metal_context();
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

        let gpu_a = GPUBuffer::from_vec(row_len, col_len, &a);
        let gpu_b = GPUBuffer::from_vec(row_len, col_len, &b);
        let gpu_out = GPUBuffer::new(row_len, col_len);

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
        initialize_metal_context();
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

        let gpu_input = GPUBuffer::from_vec(row_len, col_len, &input);
        let gpu_out = GPUBuffer::new(row_len, col_len);
        let gpu_row_factors = GPUBuffer::from_vec(row_multipliers.len(), 1, &row_multipliers);

        context.matrix_multiply_rowwise(&gpu_input, &gpu_row_factors, &gpu_out);
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
    }

    #[test]
    fn matrix_multiply_constant() {
        initialize_metal_context();
        let context = get_metal_context();
        let row_len = 3;
        let col_len = 3;
        let input: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_input = GPUBuffer::from_vec(row_len, col_len, &input);
        let gpu_out = GPUBuffer::new(row_len, col_len);
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
        initialize_metal_context();
        let context = get_metal_context();
        let input: Vec<f16> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_in = GPUBuffer::from_vec(1, input.len(), &input);
        let gpu_out = GPUBuffer::new(1, input.len());
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
        initialize_metal_context();
        let context = get_metal_context();
        let input: Vec<f16> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_in = GPUBuffer::from_vec(1, input.len(), &input);
        context.positive_indicator(&gpu_in);
        let result = gpu_in.to_cpu_vec();

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
        initialize_metal_context();
        let context = get_metal_context();
        let a: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b: Vec<f16> = vec![2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_a = GPUBuffer::from_vec(1, a.len(), &a);
        let gpu_b = GPUBuffer::from_vec(1, b.len(), &b);
        let gpu_out = GPUBuffer::new(1, 5);

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
        initialize_metal_context();
        let context = get_metal_context();
        let input: Vec<f16> = vec![-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let gpu_in = GPUBuffer::from_vec(1, 5, &input);
        context.relu(&gpu_in);
        let result = gpu_in.to_cpu_vec();

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
        initialize_metal_context();
        let context = get_metal_context();
        let row_len = 1024;
        let inner_len = 1024;
        let col_len = 1024;
        let mat_a = vec![f16::from_f32(2.0); (row_len * inner_len) as usize];
        let mat_b = vec![f16::from_f32(4.0); (inner_len * col_len) as usize];
        let total_ops = 2.0 * row_len as f64 * col_len as f64 * inner_len as f64;

        let gpu_a = GPUBuffer::from_vec(row_len, inner_len, &mat_a);
        let gpu_b = GPUBuffer::from_vec(inner_len, col_len, &mat_b);
        let gpu_out = GPUBuffer::new(row_len, col_len);

        for _ in 0..2 {
            context.matrix_multiply(&gpu_a, &gpu_b, &gpu_out, false, false);
        }

        let iterations = 5;
        let gpu_start = Instant::now();
        for _ in 0..iterations {
            context.matrix_multiply(&gpu_a, &gpu_b, &gpu_out, false, false);
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
        let _ = cpu_matrix_multiply(&mat_a, &mat_b, row_len, inner_len, col_len);
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
