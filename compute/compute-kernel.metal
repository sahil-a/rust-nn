#include <metal_stdlib>

using namespace metal;

// to compile (included in build.rs): 
// xcrun metal -c sumshader.metal -o sumshader.air
// && xcrun metallib sumshader.air -o sumshader.metallib


// 2 3 5 1 6 9 2 2 9 (example which terminates at stride=32)
// 5 3 6 1 15 9 4 2 9 (stride = 2)
// 11 3 6 1 19 9 4 2 9 (stride = 4)
// 30 3 6 1 19 9 4 2 9 (stride = 8)
// 39 3 6 1 19 9 4 2 9 (stride = 16)
kernel void sum_parallel(const device half *data [[ buffer(0) ]], 
                            volatile device atomic_uint *sum [[ buffer(1) ]],
                            const device uint *array_len [[ buffer(2) ]],
                            uint gid [[ thread_position_in_grid ]],
                            uint tid [[ threadgroup_position_in_grid ]],
                            uint lid [[ thread_position_in_threadgroup ]],
                            uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                            //uint simd_per_threadgroup [[ simdgroups_per_threadgroup ]],
                            threadgroup half *shared_mem [[ threadgroup(0) ]])
{
    // this thread group should load all data
    if (gid < *array_len) {
        shared_mem[lid] = data[gid]; // for dot product, you would multiply here
    } else {
        shared_mem[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduction within each threadgroup
    for (uint stride = 2; stride/2 < threads_per_threadgroup; stride <<= 1) {
        if (lid % stride == 0 && (lid + stride/2 < threads_per_threadgroup)) {
            shared_mem[lid] += shared_mem[lid + stride/2];
        }
        // synchronization needed per stride
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // write the final result to the output
    if (lid == 0) {
        atomic_fetch_add_explicit(sum, (uint)shared_mem[0], memory_order_relaxed);
    }
}

// same as above, with one line change for multiplication
kernel void dot_product(const device half *a [[ buffer(0) ]], 
                            const device half *b [[ buffer(1) ]],
                            volatile device atomic_uint *output [[ buffer(2) ]],
                            const device uint *array_len [[ buffer(3) ]],
                            uint gid [[ thread_position_in_grid ]],
                            uint tid [[ threadgroup_position_in_grid ]],
                            uint lid [[ thread_position_in_threadgroup ]],
                            uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                            //uint simd_per_threadgroup [[ simdgroups_per_threadgroup ]],
                            threadgroup half *shared_mem [[ threadgroup(0) ]])
{
    // this thread group should load all data
    if (gid < *array_len) {
        shared_mem[lid] = a[gid] * b[gid];
    } else {
        shared_mem[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduction within each threadgroup
    for (uint stride = 2; stride/2 < threads_per_threadgroup; stride <<= 1) {
        if (lid % stride == 0 && (lid + stride/2 < threads_per_threadgroup)) {
            shared_mem[lid] += shared_mem[lid + stride/2];
        }
        // synchronization needed per stride
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // add the final result to the output
    if (lid == 0) {
        atomic_fetch_add_explicit(output, (uint)shared_mem[0], memory_order_relaxed);
    }
}

kernel void relu(const device half *a [[ buffer(0) ]],
                device half *output [[ buffer(1) ]],
                const device uint *array_len [[ buffer(2) ]],
                uint gid [[ thread_position_in_grid ]]) {
    uint base_idx = gid * 4;
    if (base_idx + 3 < *array_len) {
        // All 4 elements are within bounds - use vectorized operation
        half4 input = half4(a[base_idx], a[base_idx + 1], a[base_idx + 2], a[base_idx + 3]);
        half4 result = max(input, 0.0h);
        output[base_idx] = result.x;
        output[base_idx + 1] = result.y;
        output[base_idx + 2] = result.z;
        output[base_idx + 3] = result.w;
    } else {
        for (uint i = 0; i < 4 && base_idx + i < *array_len; i++) {
            output[base_idx + i] = max(a[base_idx + i], 0.0h);
        }
    }
}

kernel void vector_pairwise_multiply(
                const device half *a [[ buffer(0) ]],
                const device half *b [[ buffer(1) ]],
                device half *output [[ buffer(2) ]],
                const device uint *array_len [[ buffer(3) ]],
                uint gid [[ thread_position_in_grid ]]) {
    uint base_idx = gid * 4;
    if (base_idx + 3 < *array_len) {
        // All 4 elements are within bounds - use vectorized operation
        half4 input_a = half4(a[base_idx], a[base_idx + 1], a[base_idx + 2], a[base_idx + 3]);
        half4 input_b = half4(b[base_idx], b[base_idx + 1], b[base_idx + 2], b[base_idx + 3]);
        half4 result = input_a * input_b;
        output[base_idx] = result.x;
        output[base_idx + 1] = result.y;
        output[base_idx + 2] = result.z;
        output[base_idx + 3] = result.w;
    } else {
        for (uint i = 0; i < 4 && base_idx + i < *array_len; i++) {
            output[base_idx + i] = a[base_idx + i] * b[base_idx + i];
        }
    }
}

kernel void positive_indicator(
    const device half *a [[ buffer(0) ]],
    device half *output [[ buffer(1) ]],
    const device uint *array_len [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint base_idx = gid * 4;
    if (base_idx + 3 < *array_len) {
        // Vectorized operation with correct mask conversion
        half4 input = half4(a[base_idx], a[base_idx + 1], a[base_idx + 2], a[base_idx + 3]);
        half4 result = select(half4(0.0h), half4(1.0h), input > 0.0h);
        output[base_idx] = result.x;
        output[base_idx + 1] = result.y;
        output[base_idx + 2] = result.z;
        output[base_idx + 3] = result.w;
    } else {
        // Scalar operation remains unchanged
        for (uint i = 0; i < 4 && base_idx + i < *array_len; i++) {
            output[base_idx + i] = half(a[base_idx + i] > 0.0h) ? 1.0h : 0.0h;
        }
    }
}

half exp_approx(half x) {
    return pow(M_E_H, x);
}

half4 exp_approx(half4 x) {
    return pow(half4(M_E_H), x);
}

kernel void softmax_sum(const device half *a [[ buffer(0) ]],
                    volatile device atomic_float *sum [[ buffer(1) ]],
                    const device uint *array_len [[ buffer(2) ]],
                    uint gid [[ thread_position_in_grid ]],
                    uint tid [[ threadgroup_position_in_grid ]],
                    uint lid [[ thread_position_in_threadgroup ]],
                    uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                    threadgroup float *shared_mem [[ threadgroup(0) ]]) {
    uint base_idx = gid * 4;
    if (base_idx + 3 < *array_len) {
        half4 input = half4(a[base_idx], a[base_idx + 1], a[base_idx + 2], a[base_idx + 3]);
        float4 result = float4(exp_approx(input));
        shared_mem[lid] = result.x + result.y + result.z + result.w;
    } else {
        half sum = 0;
        for (uint i = 0; i < 4 && base_idx + i < *array_len; i++) {
            sum += exp_approx(a[base_idx + i]);
        }
        shared_mem[lid] = float(sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 2; stride/2 < threads_per_threadgroup; stride <<= 1) {
        if (lid % stride == 0 && (lid + stride/2 < threads_per_threadgroup)) {
            shared_mem[lid] += shared_mem[lid + stride/2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        atomic_fetch_add_explicit(sum, shared_mem[0], memory_order_relaxed);
    }
}

kernel void softmax_output(const device half *a [[ buffer(0) ]],
                    const device float *sum [[ buffer(1) ]],
                    const device uint *array_len [[ buffer(2) ]],
                    device half *output [[ buffer(3) ]],
                    uint gid [[ thread_position_in_grid ]],
                    uint tid [[ threadgroup_position_in_grid ]],
                    uint lid [[ thread_position_in_threadgroup ]],
                    uint threads_per_threadgroup [[ threads_per_threadgroup ]],
                    threadgroup half *shared_mem [[ threadgroup(0) ]]) {
    half denominator = half(*sum);
    uint base_idx = gid * 4;

    if (base_idx + 3 < *array_len) {
        // All 4 elements are within bounds - use vectorized operation
        half4 input = half4(a[base_idx], a[base_idx + 1], a[base_idx + 2], a[base_idx + 3]);
        half4 result = exp_approx(input) / denominator;
        output[base_idx] = result.x;
        output[base_idx + 1] = result.y;
        output[base_idx + 2] = result.z;
        output[base_idx + 3] = result.w;
    } else {
        for (uint i = 0; i < 4 && base_idx + i < *array_len; i++) {
            output[base_idx + i] = exp_approx(a[base_idx + i]) / denominator;
        }
    }
}

kernel void matrix_multiply_constant(const device half *a [[ buffer(0) ]],
                            const device half *c [[ buffer(1) ]],
                            device half *output [[buffer(2) ]],
                            const device uint *row_len [[ buffer(3) ]],
                            const device uint *col_len [[ buffer(4) ]],
                            uint2 gid [[ thread_position_in_grid ]]) {
    if (gid.x < *row_len) {
        uint base_y = gid.y * 4;
        if (base_y + 3 < *col_len) {
            half4 input = half4(a[gid.x * *col_len + base_y], a[gid.x * *col_len + base_y + 1], a[gid.x * *col_len + base_y + 2], a[gid.x * *col_len + base_y + 3]);
            half4 result = *c * input;
            output[gid.x * *col_len + base_y] = result.x;
            output[gid.x * *col_len + base_y + 1] = result.y;
            output[gid.x * *col_len + base_y + 2] = result.z;
            output[gid.x * *col_len + base_y + 3] = result.w;
        } else {
            for (uint i = 0; i < 4 && base_y + i < *col_len; i++) {
                output[gid.x * *col_len + base_y + i] = a[gid.x * *col_len + base_y + i] * *c;
            }
        }
    }
}
                    

kernel void matrix_multiply(const device half *a [[ buffer(0) ]], 
                            const device half *b [[ buffer(1) ]],
                            device half *output [[ buffer(2) ]],
                            const device uint *row_len [[ buffer(3) ]],
                            const device uint *inner_len [[ buffer(4) ]],
                            const device uint *col_len [[ buffer(5) ]],
                            const device bool *a_transposed [[ buffer(6) ]],
                            const device bool *b_transposed [[ buffer(7) ]],
                            const device uint *tile_size [[ buffer(8) ]],
                            uint2 tid [[ threadgroup_position_in_grid ]],
                            uint2 lid [[ thread_position_in_threadgroup ]],
                            uint2 threads_per_threadgroup [[ threads_per_threadgroup ]],
                            // the below needs to be set to (tile_size**2) length in rs
                            threadgroup half *shared_mem_a [[ threadgroup(0) ]],
                            // the below needs to be set to (tile_size**2) length in rs
                            threadgroup half *shared_mem_b [[ threadgroup(1) ]])
{
    half sum = 0;
    uint x = tid.x * *tile_size + lid.x;
    uint y = tid.y * *tile_size + lid.y;
    for (uint inner_start = 0; inner_start < *inner_len; inner_start += *tile_size) {
        // 1. Load all rows of A and cols of B for this tile
        if (x < *row_len) { // load row x (uniform condition)
            if (lid.y+inner_start < *inner_len) {
                // load element of row x of A into shared mem
                if (*a_transposed) {
                    shared_mem_a[*tile_size * lid.x + lid.y] = a[(lid.y + inner_start) * *inner_len + x];
                } else {
                    shared_mem_a[*tile_size * lid.x + lid.y] = a[x * *inner_len + lid.y + inner_start];
                }
            }
        }
        if (y < *col_len) { // load col y (uniform condition)
            if (lid.x+inner_start < *inner_len) {
                // load element of col y of B into shared mem
                if (*b_transposed) {
                    shared_mem_b[*tile_size * lid.y + lid.x] = b[y * *col_len + lid.x + inner_start];
                } else {
                    shared_mem_b[*tile_size * lid.y + lid.x] = b[(lid.x+inner_start) * *col_len + y];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);


        // Compute output
       if (x < *row_len && y < *col_len) {
            half4 sum4 = half4(0.0h);
            uint i;
            for (i = 0; i + 3 < *tile_size && inner_start + i + 3 < *inner_len; i += 4) {
                half4 a4(shared_mem_a[*tile_size * lid.x + i],
                        shared_mem_a[*tile_size * lid.x + i + 1],
                        shared_mem_a[*tile_size * lid.x + i + 2],
                        shared_mem_a[*tile_size * lid.x + i + 3]);
                half4 b4(shared_mem_b[*tile_size * lid.y + i],
                        shared_mem_b[*tile_size * lid.y + i + 1],
                        shared_mem_b[*tile_size * lid.y + i + 2],
                        shared_mem_b[*tile_size * lid.y + i + 3]);
                sum4 += a4 * b4;
            }
            sum += sum4.x + sum4.y + sum4.z + sum4.w;
            
            // Handle remaining elements
            for (; i < *tile_size; i++) {
                if (inner_start + i < *inner_len) {
                    sum += shared_mem_a[*tile_size * lid.x + i] * shared_mem_b[*tile_size * lid.y + i];
                }
            }
        }
    }
    if (x < *row_len && y < *col_len) {
        output[x * *col_len + y] = sum;
    }
}
