use compute::*;
use dataframe::*;
use half::f16;
use model::*;
use optimizer::*;
use std::error::Error;

const INPUT_CSV_FILE: &str = "data/heart_statlog_cleveland_hungary_final.csv";

fn main() -> Result<(), Box<dyn Error>> {
    println!("starting!");
    initialize_metal_context_from("compute/compute-kernel.metallib");

    let mut df = DataFrame::from_file(INPUT_CSV_FILE)?;
    df.log(5);
    df = df
        .expand_categorical("expanded", vec![1, 2, 5, 6, 8, 10, 11])?
        .normalize("expanded_and_normalized", vec![0, 3, 4, 7, 9])?;
    df.log(15);
    df.write();

    const TARGET_LEN: usize = 2;
    let learning_rate: f16 = f16::from_f32(0.05);
    let input_size = df.col_size - TARGET_LEN;

    let model = ModelBuilder::input_size(input_size)
        .layer(Box::new(FullyConnectedLayer::new(input_size, 5, true)))?
        .layer(Box::new(FullyConnectedLayer::new(5, 5, true)))?
        .layer(Box::new(FullyConnectedLayer::new(5, TARGET_LEN, false)))?
        .loss_fn(Box::new(CrossEntropyLoss::new(TARGET_LEN)))?;
    let mut optimizer = FixedLearningRateOptimizer::new(&model, learning_rate);

    let batch_size = 32;
    let num_batches = (df.get_data_segment_size(&DataSegment::Train) + batch_size - 1) / batch_size;
    let output_buffer = GPUBuffer::new(TARGET_LEN, 1);
    df.train_val_test_split(60, 20, 20);

    let val_accuracy = calculate_accuracy(&df, DataSegment::Val, &model, &output_buffer);
    let train_accuracy = calculate_accuracy(&df, DataSegment::Train, &model, &output_buffer);
    println!(
        "epoch 0: val accuracy {:.2}, train accuracy {:.2}",
        val_accuracy, train_accuracy
    );

    for epoch in 1..=100 {
        df.shuffle(&DataSegment::Train)?;
        for batch_num in 0..num_batches {
            let batch = df.get_batch(batch_num, batch_size, &DataSegment::Train);
            for row in batch {
                let (data, labels) = row.split_at(row.len() - TARGET_LEN);
                let input_buffer = GPUBuffer::from_vec(data.len(), 1, data);
                let target_buffer = GPUBuffer::from_vec(TARGET_LEN, 1, labels);
                model.train_step(&input_buffer, &output_buffer, &target_buffer);
                optimizer.process_step();
            }
            optimizer.update_gradients();
        }

        let train_accuracy = calculate_accuracy(&df, DataSegment::Train, &model, &output_buffer);
        let val_accuracy = calculate_accuracy(&df, DataSegment::Val, &model, &output_buffer);
        println!(
            "epoch {}: val accuracy {:.2}, train accuracy {:.2}",
            epoch, val_accuracy, train_accuracy
        );
    }

    let test_accuracy = calculate_accuracy(&df, DataSegment::Test, &model, &output_buffer);
    println!("test accuracy {:.2}", test_accuracy);

    println!("done!");

    Ok(())
}

fn calculate_accuracy(
    df: &DataFrame,
    segment: DataSegment,
    model: &Model,
    output_buffer: &GPUBuffer,
) -> f32 {
    let compute = get_metal_context();
    let mut correct = 0;
    for row in df.get_segment(&segment) {
        let (data, labels) = row.split_at(row.len() - output_buffer.rows);
        let input_buffer = GPUBuffer::from_vec(data.len(), 1, data);
        model.inference(&input_buffer, output_buffer);
        compute.softmax(output_buffer, output_buffer);
        if is_correct(&output_buffer.to_cpu_vec(), labels) {
            correct += 1;
        }
    }
    (correct as f32) * 100.0 / (df.get_data_segment_size(&segment) as f32)
}

fn is_correct(softmax_output: &[f16], target: &[f16]) -> bool {
    let mut softmax_max_index = 0;
    let mut softmax_max = softmax_output[0];
    let mut target_max_index = 0;
    let mut target_max = target[0];
    for i in 1..softmax_output.len() {
        if softmax_output[i] > softmax_max {
            softmax_max = softmax_output[i];
            softmax_max_index = i;
        }
        if target[i] > target_max {
            target_max = target[i];
            target_max_index = i;
        }
    }
    softmax_max_index == target_max_index
}
