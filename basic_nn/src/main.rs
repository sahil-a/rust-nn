use compute::*;
use dataframe::*;
use model::*;
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
    let input_size = df.col_size - TARGET_LEN;

    let model = ModelBuilder::input_size(input_size)
        .layer(Box::new(FullyConnectedLayer::new(input_size, 20, true)))?
        .layer(Box::new(FullyConnectedLayer::new(20, 10, true)))?
        .layer(Box::new(FullyConnectedLayer::new(10, 20, true)))?
        .layer(Box::new(FullyConnectedLayer::new(20, TARGET_LEN, false)))?
        .loss_fn(Box::new(CrossEntropyLoss::new(TARGET_LEN)))?;

    let batch_size = 32;
    let num_batches = (df.get_data_segment_size(DataSegment::Train) + batch_size - 1) / batch_size;
    let output_buffer = GPUBuffer::new(TARGET_LEN, 1);

    for epoch in 1..=100 {
        println!("starting epoch {}", epoch);
        for batch_num in 0..num_batches {
            let batch = df.get_batch(batch_num, batch_size, DataSegment::Train);
            for item in batch {
                let (data, labels) = item.split_at(item.len() - TARGET_LEN);
                let input_buffer = GPUBuffer::from_vec(data.len(), 1, data);
                let target_buffer = GPUBuffer::from_vec(TARGET_LEN, 1, labels);
                model.train_step(&input_buffer, &output_buffer, &target_buffer);
            }
        }
    }

    println!("done!");

    Ok(())
}
