use dataframe::*;
use model::*;
use std::error::Error;

const INPUT_CSV_FILE: &str = "data/heart_statlog_cleveland_hungary_final.csv";

fn main() -> Result<(), Box<dyn Error>> {
    println!("starting!");

    let mut df = DataFrame::from_file(INPUT_CSV_FILE)?;
    df.log(5);
    df = df
        .expand_categorical("expanded", vec![1, 2, 5, 6, 8, 10])?
        .normalize("expanded_and_normalized", vec![0, 3, 4, 7, 9])?;
    df.log(15);
    df.write();

    let mut schema = ModelSchema::new(df.col_size - 1); // remove target col
    println!("{}", schema);

    let fc1 = Mask::fully_connected(schema.current_output_size(), 8);
    schema.add_layer(fc1)?;
    println!("{}", schema);
    let fc2 = Mask::fully_connected(schema.current_output_size(), 5);
    schema.add_layer(fc2)?;
    println!("{}", schema);

    println!("done!");

    Ok(())
}
