use dataframe::*;
use std::error::Error;

const INPUT_CSV_FILE: &str = "data/heart_statlog_cleveland_hungary_final.csv";

fn main() -> Result<(), Box<dyn Error>> {
    println!("starting!");

    let mut df = DataFrame::from_file(INPUT_CSV_FILE)?;
    df.log(5);
    df = df.expand_categorical("expanded", vec![1, 2, 5, 6, 8, 10])?;
    df.log(5);
    df = df.normalize("expanded_and_normalized", vec![0, 3, 4, 7, 9])?;
    df.log(5);
    df.write();

    println!("done!");

    Ok(())
}
