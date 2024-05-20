use dataframe::*;
use std::error::Error;

const INPUT_CSV_FILE: &str = "data/heart_statlog_cleveland_hungary_final.csv";

fn main() -> Result<(), Box<dyn Error>> {
    println!("starting!");

    let df = DataFrame::from_file(INPUT_CSV_FILE)?;
    println!("{}", df.get(6, 3));

    // TODO: deal with categorical vs. numerical fields separately

    println!("done!");

    Ok(())
}
