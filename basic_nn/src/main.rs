use dataframe::*;
use std::error::Error;

const INPUT_CSV_FILE: &str = "data/heart_statlog_cleveland_hungary_final.csv";

fn main() -> Result<(), Box<dyn Error>> {
    println!("starting!");

    let mut df = DataFrame::from_file(INPUT_CSV_FILE)?;
    df.write();
    println!("{}", df.get(6, 3));
    // TODO: figure out how to manipulate f16 & if it is worth it
    df = df.transform("transformed", |prev| {
        /*
        // multiply each value by 2
        let mut new = Vec::new();
        for val in prev {
            new.push(val * 2.0);
        }
        new
        */
        prev
    })?;
    df.write();
    df = df.expand_categorical("expanded_chest_pain_type", vec![2])?;
    df.write();

    // TODO: deal with categorical vs. numerical fields separately

    println!("done!");

    Ok(())
}
