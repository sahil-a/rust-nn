use csv::StringRecord;
use std::error::Error;

const INPUT_CSV_FILE: &str = "data/heart_statlog_cleveland_hungary_final.csv";

fn main() -> Result<(), Box<dyn Error>> {
    println!("starting!");

    let mut csv_reader = csv::Reader::from_path(INPUT_CSV_FILE)?;
    let column_names = csv_reader.headers()?.clone();

    // iterate over the records and print them
    for result in csv_reader.records() {
        let record: StringRecord = result?;
        for (i, field) in record.iter().enumerate() {
            println!("{}: {}", column_names.get(i).unwrap_or(""), field);
        }
        println!("")
    }

    Ok(())
}
