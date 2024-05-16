use std::io::{self, Error};
const INPUT_CSV_FILE: &str = "data/heart_statlog_cleveland_hungary_final.csv";

fn main() -> std::io::Result<()> {
    println!("starting!");

    let mut csv_reader = csv::Reader::from_path(INPUT_CSV_FILE)?;
    let column_names = csv_reader.headers().unwrap().clone();

    // iterate over the records and print them
    for result in csv_reader.records() {
        match result {
            Ok(record) => {
                for (i, field) in record.iter().enumerate() {
                    println!("{}: {}", column_names.get(i).unwrap(), field);
                }
                println!("")
            }
            Err(err) => {
                return Err(Error::new(io::ErrorKind::Other, err.to_string()));
            }
        }
    }

    Ok(())
}
