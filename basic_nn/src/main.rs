use csv::StringRecord;
use std::error::Error;

const INPUT_CSV_FILE: &str = "data/heart_statlog_cleveland_hungary_final.csv";

fn main() -> Result<(), Box<dyn Error>> {
    println!("starting!");

    let mut csv_reader = csv::Reader::from_path(INPUT_CSV_FILE)?;

    let column_names = csv_reader.headers()?.clone();
    let num_columns = column_names.len();
    let num_rows = csv_reader.records().count();

    // print the csv
    iter_csv(|record| {
        for (i, field) in record.iter().enumerate() {
            println!("{}: {}", column_names.get(i).unwrap_or(""), field);
        }
        println!();
    })?;

    // TOOD: deal with categorical vs. numerical fields separately

    // normalize each field
    let mut totals: Vec<f64> = vec![0.0; num_columns];
    let mut means: Vec<f64> = vec![0.0; num_columns];
    let mut variances: Vec<f64> = vec![0.0; num_columns];
    let mut standard_deviations: Vec<f64> = vec![0.0; num_columns];

    iter_csv(|record| {
        for (i, field) in record.iter().enumerate() {
            totals[i] += field.parse::<f64>().unwrap_or(0.0);
        }
    })?;

    for i in 0..num_columns {
        means[i] = totals[i] / (num_rows as f64);
    }

    iter_csv(|record| {
        for (i, field) in record.iter().enumerate() {
            let value = field.parse::<f64>().unwrap_or(0.0);
            variances[i] += (value - means[i]).powi(2);
        }
    })?;

    // TODO: operations like this should be SIMD'd
    for i in 0..num_columns {
        variances[i] /= num_rows as f64;
    }
    for i in 0..num_columns {
        standard_deviations[i] = variances[i].sqrt();
    }

    println!("Normalized Data:");
    iter_norm_csv(means.clone(), standard_deviations.clone(), |record| {
        for (i, field) in record.iter().enumerate() {
            println!("{}: {}", column_names.get(i).unwrap_or(""), field);
        }
        println!();
    })?;

    println!("Means:");
    for (i, mean) in means.iter().enumerate() {
        println!("{}: {}", column_names.get(i).unwrap_or(""), mean);
    }
    println!();

    println!("Variances:");
    for (i, variance) in variances.iter().enumerate() {
        println!("{}: {}", column_names.get(i).unwrap_or(""), variance);
    }
    println!();

    println!("Standard Deviations:");
    for (i, standard_deviation) in standard_deviations.iter().enumerate() {
        println!(
            "{}: {}",
            column_names.get(i).unwrap_or(""),
            standard_deviation
        );
    }
    println!();

    println!("done!");

    Ok(())
}

// TODO: iter csv should also have column iteration
fn iter_csv<F>(mut f: F) -> Result<(), Box<dyn Error>>
where
    F: FnMut(StringRecord),
{
    let mut csv_reader = csv::Reader::from_path(INPUT_CSV_FILE)?;
    for result in csv_reader.records() {
        let record: StringRecord = result?;
        f(record)
    }
    Ok(())
}

fn iter_norm_csv<F>(means: Vec<f64>, std_devs: Vec<f64>, mut f: F) -> Result<(), Box<dyn Error>>
where
    F: FnMut(Vec<f64>),
{
    let mut csv_reader = csv::Reader::from_path(INPUT_CSV_FILE)?;
    for result in csv_reader.records() {
        let record: StringRecord = result?;
        let mut record_f64: Vec<f64> = vec![0.0; record.len()];
        for (i, field) in record.iter().enumerate() {
            record_f64[i] = field.parse::<f64>().unwrap_or(0.0);
        }
        for i in 0..record_f64.len() {
            record_f64[i] = (record_f64[i] - means[i]) / std_devs[i];
        }
        f(record_f64)
    }
    Ok(())
}
