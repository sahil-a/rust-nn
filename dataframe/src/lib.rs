use half::f16;
use memmap::{Mmap, MmapOptions};
use std::error::Error;
use std::fs::{remove_file, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

#[derive(Debug)]
pub struct DataFrame {
    pub file_name: String,
    mmap: Mmap,
    data_start: usize,
    pub rows: usize,
    pub cols: usize,
    pub col_sizes: Vec<usize>,
    pub col_names: Vec<String>,
}

// A DataFrame is always backed by a file (on transform or from_file, we create a new file)
impl DataFrame {
    pub fn get(self, i: usize, j: usize) -> f16 {
        let loc = self.data_start + ((i * self.cols + j) * 2);
        let bytes = unsafe { *(self.mmap[loc..=loc + 1].as_ptr() as *const [u8; 2]) };
        f16::from_ne_bytes(bytes)
    }

    pub fn from_file(file: &str) -> Result<DataFrame, Box<dyn Error>> {
        let mut csv_reader = csv::Reader::from_path(file)?;
        let column_names_rec = csv_reader.headers()?.clone();
        let num_cols = column_names_rec.len();
        let num_rows = csv_reader.records().count();
        let mut column_names = vec!["".to_string(); num_cols];
        csv_reader = csv::Reader::from_path(file)?; // csv_reader should buffer for us;
                                                    // recreate because we alrd counted

        for (i, name) in column_names_rec.iter().enumerate() {
            column_names[i] = name.to_string();
        }

        // open a file to write to with the same name as "file" but with type .df instead of type
        // .csv
        let df_file_name = file.replace(".csv", ".df");
        if Path::new(&df_file_name).exists() {
            remove_file(&df_file_name)?;
        }
        let write_fd = OpenOptions::new()
            .write(true)
            .create_new(true)
            .append(true)
            .open(df_file_name.clone())?;
        let mut writer = BufWriter::new(write_fd);

        // write num rows, cols
        writer.write(&num_rows.to_ne_bytes())?;
        writer.write(&num_cols.to_ne_bytes())?;

        // write column vector vector lengths
        let one: usize = 1;
        let col_lens = vec![one; num_cols];
        for _ in 0..num_cols {
            writer.write(&one.to_ne_bytes())?;
        }

        // write column names
        let newline = "\n".as_bytes();
        for i in 0..num_cols {
            writer.write(column_names[i].as_bytes())?;
        }
        writer.write(newline)?;

        // compute the number of bytes written so far
        let usize_bytes: usize = (usize::BITS / 8) as usize;
        let bytes_written = usize_bytes
            + usize_bytes
            + (usize_bytes * num_cols)
            + column_names.iter().map(|s| s.len()).sum::<usize>()
            + 1;

        // write data
        for record in csv_reader.records() {
            for (_, val) in record?.iter().enumerate() {
                let parsed = half::f16::from_str(val)?;
                writer.write(&parsed.to_ne_bytes())?;
            }
        }

        // Finished writing, create a readonly mmap
        drop(writer);
        let read_fd = OpenOptions::new()
            .read(true)
            .write(false)
            .open(df_file_name.clone())?;
        let mmap = unsafe { MmapOptions::new().map(&read_fd)? };

        Ok(DataFrame {
            file_name: df_file_name,
            mmap,
            rows: num_rows,
            cols: num_cols,
            col_sizes: col_lens,
            col_names: column_names,
            data_start: bytes_written,
        })
    }

    // TODO: transform should be used to make an expand_categorical and normalize_scalar func
    pub fn transform<F: FnMut(Vec<f16>) -> Vec<f16>, R>(self, f: F)
    //-> DataFrame<R> (TODO)
    {
    }
}
