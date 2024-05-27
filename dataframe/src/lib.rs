use half::f16;
use memmap::{Mmap, MmapOptions};
use std::error::Error;
use std::fs::{remove_file, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

// TODO: restructure header writes
// TODO: column names need to be comma separated

#[derive(Debug)]
pub struct DataFrame {
    pub file_name: String,
    mmap: Mmap,
    data_start: usize,
    pub rows: usize,
    pub cols: usize,
    pub col_sizes: Vec<usize>,
    pub col_starts: Vec<usize>,
    pub col_size: usize,
    pub col_names: Vec<String>,
    permanent: bool,
}

const FILE_TYPE: &str = ".df";

// Custom drop implementation to delete the file that backs the DF
impl Drop for DataFrame {
    fn drop(&mut self) {
        if !self.permanent && Path::new(&self.file_name).exists() {
            remove_file(&self.file_name).unwrap();
        }
    }
}

// A DataFrame is always backed by a file (on transform or from_file, we create a new file)
// A DataFrame is mmaped - if we iterate row by row, we'll never need to have the whole DF in
// memory
impl DataFrame {
    // TODO: load from .df file

    // make the file backing this DF permanent
    pub fn write(&mut self) {
        self.permanent = true;
    }

    pub fn log(&self, rows: usize) {
        for j in 0..self.cols {
            println!("{}: {}", self.col_names[j], self.col_sizes[j]);
        }
        for i in 0..rows {
            let row = self.get_row(i);
            for x in row.iter() {
                print!("{} ", x);
            }
            println!();
        }
    }

    pub fn normalize(&self, to_name: &str, cols: Vec<usize>) -> Result<DataFrame, Box<dyn Error>> {
        let mut ave = vec![f16::ZERO; self.col_size];
        let rows = f16::from_f64(self.rows as f64);
        for i in 0..self.rows {
            let vec = self.get_row(i);
            for j in 0..self.col_size {
                ave[j] += vec[j] / rows;
            }
        }
        let mut stddev = vec![0.0; self.col_size]; // need more precision
        let mut stddev_half = vec![f16::ZERO; self.col_size];
        for i in 0..self.rows {
            let vec = self.get_row(i);
            for j in 0..self.col_size {
                let x = vec[j] - ave[j];
                stddev[j] += ((x * x) / rows).to_f64();
            }
        }
        for j in 0..self.col_size {
            stddev[j] = f64::sqrt(stddev[j]);
            stddev_half[j] = f16::from_f64(stddev[j]);
        }
        let f = |vec: Vec<f16>| {
            let mut normalized = vec.clone();
            for j in cols.iter().copied() {
                let start = self.col_starts[j];
                normalized[start] = (normalized[start] - ave[start]) / stddev_half[start];
            }
            normalized
        };
        self.transform(to_name, f)
    }

    // TODO: assert that cols are size 1
    pub fn expand_categorical(
        &self,
        to_name: &str,
        cols: Vec<usize>,
    ) -> Result<DataFrame, Box<dyn Error>> {
        let mut to_col_sizes = self.col_sizes.clone();
        for i in 0..self.rows {
            for c in cols.iter().copied() {
                let val = self.get(i, c).to_f32() as usize;
                if val + 1 > to_col_sizes[c] {
                    // zero indexed
                    to_col_sizes[c] = val + 1;
                }
            }
        }
        let total_size: usize = to_col_sizes.iter().sum();
        let expand = |vec: Vec<f16>| {
            let mut expanded = vec![f16::ZERO; total_size];
            let mut f: usize = 0;
            let mut t: usize = 0;
            for j in 0..self.cols {
                let size = to_col_sizes[j];
                if self.col_sizes[j] == size {
                    for _ in 0..size {
                        expanded[t] = vec[f];
                        t += 1;
                        f += 1;
                    }
                } else {
                    // assuming previous size was 1
                    let val = vec[f].to_f32() as usize;
                    for k in 0..size {
                        if k == val {
                            expanded[t] = f16::ONE;
                        }
                        t += 1;
                    }
                    f += 1;
                }
            }
            expanded
        };
        self.transform_with_sizes(to_name, to_col_sizes.clone(), expand)
    }

    // KNOWN FAULT: f could expand/shrink Vec size without updating `col_sizes`
    pub fn transform<F: FnMut(Vec<f16>) -> Vec<f16>>(
        &self,
        to_name: &str,
        f: F,
    ) -> Result<DataFrame, Box<dyn Error>> {
        self.transform_with_sizes(to_name, self.col_sizes.clone(), f)
    }

    // make sure to `drop` the previous df
    fn transform_with_sizes<F: FnMut(Vec<f16>) -> Vec<f16>>(
        &self,
        to_name: &str,
        to_col_sizes: Vec<usize>,
        mut f: F,
    ) -> Result<DataFrame, Box<dyn Error>> {
        let df_file_name = format!("data/{}{}", to_name, FILE_TYPE);
        if Path::new(&df_file_name).exists() {
            remove_file(&df_file_name)?;
        }
        let write_fd = OpenOptions::new()
            .write(true)
            .create_new(true)
            .append(true)
            .open(df_file_name.clone())?;
        let mut writer = BufWriter::new(write_fd);

        // write header: num rows, cols, col sizes, col names
        writer.write(&self.rows.to_ne_bytes())?;
        writer.write(&self.cols.to_ne_bytes())?;
        let mut to_col_size: usize = 0;
        for i in 0..self.cols {
            writer.write(&to_col_sizes[i].to_ne_bytes())?;
            to_col_size += to_col_sizes[i];
        }
        let newline = "\n".as_bytes();
        for i in 0..self.cols {
            writer.write(self.col_names[i].as_bytes())?;
        }
        writer.write(newline)?;

        // compute the number of bytes written so far
        let usize_bytes: usize = (usize::BITS / 8) as usize;
        let bytes_written = usize_bytes
            + usize_bytes
            + (usize_bytes * self.cols)
            + self.col_names.iter().map(|s| s.len()).sum::<usize>()
            + 1;

        // write mapped data
        for i in 0..self.rows {
            for val in f(self.get_row(i)) {
                writer.write(&val.to_ne_bytes())?;
            }
        }

        // Finished writing, create a readonly mmap
        drop(writer);
        let read_fd = OpenOptions::new()
            .read(true)
            .write(false)
            .open(df_file_name.clone())?;
        let mmap = unsafe { MmapOptions::new().map(&read_fd)? };

        let mut col_starts = vec![0; self.cols];
        let mut prefix_sum = 0;
        for j in 0..self.cols {
            col_starts[j] = prefix_sum;
            prefix_sum += to_col_sizes[j];
        }

        Ok(DataFrame {
            file_name: df_file_name,
            mmap,
            rows: self.rows,
            cols: self.cols,
            col_sizes: to_col_sizes,
            col_starts,
            col_names: self.col_names.clone(),
            data_start: bytes_written,
            col_size: to_col_size,
            permanent: false,
        })
    }

    pub fn get(&self, i: usize, j: usize) -> f16 {
        let loc = self.data_start + ((i * self.col_size + j) * 2);
        let bytes = unsafe { *(self.mmap[loc..=loc + 1].as_ptr() as *const [u8; 2]) };
        f16::from_ne_bytes(bytes)
    }

    // TODO: shouldn't need to iter if possible
    pub fn get_row(&self, i: usize) -> Vec<f16> {
        let mut res = vec![f16::ZERO; self.col_size];
        let mut loc = self.data_start + ((i * self.col_size) * 2);
        for j in 0..self.col_size {
            let bytes = unsafe { *(self.mmap[loc..=loc + 1].as_ptr() as *const [u8; 2]) };
            res[j] = f16::from_ne_bytes(bytes);
            loc += 2;
        }
        res
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
        let df_file_name = file.replace(".csv", FILE_TYPE);
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

        let mut col_starts = vec![0; num_cols];
        for j in 0..num_cols {
            col_starts[j] = j;
        }

        Ok(DataFrame {
            file_name: df_file_name,
            mmap,
            rows: num_rows,
            cols: num_cols,
            col_sizes: col_lens,
            col_starts,
            col_names: column_names,
            col_size: num_cols,
            data_start: bytes_written,
            permanent: false,
        })
    }
}
