use half::{f16, vec};
use memmap::{Mmap, MmapOptions};
use std::error::Error;
use std::fs::{remove_file, OpenOptions};
use std::io::{BufWriter, Write};
use std::ops::Index;
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
    pub col_size: usize,
    pub col_names: Vec<String>,
    permanent: bool,
}

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

    // TODO: assert that cols are size 1
    // TODO: impl
    pub fn expand_categorical(self, cols: Vec<usize>) -> Result<DataFrame, Box<dyn Error>> {
        Ok(self)
    }

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
        let df_file_name = to_name.to_string() + ".df";
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

        Ok(DataFrame {
            file_name: df_file_name,
            mmap,
            rows: self.rows,
            cols: self.cols,
            col_sizes: to_col_sizes,
            col_names: self.col_names.clone(),
            data_start: bytes_written,
            col_size: to_col_size,
            permanent: false,
        })
    }

    pub fn get(&self, i: usize, j: usize) -> f16 {
        let loc = self.data_start + ((i * self.cols + j) * 2);
        let bytes = unsafe { *(self.mmap[loc..=loc + 1].as_ptr() as *const [u8; 2]) };
        f16::from_ne_bytes(bytes)
    }

    // TODO: shouldn't need to iter if possible
    pub fn get_row(&self, i: usize) -> Vec<f16> {
        let mut res = vec![f16::ZERO; self.col_size];
        let mut loc = self.data_start + ((i * self.cols) * 2);
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
            col_size: num_cols,
            data_start: bytes_written,
            permanent: false,
        })
    }
}
