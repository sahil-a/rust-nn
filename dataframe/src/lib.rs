use half::f16;
use memmap::{Mmap, MmapOptions};
use rand::Rng;
use std::error::Error;
use std::fs::{remove_file, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::str::FromStr;

fn write_header(
    writer: &mut BufWriter<std::fs::File>,
    rows: usize,
    cols: usize,
    col_sizes: &[usize],
    col_names: &[String],
) -> Result<usize, std::io::Error> {
    // header format: num rows, cols, col sizes, col names
    writer.write(&rows.to_ne_bytes())?;
    writer.write(&cols.to_ne_bytes())?;
    for size in col_sizes {
        writer.write(&size.to_ne_bytes())?;
    }
    let newline = "\n".as_bytes();
    for (i, name) in col_names.iter().enumerate() {
        writer.write(name.as_bytes())?;
        if i < col_names.len() - 1 {
            writer.write(",".as_bytes())?;
        }
    }
    writer.write(newline)?;

    // compute and return the number of bytes written
    let usize_bytes: usize = (usize::BITS / 8) as usize;
    Ok(usize_bytes
        + usize_bytes
        + (usize_bytes * cols)
        + col_names.iter().map(|s| s.len()).sum::<usize>()
        + if cols > 0 { cols - 1 } else { 0 }  // Account for commas
        + 1) // Newline
}

#[derive(Debug)]
pub struct DataFrame {
    pub file_name: String,
    mmap: Mmap,
    data_start: usize,
    pub rows: usize,
    pub cols: usize,
    // default train/val/test split is even
    val_start: usize,
    test_start: usize,
    pub col_sizes: Vec<usize>,
    pub col_starts: Vec<usize>, // prefix sum of `col_sizes`
    pub col_size: usize,        // sum of `col_sizes`
    pub col_names: Vec<String>,
    permanent: bool,
}

const FILE_TYPE: &str = ".df";

pub enum DataSegment {
    Train,
    Val,
    Test,
}

// Custom drop implementation to delete the file that backs the DF (unless permanent)
impl Drop for DataFrame {
    fn drop(&mut self) {
        if !self.permanent && Path::new(&self.file_name).exists() {
            remove_file(&self.file_name).unwrap();
        }
    }
}

fn round(value: f16, places: i32) -> f16 {
    let scale = (10.0 as f32).powi(places);
    f16::from_f32((value.to_f32() * scale).round() / scale)
}

// A DataFrame is always backed by a file (on transform or from_file, we create a new file)
// A DataFrame is mmaped - if we iterate row by row, we'll never need to have the whole DF in
// memory
impl DataFrame {
    // make the file backing this DF permanent
    pub fn write(&mut self) {
        self.permanent = true;
    }

    // Restructures the train/val/test split
    pub fn train_val_test_split(
        &mut self,
        train_weight: usize,
        val_weight: usize,
        test_weight: usize,
    ) {
        let sum = (train_weight + val_weight + test_weight) as f32;
        let train_percentage = (train_weight as f32) / (sum);
        let val_percentage = (val_weight as f32) / (sum);
        let train_rows = ((self.rows as f32) * train_percentage) as usize;
        let val_rows = ((self.rows as f32) * val_percentage) as usize;

        self.val_start = train_rows;
        self.test_start = train_rows + val_rows;
    }

    pub fn shuffle(&mut self, segment: &DataSegment) -> Result<(), std::io::Error> {
        let mut fd1 = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.file_name)?;
        let mut fd2 = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.file_name)?;

        // for each row in the segment, pick a random remaining row to swap
        let start_row = match segment {
            DataSegment::Train => 0,
            DataSegment::Val => self.val_start,
            DataSegment::Test => self.test_start,
        };

        let num_rows = match segment {
            DataSegment::Train => self.val_start,
            DataSegment::Val => self.test_start - self.val_start,
            DataSegment::Test => self.rows - self.test_start,
        };

        let mut buffer1 = vec![0u8; 2 * self.col_size];
        let mut buffer2 = vec![0u8; 2 * self.col_size];

        for i in start_row..(start_row + num_rows) {
            fd1.seek(SeekFrom::Start(
                (self.data_start + (i * self.col_size * 2)) as u64,
            ))?;
            fd1.read_exact(&mut buffer1)?;
            let j = rand::thread_rng().gen_range(i..start_row + num_rows);
            fd2.seek(SeekFrom::Start(
                (self.data_start + (j * self.col_size * 2)) as u64,
            ))?;
            fd2.read_exact(&mut buffer2)?;

            // Write row j to position i
            fd1.seek(SeekFrom::Start(
                (self.data_start + (i * self.col_size * 2)) as u64,
            ))?;
            fd1.write_all(&buffer2)?;

            // Write row i to position j
            fd2.seek(SeekFrom::Start(
                (self.data_start + (j * self.col_size * 2)) as u64,
            ))?;
            fd2.write_all(&buffer1)?;
        }

        // Drop existing file descriptors
        drop(fd1);
        drop(fd2);

        // Recreate the mmap
        let read_fd = OpenOptions::new()
            .read(true)
            .write(false)
            .open(&self.file_name)?;
        self.mmap = unsafe { MmapOptions::new().map(&read_fd)? };

        Ok(())
    }

    pub fn log(&self, rows: usize) {
        println!();
        for j in 0..self.cols {
            let len = self.col_names[j].len();
            print!("{:<width$}", self.col_names[j], width = len + 4);
        }
        println!();
        for i in 0..rows {
            let row = self.get_row(i);
            let mut curr: usize = 0;
            for j in 0..self.cols {
                let mut start = self.col_size;
                if j != self.cols - 1 {
                    start = self.col_starts[j + 1];
                }
                let mut s: String = String::from("");
                while curr != start {
                    let x = row[curr];
                    if x % f16::ONE == f16::ZERO {
                        s.push_str(&format!("{} ", x));
                    } else {
                        s.push_str(&format!("{:.2} ", x));
                    }
                    curr += 1;
                }
                let len = self.col_names[j].len();
                print!("{:<width$}", s, width = len + 4);
            }
            println!();
        }
        println!();
    }

    pub fn normalize(&self, to_name: &str, cols: Vec<usize>) -> Result<DataFrame, Box<dyn Error>> {
        let mut ave = vec![f16::ZERO; self.col_size];
        let rows = f16::from_f64(self.rows as f64);
        for i in 0..self.rows {
            let vec = self.get_row(i);
            for j in 0..self.col_size {
                ave[j] += round(vec[j] / rows, 2);
            }
        }
        let mut stddev = vec![0.0; self.col_size]; // need more precision
        let mut stddev_half = vec![f16::ZERO; self.col_size];
        for i in 0..self.rows {
            let vec = self.get_row(i);
            for j in 0..self.col_size {
                let x = (vec[j] - ave[j]).to_f64();
                stddev[j] += (x * x) / rows.to_f64();
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

    pub fn expand_categorical(
        &self,
        to_name: &str,
        cols: Vec<usize>,
    ) -> Result<DataFrame, Box<dyn Error>> {
        // Verify all specified columns have size 1
        for &col in cols.iter() {
            if self.col_sizes[col] != 1 {
                return Err(format!(
                    "Column {} must have size 1 for categorical expansion, but has size {}",
                    self.col_names[col], self.col_sizes[col]
                )
                .into());
            }
        }
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

        let to_col_size: usize = to_col_sizes.iter().sum();
        let bytes_written = write_header(
            &mut writer,
            self.rows,
            self.cols,
            &to_col_sizes,
            &self.col_names,
        )?;

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
            val_start: self.val_start,
            test_start: self.test_start,
        })
    }

    pub fn get(&self, i: usize, j: usize) -> f16 {
        let loc = self.data_start + ((i * self.col_size + j) * 2);
        let bytes = unsafe { *(self.mmap[loc..=loc + 1].as_ptr() as *const [u8; 2]) };
        f16::from_ne_bytes(bytes)
    }

    pub fn get_row(&self, i: usize) -> Vec<f16> {
        let start = self.data_start + (i * self.col_size * 2);
        let end = start + (self.col_size * 2);

        // Safety: We know the mmap contains valid f16 data in native endian format
        // and we're staying within bounds of the allocation
        unsafe {
            let ptr = self.mmap[start..end].as_ptr() as *const f16;
            // Create a slice from the raw pointer without taking ownership
            std::slice::from_raw_parts(ptr, self.col_size).to_vec()
        }
    }

    pub fn get_batch(
        &self,
        batch_num: usize,
        batch_size: usize,
        segment: &DataSegment,
    ) -> Vec<Vec<f16>> {
        let (start_row, end_row) = match segment {
            DataSegment::Train => (0, self.val_start),
            DataSegment::Val => (self.val_start, self.test_start),
            DataSegment::Test => (self.test_start, self.rows),
        };

        let batch_start = start_row + (batch_num * batch_size);

        if batch_start >= end_row {
            return Vec::new();
        }

        let available_rows = end_row - batch_start;
        let actual_batch_size = batch_size.min(available_rows);

        let mut batch = Vec::with_capacity(actual_batch_size);

        for i in 0..actual_batch_size {
            let row_idx = batch_start + i;
            batch.push(self.get_row(row_idx));
        }

        batch
    }

    pub fn get_segment(&self, segment: &DataSegment) -> Vec<Vec<f16>> {
        let (start_row, end_row) = match segment {
            DataSegment::Train => (0, self.val_start),
            DataSegment::Val => (self.val_start, self.test_start),
            DataSegment::Test => (self.test_start, self.rows),
        };

        let mut segment_data = Vec::with_capacity(end_row - start_row);

        for i in start_row..end_row {
            segment_data.push(self.get_row(i));
        }

        segment_data
    }

    pub fn get_data_segment_size(&self, segment: &DataSegment) -> usize {
        match segment {
            DataSegment::Train => self.val_start,
            DataSegment::Val => self.test_start - self.val_start,
            DataSegment::Test => self.rows - self.test_start,
        }
    }

    pub fn load_df(file: &str) -> Result<DataFrame, Box<dyn Error>> {
        let read_fd = OpenOptions::new().read(true).write(false).open(file)?;
        let mmap = unsafe { MmapOptions::new().map(&read_fd)? };

        // Read header: num rows, cols, col sizes
        let usize_bytes = (usize::BITS / 8) as usize;
        let mut pos = 0;

        // Read rows and cols
        let rows = usize::from_ne_bytes(mmap[pos..pos + usize_bytes].try_into()?);
        pos += usize_bytes;
        let cols = usize::from_ne_bytes(mmap[pos..pos + usize_bytes].try_into()?);
        pos += usize_bytes;

        // Read column sizes
        let mut col_sizes = Vec::with_capacity(cols);
        for _ in 0..cols {
            let size = usize::from_ne_bytes(mmap[pos..pos + usize_bytes].try_into()?);
            col_sizes.push(size);
            pos += usize_bytes;
        }

        // Read column names until newline
        let mut col_names = Vec::with_capacity(cols);
        let mut name_buf = String::new();
        while pos < mmap.len() && mmap[pos] as char != '\n' {
            if mmap[pos] as char == ',' {
                col_names.push(name_buf);
                name_buf = String::new();
            } else {
                name_buf.push(mmap[pos] as char);
            }
            pos += 1;
        }
        col_names.push(name_buf); // Push final name
        pos += 1; // Skip newline

        // Calculate column starts
        let mut col_starts = vec![0; cols];
        let mut prefix_sum = 0;
        for j in 0..cols {
            col_starts[j] = prefix_sum;
            prefix_sum += col_sizes[j];
        }

        let col_size: usize = col_sizes.iter().sum();
        let mut df = DataFrame {
            file_name: file.to_string(),
            mmap,
            rows,
            cols,
            col_sizes,
            col_starts,
            col_names,
            col_size,
            data_start: pos,
            permanent: true, // Since we're loading an existing file
            val_start: 0,
            test_start: 0,
        };

        df.train_val_test_split(1, 1, 1);

        Ok(df)
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

        let one: usize = 1;
        let col_lens = vec![one; num_cols];
        let bytes_written =
            write_header(&mut writer, num_rows, num_cols, &col_lens, &column_names)?;

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

        let mut df = DataFrame {
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
            val_start: 0,
            test_start: 0,
        };

        df.train_val_test_split(1, 1, 1);

        Ok(df)
    }
}
