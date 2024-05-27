use core::fmt;
use std::fmt::Display;

// TODO: split up with `mod`

pub struct ModelSchema {
    pub num_layers: usize,
    pub input_size: usize,
    pub masks: Vec<Mask>,
}

pub struct Model {
    pub layers: Vec<Layer>,
    pub schema: ModelSchema,
}

// grid[i][j] true if output i inputs j
// rows correspond to outputs
pub struct Mask {
    pub grid: Vec<Vec<bool>>,
}

#[derive(Clone)]
pub struct Layer {}

impl Mask {
    pub fn input_size(&self) -> usize {
        match self.grid.first() {
            Some(x) => x.len(),
            None => 0,
        }
    }

    pub fn output_size(&self) -> usize {
        self.grid.len()
    }

    pub fn fully_connected(input_size: usize, output_size: usize) -> Mask {
        Mask {
            grid: vec![vec![true; input_size]; output_size],
        }
    }
}

impl ModelSchema {
    pub fn new(input_size: usize) -> ModelSchema {
        ModelSchema {
            num_layers: 0,
            input_size,
            masks: vec![],
        }
    }

    pub fn current_output_size(&self) -> usize {
        match self.masks.last() {
            Some(last) => last.output_size(),
            None => self.input_size,
        }
    }

    pub fn add_layer(&mut self, mask: Mask) -> Result<(), String> {
        if mask.input_size() != self.current_output_size() {
            return Err(String::from(format!(
                "mask is incorrectly sized: previous output size was {}, but mask expects {}",
                self.current_output_size(),
                mask.input_size()
            )));
        }

        self.num_layers += 1;
        self.masks.push(mask);
        Ok(())
    }
}

impl Display for ModelSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            f,
            "model schema with {} layers and {} output size",
            self.num_layers,
            self.current_output_size()
        )
    }
}

impl Model {
    pub fn new(schema: ModelSchema) -> Model {
        Model {
            layers: vec![Layer {}; schema.num_layers],
            schema,
        }
    }
}
