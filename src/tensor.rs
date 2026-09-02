#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    shape: (usize, usize),
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Self {
        assert!(data.len() == shape.0 * shape.1);
        Self { data, shape }
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        let i = shape.0 * shape.1;
        let data = vec![0.0; i];
        Self { data, shape }
    }

    pub fn ones(shape: (usize, usize)) -> Self {
        let i = shape.0 * shape.1;
        let data = vec![1.0; i];
        Self { data, shape }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(self.shape.0 > row && self.shape.1 > col);

        unsafe { *self.data.get_unchecked(row * self.shape.1 + col) }
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        assert!(self.shape.0 > row && self.shape.1 > col);
        self.data[row * self.shape.1 + col] = val;
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn add(&self, other: &Self) -> Self {
        assert!(self.shape == other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<f32>>();

        Self {
            data,
            shape: (self.shape.0, other.shape.1),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert!(self.shape == other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f32>>();

        Self {
            data,
            shape: (self.shape.0, other.shape.1),
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.1 == other.shape.0);

        // outout tensor row and col
        let or = self.shape.0;
        let oc = other.shape.1;

        let mut z = or * oc;
        let mut data = Vec::<f32>::with_capacity(z);

        let mut current_self_row = 0;
        let mut current_self_col = 0;
        let mut current_other_row = 0;
        let mut current_other_col = 0;

        loop {
            if z <= 0 {
                break;
            }

            let mut v = 0.0;

            for _ in 0..oc {
                v += self.data[current_self_row * self.shape.1 + current_self_col]
                    * other.data[current_other_row * other.shape.1 + current_other_col];

                current_self_col += 1;
                current_other_row += 1;
            }
            current_self_row += 1;
            current_other_col += 1;

            data.push(v);

            z -= 1;
        }

        Self {
            data,
            shape: (or, oc),
        }
    }
    pub fn scale(&self, scalar: f32) -> Self {
        let data = self.data.iter().map(|i| i * scalar).collect::<Vec<f32>>();

        Self {
            data,
            shape: self.shape,
        }
    }
    pub fn add_bias(&self, bias: &Self) -> Self {
        assert!(self.shape.1 == bias.shape.1);

        let mut i = 0;
        let mut j = 0;
        let mut data = Vec::with_capacity(self.shape.0 * self.shape.1);

        loop {
            if i >= self.data.len() {
                break;
            }
            let x = self.data[i] + bias.data[j];
            data.push(x);
            i += 1;
            j += 1;
            if j >= bias.shape.1 {
                j = 0;
            }
        }

        Self {
            data,
            shape: self.shape,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_zeros_and_ones() {
        let z = Tensor::zeros((2, 3));
        assert_eq!(z.shape(), (2, 3));
        assert_eq!(z.data.len(), 6);
        assert!(z.data.iter().all(|&x| x == 0.0));

        let o = Tensor::ones((3, 2));
        assert_eq!(o.shape(), (3, 2));
        assert_eq!(o.data.len(), 6);
        assert!(o.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_get_and_set_indexing() {
        let mut t = Tensor::zeros((3, 4));
        // Test index (0, 1) -> flat index 1
        t.set(0, 1, 42.0);
        assert_eq!(t.get(0, 1), 42.0);
        assert_eq!(t.data[1], 42.0);

        // Test index (2, 3) -> flat index 2 * 4 + 3 = 11
        t.set(2, 3, 99.0);
        assert_eq!(t.get(2, 3), 99.0);
        assert_eq!(t.data[11], 99.0);

        // Ensure length did not change
        assert_eq!(t.data.len(), 12);
    }

    #[test]
    fn test_elementwise_arithmetic() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], (2, 2));

        let sum = a.add(&b);
        assert_eq!(sum.data, vec![11.0, 22.0, 33.0, 44.0]);

        let diff = b.sub(&a);
        assert_eq!(diff.data, vec![9.0, 18.0, 27.0, 36.0]);

        let prod = a.mul(&b);
        assert_eq!(prod.data, vec![10.0, 40.0, 90.0, 160.0]);

        let scaled = a.scale(2.5);
        assert_eq!(scaled.data, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_add_bias_broadcasting() {
        // (3 rows, 2 cols)
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
        // (1 row, 2 cols)
        let b = Tensor::new(vec![10.0, 20.0], (1, 2));

        let y = x.add_bias(&b);
        assert_eq!(y.shape(), (3, 2));
        assert_eq!(y.data, vec![11.0, 22.0, 13.0, 24.0, 15.0, 26.0,]);
    }

    #[test]
    #[should_panic]
    fn test_shape_mismatch_add() {
        let a = Tensor::zeros((2, 3));
        let b = Tensor::zeros((3, 2));
        let _ = a.add(&b);
    }

    #[test]
    #[should_panic]
    fn test_bias_shape_mismatch() {
        let x = Tensor::zeros((3, 4));
        let b = Tensor::zeros((1, 3)); // should be (1, 4)
        let _ = x.add_bias(&b);
    }
}
