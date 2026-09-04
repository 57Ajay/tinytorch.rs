use std::vec;

use rand::RngExt;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: (usize, usize),
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
        assert!(self.shape == other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<f32>>();

        Self {
            data,
            shape: (self.shape.0, other.shape.1),
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

    pub fn transpose(&self) -> Self {
        let shape = (self.shape.1, self.shape.0);
        let l = self.data.len();
        let mut data = Vec::<f32>::with_capacity(l);

        for c in 0..self.shape.1 {
            for r in 0..self.shape.0 {
                data.push(self.get(r, c));
            }
        }
        Self { data, shape }
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert!(self.shape.1 == other.shape.0);

        let r = self.shape.0;
        let c = other.shape.1;

        let shape = (r, c);
        let mut data = Vec::with_capacity(r * c);

        for i in 0..r {
            for j in 0..c {
                let mut sum = 0.0;
                for k in 0..self.shape.1 {
                    sum += self.get(i, k) * other.get(k, j);
                }
                data.push(sum);
            }
        }

        Self { data, shape }
    }

    pub fn rand_uniform(shape: (usize, usize), low: f32, high: f32) -> Self {
        let mut data = Vec::with_capacity(shape.0 * shape.1);
        let mut i = 0;
        loop {
            if i == shape.0 * shape.1 {
                break;
            }
            let v = rand::rng().random_range(low..high);
            data.push(v);
            i += 1;
        }

        Self { data, shape }
    }

    pub fn xavier(shape: (usize, usize)) -> Self {
        let limit = (6.0 / (shape.0 + shape.1) as f32).sqrt();
        Self::rand_uniform(shape, -limit, limit)
    }

    pub fn he(shape: (usize, usize)) -> Self {
        let limit = (6.0 / shape.0 as f32).sqrt();
        Self::rand_uniform(shape, -limit, limit)
    }

    pub fn sum_axis0(&self) -> Self {
        let mut data = Vec::<f32>::with_capacity(self.shape.1);

        for i in 0..self.shape.1 {
            let mut sum = 0.0;
            for j in 0..self.shape.0 {
                sum += self.get(j, i);
            }
            data.push(sum);
        }

        Self {
            data,
            shape: (1, self.shape.1),
        }
    }
}

#[cfg(test)]
mod test {
    use std::{assert_eq, vec};

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

    #[test]
    fn test_transpose() {
        // [[1, 2, 3],
        //  [4, 5, 6]] (2x3)
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
        let at = a.transpose();

        assert_eq!(at.shape(), (3, 2));
        // Expected transpose:
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        assert_eq!(at.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul() {
        // A: (2x3)
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));

        // B: (3x2)
        // [[7,  8],
        //  [9,  1],
        //  [2,  3]]
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0], (3, 2));

        // C = A x B: (2x2)
        // Row 0 x Col 0 = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        // Row 0 x Col 1 = 1*8 + 2*1 + 3*3 = 8 + 2 + 9  = 19
        // Row 1 x Col 0 = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
        // Row 1 x Col 1 = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
        let c = a.matmul(&b);
        assert_eq!(c.shape(), (2, 2));
        assert_eq!(c.data, vec![31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    #[should_panic]
    fn test_matmul_shape_mismatch() {
        let a = Tensor::zeros((2, 3));
        let b = Tensor::zeros((4, 2)); // 3 != 4, must panic!
        let _ = a.matmul(&b);
    }

    #[test]
    fn test_rand_uniform() {
        let t = Tensor::rand_uniform((10, 20), -2.0, 3.0);
        assert_eq!(t.shape(), (10, 20));
        assert_eq!(t.data.len(), 200);

        // Check bounds
        for &val in &t.data {
            assert!(val >= -2.0 && val <= 3.0, "Value {} out of bounds", val);
        }

        // Check that not all elements are identical (true randomness)
        let first = t.data[0];
        assert!(t.data.iter().any(|&x| (x - first).abs() > 1e-5));
    }

    #[test]
    fn test_xavier_init() {
        // fan_in = 100, fan_out = 200
        // limit = sqrt(6 / 300) = sqrt(0.02) ≈ 0.14142
        let t = Tensor::xavier((100, 200));
        assert_eq!(t.shape(), (100, 200));
        let expected_limit = (6.0 / 300.0_f32).sqrt();

        for &val in &t.data {
            assert!(
                val >= -expected_limit && val <= expected_limit,
                "Value {} exceeds Xavier limit {}",
                val,
                expected_limit
            );
        }
    }

    #[test]
    fn test_he_init() {
        // fan_in = 100, fan_out = 200
        // limit = sqrt(6 / 100) = sqrt(0.06) ≈ 0.24495
        let t = Tensor::he((100, 200));
        assert_eq!(t.shape(), (100, 200));
        let expected_limit = (6.0 / 100.0_f32).sqrt();

        for &val in &t.data {
            assert!(
                val >= -expected_limit && val <= expected_limit,
                "Value {} exceeds He limit {}",
                val,
                expected_limit
            );
        }
    }

    #[test]
    fn test_sum_axis0() {
        // (3, 2)
        // [[1.0, 2.0],
        //  [3.0, 4.0],
        //  [5.0, 6.0]]
        // sum_axis0 -> [[9.0, 12.0]] shape (1, 2)
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));
        let s = t.sum_axis0();
        assert_eq!(s.shape(), (1, 2));
        assert_eq!(s.data, vec![9.0, 12.0]);
    }
}
