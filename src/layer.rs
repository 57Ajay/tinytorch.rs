use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
}

pub struct Linear {
    pub weights: Tensor,
    pub bias: Tensor,
    pub input: Option<Tensor>,
    pub d_weights: Option<Tensor>,
    pub d_bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weights = Tensor::xavier((in_features, out_features));
        let bias = Tensor::zeros((1, out_features));

        Self {
            weights,
            bias,
            input: None,
            d_weights: None,
            d_bias: None,
        }
    }
    pub fn from_weights(weights: Tensor, bias: Tensor) -> Self {
        assert!(bias.shape() == (1, weights.shape().1));
        Self {
            weights,
            bias,
            input: None,
            d_weights: None,
            d_bias: None,
        }
    }
}
impl Layer for Linear {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        assert!(input.shape().1 == self.weights.shape().0);
        self.input = Some(input.clone());
        input.matmul(&self.weights).add_bias(&self.bias)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        self.d_weights = Some(self.input.as_ref().unwrap().transpose().matmul(grad_output));
        self.d_bias = Some(grad_output.sum_axis0());

        grad_output.matmul(&self.weights.transpose())
    }
}

#[derive(Default)]
pub struct ReLU {
    pub input: Option<Tensor>,
}

#[derive(Default)]
pub struct Sigmoid {
    pub output: Option<Tensor>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { input: None }
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input = Some(input.clone());
        let data = input.data.iter().map(|x| x.max(0.0)).collect::<Vec<f32>>();
        Tensor::new(data, input.shape)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input = self.input.as_ref().unwrap();
        let data = input
            .data
            .iter()
            .zip(grad_output.data.iter())
            .map(|(&x, &grad)| if x > 0.0 { grad } else { 0.0 })
            .collect();

        Tensor::new(data, grad_output.shape)
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let data = input
            .data
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect::<Vec<f32>>();

        let output = Tensor::new(data, input.shape);
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let output = self.output.as_ref().unwrap();
        let data = output
            .data
            .iter()
            .zip(grad_output.data.iter())
            .map(|(&y, &grad)| grad * y * (1.0 - y))
            .collect();

        Tensor::new(data, grad_output.shape)
    }
}

#[cfg(test)]
mod test {
    use std::{assert_eq, vec};

    use super::*;

    #[test]
    fn test_linear_forward_shape() {
        let mut layer = Linear::new(3, 2);
        // Input: batch of 4 samples, each with 3 features -> shape (4, 3)
        let x = Tensor::ones((4, 3));
        let out = layer.forward(&x);

        // Output should be (4, 2)
        assert_eq!(out.shape(), (4, 2));
    }

    #[test]
    fn test_linear_forward_computation() {
        // W: (2, 3)
        // [ [1.0, 2.0, 3.0],
        //   [4.0, 5.0, 6.0] ]
        // b: (1, 3)
        // [ [0.5, -0.5, 1.0] ]
        let weights = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
        let bias = Tensor::new(vec![0.5, -0.5, 1.0], (1, 3));

        let mut layer = Linear::from_weights(weights, bias);

        // X: (1, 2) -> [ [2.0, 3.0] ]
        // X * W = [ 2*1 + 3*4, 2*2 + 3*5, 2*3 + 3*6 ] = [ 14.0, 19.0, 24.0 ]
        // + b   = [ 14.5, 18.5, 25.0 ]
        let x = Tensor::new(vec![2.0, 3.0], (1, 2));
        let out = layer.forward(&x);

        assert_eq!(out.shape(), (1, 3));
        assert_eq!(out.data, vec![14.5, 18.5, 25.0]);
    }

    #[test]
    #[should_panic]
    fn test_linear_input_dim_mismatch() {
        let mut layer = Linear::new(3, 2);
        // Passed 4 features when layer expects 3
        let x = Tensor::ones((2, 4));
        let _ = layer.forward(&x);
    }

    #[test]
    fn test_relu_forward() {
        let mut relu = ReLU::new();
        // Shape (2, 3) with negative, zero, and positive values
        let x = Tensor::new(vec![-3.0, -0.5, 0.0, 1.5, 2.0, -10.0], (2, 3));
        let out = relu.forward(&x);

        assert_eq!(out.shape(), (2, 3));
        assert_eq!(out.data, vec![0.0, 0.0, 0.0, 1.5, 2.0, 0.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let mut sigmoid = Sigmoid::new();
        let x = Tensor::new(vec![0.0, 2.0, -2.0], (1, 3));
        let out = sigmoid.forward(&x);

        assert_eq!(out.shape(), (1, 3));
        // sigma(0) = 0.5
        assert!((out.data[0] - 0.5).abs() < 1e-5);
        // sigma(2) ≈ 0.880797
        assert!((out.data[1] - 0.880797).abs() < 1e-5);
        // sigma(-2) ≈ 0.119203
        assert!((out.data[2] - 0.119203).abs() < 1e-5);
    }

    #[test]
    fn test_relu_backward() {
        let mut relu = ReLU::new();
        let x = Tensor::new(vec![-2.0, 0.0, 3.0, -0.5, 4.0, 0.0], (2, 3));
        let _y = relu.forward(&x);

        let grad_out = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
        let grad_in = relu.backward(&grad_out);

        assert_eq!(grad_in.shape(), (2, 3));
        // When x > 0: grad_in = grad_out; else 0
        assert_eq!(grad_in.data, vec![0.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn test_sigmoid_backward() {
        let mut sigmoid = Sigmoid::new();
        let x = Tensor::new(vec![0.0], (1, 1));
        let _ = sigmoid.forward(&x); // out is 0.5

        let grad_out = Tensor::new(vec![2.0], (1, 1));
        let grad_in = sigmoid.backward(&grad_out);

        // grad_in = grad_out * out * (1 - out) = 2.0 * 0.5 * 0.5 = 0.5
        assert_eq!(grad_in.shape(), (1, 1));
        assert!((grad_in.data[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_linear_backward() {
        // X: (2, 2)
        // [[1.0, 2.0],
        //  [3.0, 4.0]]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));

        // W: (2, 1)
        // [[0.5],
        //  [1.5]]
        let weights = Tensor::new(vec![0.5, 1.5], (2, 1));

        // b: (1, 1)
        // [[0.1]]
        let bias = Tensor::new(vec![0.1], (1, 1));

        let mut linear = Linear::from_weights(weights, bias);
        let _ = linear.forward(&x);

        // grad_out: (2, 1)
        // [[1.0],
        //  [2.0]]
        let grad_out = Tensor::new(vec![1.0, 2.0], (2, 1));
        let grad_in = linear.backward(&grad_out);

        // 1. grad_in = grad_out * W^T
        // grad_out: (2, 1), W^T: (1, 2) [[0.5, 1.5]]
        // [[1.0*0.5, 1.0*1.5],
        //  [2.0*0.5, 2.0*1.5]] = [[0.5, 1.5], [1.0, 3.0]]
        assert_eq!(grad_in.shape(), (2, 2));
        assert_eq!(grad_in.data, vec![0.5, 1.5, 1.0, 3.0]);

        // 2. d_weights = X^T * grad_out
        // X^T: (2, 2) [[1.0, 3.0], [2.0, 4.0]]
        // grad_out: (2, 1) [[1.0], [2.0]]
        // row 0: 1*1 + 3*2 = 7.0
        // row 1: 2*1 + 4*2 = 10.0
        let dw = linear.d_weights.as_ref().unwrap();
        assert_eq!(dw.shape(), (2, 1));
        assert_eq!(dw.data, vec![7.0, 10.0]);

        // 3. d_bias = sum_axis0(grad_out)
        // grad_out: [[1.0], [2.0]] -> sum is [[3.0]] (1, 1)
        let db = linear.d_bias.as_ref().unwrap();
        assert_eq!(db.shape(), (1, 1));
        assert_eq!(db.data, vec![3.0]);
    }
}
