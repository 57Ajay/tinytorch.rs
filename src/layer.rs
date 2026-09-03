use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
}

pub struct Linear {
    pub weights: Tensor,
    pub bias: Tensor,
    pub input: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weights = Tensor::xavier((in_features, out_features));
        let bias = Tensor::zeros((1, out_features));
        let input = None;

        Self {
            weights,
            bias,
            input,
        }
    }
    pub fn from_weights(weights: Tensor, bias: Tensor) -> Self {
        assert!(bias.shape() == (1, weights.shape().1));
        Self {
            weights,
            bias,
            input: None,
        }
    }
}
impl Layer for Linear {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        assert!(input.shape().1 == self.weights.shape().0);
        self.input = Some(input.clone());
        input.matmul(&self.weights).add_bias(&self.bias)
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
}
