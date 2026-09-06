use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    fn update(&mut self, _lr: f32) {}
    fn zero_grad(&mut self) {}
    fn update_adam(&mut self, _lr: f32, _beta1: f32, _beta2: f32, _eps: f32) {}
}

pub struct Linear {
    pub weights: Tensor,
    pub bias: Tensor,
    pub input: Option<Tensor>,
    pub d_weights: Option<Tensor>,
    pub d_bias: Option<Tensor>,

    // these are all for adam optemizer
    pub m_weights: Option<Tensor>,
    pub v_weights: Option<Tensor>,
    pub m_bias: Option<Tensor>,
    pub v_bias: Option<Tensor>,
    pub t: usize,
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
            m_weights: None,
            v_weights: None,
            m_bias: None,
            v_bias: None,
            t: 0,
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
            m_weights: None,
            v_weights: None,
            m_bias: None,
            v_bias: None,
            t: 0,
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

    fn update(&mut self, _lr: f32) {
        self.weights = self
            .weights
            .sub(&self.d_weights.as_ref().unwrap().scale(_lr));

        self.bias = self.bias.sub(&self.d_bias.as_ref().unwrap().scale(_lr));
    }

    fn zero_grad(&mut self) {
        self.d_weights = None;
        self.d_bias = None;
    }

    #[allow(clippy::needless_range_loop)]
    fn update_adam(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        self.t += 1;

        if self.m_weights.is_none() {
            self.m_weights = Some(Tensor::zeros(self.weights.shape));
        }
        if self.v_weights.is_none() {
            self.v_weights = Some(Tensor::zeros(self.weights.shape));
        }
        if self.m_bias.is_none() {
            self.m_bias = Some(Tensor::zeros(self.bias.shape));
        }
        if self.v_bias.is_none() {
            self.v_bias = Some(Tensor::zeros(self.bias.shape));
        }

        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);

        let dw = &self.d_weights.as_ref().expect("d_weights missing").data;
        let db = &self.d_bias.as_ref().expect("d_bias missing").data;

        let mw = self.m_weights.as_mut().unwrap();
        let vw = self.v_weights.as_mut().unwrap();

        for i in 0..self.weights.data.len() {
            mw.data[i] = beta1 * mw.data[i] + (1.0 - beta1) * dw[i];
            vw.data[i] = beta2 * vw.data[i] + (1.0 - beta2) * dw[i].powi(2);

            let m_hat = mw.data[i] / bias_correction1;
            let v_hat = vw.data[i] / bias_correction2;

            let delta = (lr * m_hat) / (v_hat.sqrt() + eps);
            self.weights.data[i] -= delta;
        }

        let mb = self.m_bias.as_mut().unwrap();
        let vb = self.v_bias.as_mut().unwrap();

        for i in 0..self.bias.data.len() {
            mb.data[i] = beta1 * mb.data[i] + (1.0 - beta1) * db[i];
            vb.data[i] = beta2 * vb.data[i] + (1.0 - beta2) * db[i].powi(2);

            let m_hat = mb.data[i] / bias_correction1;
            let v_hat = vb.data[i] / bias_correction2;

            let delta = (lr * m_hat) / (v_hat.sqrt() + eps);
            self.bias.data[i] -= delta;
        }
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

#[derive(Default)]
pub struct Softmax {
    pub output: Option<Tensor>,
}

impl Softmax {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut output = Vec::<f32>::with_capacity(input.data.len());

        let c = input.shape.1;
        let r_ = input.shape.0;

        for r in 0..r_ {
            let mut r_vec = Vec::with_capacity(c);
            let mut m = input.get(r, 0);
            let mut sum = 0.0;

            for j in 0..c {
                let v = input.get(r, j);
                if v > m {
                    m = v;
                }
                r_vec.push(v);
            }

            for v in r_vec.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }

            for v in r_vec {
                output.push(v / sum);
            }
        }

        let tensor = Tensor::new(output, input.shape);
        self.output = Some(tensor.clone());
        tensor
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let p = self.output.as_ref().unwrap();

        let (r, c) = grad_output.shape;
        let mut data = Vec::<f32>::with_capacity(r * c);

        for i in 0..r {
            let mut row_sum = 0.0;
            for j in 0..c {
                row_sum += p.get(i, j) * grad_output.get(i, j);
            }

            for j in 0..c {
                let p_ij = p.get(i, j);
                let g_ij = grad_output.get(i, j);
                let grad = p_ij * (g_ij - row_sum);
                data.push(grad);
            }
        }

        Tensor::new(data, (r, c))
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

    #[test]
    fn test_numerical_gradient_check_linear() {
        use crate::loss::{Loss, MSELoss};

        let x = Tensor::new(vec![1.5, -0.5, 2.0, 1.0], (2, 2));
        let target = Tensor::new(vec![0.8, 1.2], (2, 1));

        let weights = Tensor::new(vec![0.4, -0.7], (2, 1));
        let bias = Tensor::new(vec![0.2], (1, 1));

        let mut layer = Linear::from_weights(weights, bias);
        let loss_fn = MSELoss::new();

        // 1. Analytical backward pass
        let pred = layer.forward(&x);
        let loss_grad = loss_fn.backward(&pred, &target);
        let _ = layer.backward(&loss_grad);

        let analytical_dw = layer.d_weights.as_ref().unwrap().clone();
        let analytical_db = layer.d_bias.as_ref().unwrap().clone();

        let eps = 1e-3_f32;

        // 2. Numerical gradient check for weights
        for i in 0..layer.weights.data.len() {
            let orig = layer.weights.data[i];

            layer.weights.data[i] = orig + eps;
            let pred_pos = layer.forward(&x);
            let loss_pos = loss_fn.forward(&pred_pos, &target);

            layer.weights.data[i] = orig - eps;
            let pred_neg = layer.forward(&x);
            let loss_neg = loss_fn.forward(&pred_neg, &target);

            layer.weights.data[i] = orig;

            let num_grad = (loss_pos - loss_neg) / (2.0 * eps);
            let ana_grad = analytical_dw.data[i];

            let diff = (num_grad - ana_grad).abs();
            let norm = num_grad.abs().max(ana_grad.abs()).max(1e-6);
            let rel_error = diff / norm;

            assert!(
                rel_error < 1e-3,
                "Weight gradient check failed at index {}: numerical = {}, analytical = {}, rel_error = {}",
                i,
                num_grad,
                ana_grad,
                rel_error
            );
        }

        // 3. Numerical gradient check for bias
        for i in 0..layer.bias.data.len() {
            let orig = layer.bias.data[i];

            layer.bias.data[i] = orig + eps;
            let pred_pos = layer.forward(&x);
            let loss_pos = loss_fn.forward(&pred_pos, &target);

            layer.bias.data[i] = orig - eps;
            let pred_neg = layer.forward(&x);
            let loss_neg = loss_fn.forward(&pred_neg, &target);

            layer.bias.data[i] = orig;

            let num_grad = (loss_pos - loss_neg) / (2.0 * eps);
            let ana_grad = analytical_db.data[i];

            let diff = (num_grad - ana_grad).abs();
            let norm = num_grad.abs().max(ana_grad.abs()).max(1e-6);
            let rel_error = diff / norm;

            assert!(
                rel_error < 1e-3,
                "Bias gradient check failed at index {}: numerical = {}, analytical = {}, rel_error = {}",
                i,
                num_grad,
                ana_grad,
                rel_error
            );
        }
    }

    #[test]
    fn test_numerical_gradient_check_two_layer() {
        use crate::loss::{Loss, MSELoss};

        // Two layer chain: Linear -> Sigmoid -> MSELoss
        let x = Tensor::new(vec![0.5, -0.2, 0.1, 0.8], (2, 2));
        let target = Tensor::new(vec![0.7, 0.3], (2, 1));

        let weights = Tensor::new(vec![0.6, -0.4], (2, 1));
        let bias = Tensor::new(vec![0.1], (1, 1));

        let mut linear = Linear::from_weights(weights, bias);
        let mut sigmoid = Sigmoid::new();
        let loss_fn = MSELoss::new();

        // Forward
        let a1 = linear.forward(&x);
        let pred = sigmoid.forward(&a1);

        // Backward
        let loss_grad = loss_fn.backward(&pred, &target);
        let grad_a1 = sigmoid.backward(&loss_grad);
        let _ = linear.backward(&grad_a1);

        let analytical_dw = linear.d_weights.as_ref().unwrap().clone();
        let analytical_db = linear.d_bias.as_ref().unwrap().clone();

        let eps = 1e-3_f32;

        // Check weights
        for i in 0..linear.weights.data.len() {
            let orig = linear.weights.data[i];

            linear.weights.data[i] = orig + eps;
            let p_pos = sigmoid.forward(&linear.forward(&x));
            let l_pos = loss_fn.forward(&p_pos, &target);

            linear.weights.data[i] = orig - eps;
            let p_neg = sigmoid.forward(&linear.forward(&x));
            let l_neg = loss_fn.forward(&p_neg, &target);

            linear.weights.data[i] = orig;

            let num_grad = (l_pos - l_neg) / (2.0 * eps);
            let ana_grad = analytical_dw.data[i];

            let diff = (num_grad - ana_grad).abs();
            let norm = num_grad.abs().max(ana_grad.abs()).max(1e-6);
            let rel_error = diff / norm;

            assert!(
                rel_error < 1e-3,
                "Two-layer weight gradient failed at index {}: num = {}, ana = {}, rel_err = {}",
                i,
                num_grad,
                ana_grad,
                rel_error
            );
        }

        // Check bias
        for i in 0..linear.bias.data.len() {
            let orig = linear.bias.data[i];

            linear.bias.data[i] = orig + eps;
            let p_pos = sigmoid.forward(&linear.forward(&x));
            let l_pos = loss_fn.forward(&p_pos, &target);

            linear.bias.data[i] = orig - eps;
            let p_neg = sigmoid.forward(&linear.forward(&x));
            let l_neg = loss_fn.forward(&p_neg, &target);

            linear.bias.data[i] = orig;

            let num_grad = (l_pos - l_neg) / (2.0 * eps);
            let ana_grad = analytical_db.data[i];

            let diff = (num_grad - ana_grad).abs();
            let norm = num_grad.abs().max(ana_grad.abs()).max(1e-6);
            let rel_error = diff / norm;

            assert!(
                rel_error < 1e-3,
                "Two-layer bias gradient failed at index {}: num = {}, ana = {}, rel_err = {}",
                i,
                num_grad,
                ana_grad,
                rel_error
            );
        }
    }

    #[test]
    fn test_softmax_forward_basic() {
        let mut softmax = Softmax::new();
        // Shape (2, 3)
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0], (2, 3));
        let out = softmax.forward(&x);

        assert_eq!(out.shape(), (2, 3));

        // Check row 0: sum should be 1.0, and out[0, 2] > out[0, 1] > out[0, 0]
        let sum_row0 = out.get(0, 0) + out.get(0, 1) + out.get(0, 2);
        assert!((sum_row0 - 1.0).abs() < 1e-5);
        assert!(out.get(0, 2) > out.get(0, 1) && out.get(0, 1) > out.get(0, 0));

        // Check row 1: equal inputs [0, 0, 0] should produce equal probabilities [1/3, 1/3, 1/3]
        let sum_row1 = out.get(1, 0) + out.get(1, 1) + out.get(1, 2);
        assert!((sum_row1 - 1.0).abs() < 1e-5);
        assert!((out.get(1, 0) - 1.0 / 3.0).abs() < 1e-5);
        assert!((out.get(1, 1) - 1.0 / 3.0).abs() < 1e-5);
        assert!((out.get(1, 2) - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut softmax = Softmax::new();
        // Large values that would overflow exp() without max subtraction!
        let x = Tensor::new(vec![1000.0, 1001.0, 1002.0], (1, 3));
        let out = softmax.forward(&x);

        // Check no NaNs or Infs
        for &val in &out.data {
            assert!(
                f32::is_finite(val),
                "Softmax output must be finite, got {}",
                val
            );
        }

        // Sum must still be 1.0
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Relative ratios: exp(1002 - 1002)=1.0, exp(1001 - 1002)=e^-1, exp(1000 - 1002)=e^-2
        let e2 = (-2.0_f32).exp();
        let e1 = (-1.0_f32).exp();
        let e0 = 1.0_f32;
        let expected_sum = e2 + e1 + e0;
        assert!((out.get(0, 0) - e2 / expected_sum).abs() < 1e-5);
        assert!((out.get(0, 1) - e1 / expected_sum).abs() < 1e-5);
        assert!((out.get(0, 2) - e0 / expected_sum).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_backward() {
        let mut softmax = Softmax::new();
        let x = Tensor::new(vec![0.0, 0.0], (1, 2));
        let _ = softmax.forward(&x); // p = [0.5, 0.5]

        let grad_output = Tensor::new(vec![1.0, 0.0], (1, 2));
        let grad_input = softmax.backward(&grad_output);

        // S = grad_0 * p_0 + grad_1 * p_1 = 1.0*0.5 + 0.0*0.5 = 0.5
        // grad_in_0 = p_0 * (grad_0 - S) = 0.5 * (1.0 - 0.5) = 0.25
        // grad_in_1 = p_1 * (grad_1 - S) = 0.5 * (0.0 - 0.5) = -0.25
        assert_eq!(grad_input.shape(), (1, 2));
        assert!((grad_input.get(0, 0) - 0.25).abs() < 1e-5);
        assert!((grad_input.get(0, 1) - (-0.25)).abs() < 1e-5);
    }
}
