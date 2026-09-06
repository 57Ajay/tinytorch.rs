use crate::tensor::Tensor;

pub trait Loss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> f32;
    fn backward(&self, pred: &Tensor, target: &Tensor) -> Tensor;
}

#[derive(Default)]
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MSELoss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> f32 {
        assert!(pred.shape == target.shape);

        let m = pred.data.len();
        let mut loss = pred
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(p, t)| (p - t).powf(2.0))
            .sum::<f32>();
        loss /= m as f32;

        loss
    }

    fn backward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        let shape = target.shape;
        let m = pred.data.len() as f32;

        let data = pred
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(p, t)| (2.0 / m) * (p - t))
            .collect::<Vec<f32>>();

        Tensor::new(data, shape)
    }
}

#[derive(Default)]
pub struct BCELoss;

impl BCELoss {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for BCELoss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> f32 {
        assert!(pred.shape == target.shape);
        let esp = 1e-7_f32;
        let m = pred.data.len();

        let mut loss = pred
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(p, t)| -(t * (p + esp).ln() + (1.0 - t) * (1.0 - p + esp).ln()))
            .sum::<f32>();

        loss /= m as f32;
        loss
    }

    fn backward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        let eps = 1e-7_f32;
        let m = pred.data.len() as f32;
        let shape = pred.shape;

        let data = pred
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(p, t)| (1.0 / m) * ((p - t) / ((p + eps) * (1.0 - p + eps))))
            .collect::<Vec<f32>>();

        Tensor::new(data, shape)
    }
}

#[derive(Default)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> f32 {
        let n = pred.shape.0 as f32;
        let eps = 1e-7_f32;

        let mut loss = pred
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(p, t)| t * ((p + eps).ln()))
            .sum::<f32>();

        loss /= n;
        -loss
    }
    fn backward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        let n = pred.shape.0 as f32;
        let eps = 1e-6_f32;

        let data = pred
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(p, t)| -(1.0 / n) * ((t) / (p + eps)))
            .collect::<Vec<f32>>();

        Tensor::new(data, pred.shape)
    }
}

#[cfg(test)]
mod test {
    use std::{assert_eq, vec};

    use super::*;

    #[test]
    fn test_mse_loss_zero() {
        let mse = MSELoss::new();
        let pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let target = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));

        let loss = mse.forward(&pred, &target);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_mse_loss_computation() {
        let mse = MSELoss::new();
        let pred = Tensor::new(vec![0.0, 1.0], (2, 1));
        let target = Tensor::new(vec![2.0, 5.0], (2, 1));

        // ( (0-2)^2 + (1-5)^2 ) / 2 = (4 + 16) / 2 = 10.0
        let loss = mse.forward(&pred, &target);
        assert!((loss - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_bce_loss_computation() {
        let bce = BCELoss::new();
        let pred = Tensor::new(vec![0.9, 0.1], (2, 1));
        let target = Tensor::new(vec![1.0, 0.0], (2, 1));

        // target=1: -ln(0.9) ≈ 0.1053605
        // target=0: -ln(1 - 0.1) = -ln(0.9) ≈ 0.1053605
        // mean = 0.1053605
        let loss = bce.forward(&pred, &target);
        assert!((loss - 0.10536).abs() < 1e-4);
    }

    #[test]
    #[should_panic]
    fn test_loss_shape_mismatch() {
        let mse = MSELoss::new();
        let pred = Tensor::zeros((2, 2));
        let target = Tensor::zeros((3, 2));
        let _ = mse.forward(&pred, &target);
    }

    #[test]
    fn test_mse_loss_backward() {
        let mse = MSELoss::new();
        let pred = Tensor::new(vec![2.0, 5.0], (2, 1));
        let target = Tensor::new(vec![1.0, 3.0], (2, 1));

        // M = 2
        // dL/dp_0 = 2/2 * (2 - 1) = 1.0
        // dL/dp_1 = 2/2 * (5 - 3) = 2.0
        let grad = mse.backward(&pred, &target);
        assert_eq!(grad.shape(), (2, 1));
        assert_eq!(grad.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_bce_loss_backward() {
        let bce = BCELoss::new();
        let pred = Tensor::new(vec![0.8, 0.2], (2, 1));
        let target = Tensor::new(vec![1.0, 0.0], (2, 1));

        // M = 2
        // For element 0: target = 1.0, pred = 0.8
        // dL/dp = (1/M) * (pred - target) / (pred * (1 - pred))
        //       = 0.5 * (0.8 - 1.0) / (0.8 * 0.2) = 0.5 * (-0.2) / 0.16 = -0.625
        // For element 1: target = 0.0, pred = 0.2
        // dL/dp = 0.5 * (0.2 - 0.0) / (0.2 * 0.8) = 0.5 * (0.2) / 0.16 = 0.625
        let grad = bce.backward(&pred, &target);
        assert_eq!(grad.shape(), (2, 1));
        assert!((grad.data[0] - (-0.625)).abs() < 1e-4);
        assert!((grad.data[1] - 0.625).abs() < 1e-4);
    }

    #[test]
    fn test_cross_entropy_loss_forward() {
        let ce = CrossEntropyLoss::new();
        // Batch of 2 samples, 3 classes each
        // Target: sample 0 is class 1 [0, 1, 0]; sample 1 is class 2 [0, 0, 1]
        let target = Tensor::new(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0], (2, 3));
        // Pred: probabilities from Softmax
        let pred = Tensor::new(vec![0.1, 0.8, 0.1, 0.2, 0.2, 0.6], (2, 3));

        // Sample 0: -ln(0.8) ≈ 0.22314355
        // Sample 1: -ln(0.6) ≈ 0.5108256
        // Batch mean = (0.22314355 + 0.5108256) / 2 = 0.36698458
        let loss = ce.forward(&pred, &target);
        assert!((loss - 0.36698).abs() < 1e-4);
    }

    #[test]
    fn test_cross_entropy_loss_backward() {
        let ce = CrossEntropyLoss::new();
        let target = Tensor::new(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0], (2, 3));
        let pred = Tensor::new(vec![0.1, 0.8, 0.1, 0.2, 0.2, 0.6], (2, 3));

        // dL/dp = -(1 / N) * (target / (pred + eps)) where N = 2
        let grad = ce.backward(&pred, &target);
        assert_eq!(grad.shape(), (2, 3));

        // Sample 0, class 1: -0.5 * (1.0 / 0.8) = -0.625
        assert!((grad.get(0, 1) - (-0.625)).abs() < 1e-4);
        // Sample 0, class 0: -0.5 * (0.0 / 0.1) = 0.0
        assert_eq!(grad.get(0, 0), 0.0);

        // Sample 1, class 2: -0.5 * (1.0 / 0.6) ≈ -0.83333
        assert!((grad.get(1, 2) - (-0.83333)).abs() < 1e-4);
    }
}
