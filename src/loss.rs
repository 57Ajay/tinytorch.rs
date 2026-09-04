use crate::tensor::Tensor;

pub trait Loss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> f32;
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
}
