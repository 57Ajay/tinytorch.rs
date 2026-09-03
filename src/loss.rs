use crate::tensor::Tensor;

pub trait Loss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> f32;
}

// =========================================================================
// PADAWAN: Implement `MSELoss` and `BCELoss` below!
//
// 1. MSELoss:
//    L = (1 / M) * sum((pred_i - target_i)^2)
//    where M is total elements (pred.data.len()).
//
// 2. BCELoss:
//    L = - (1 / M) * sum( target_i * ln(pred_i + eps) + (1 - target_i) * ln(1 - pred_i + eps) )
//    Use eps = 1e-7 to prevent ln(0).
// =========================================================================

#[cfg(test)]
mod test {
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
