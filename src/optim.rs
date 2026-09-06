use crate::model::Sequential;

pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    pub fn step(&self, model: &mut Sequential) {
        model.update(self.lr);
    }
}

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    pub fn step(&self, model: &mut Sequential) {
        model.update_adam(self.lr, self.beta1, self.beta2, self.eps);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::layer::{Layer, Linear, Sigmoid};
    use crate::loss::{Loss, MSELoss};
    use crate::tensor::Tensor;

    #[test]
    fn test_sgd_step() {
        let layers: Vec<Box<dyn Layer>> = vec![Box::new(Linear::new(2, 1))];
        let mut model = Sequential::new(layers);
        let optimizer = SGD::new(0.05);
        let loss_fn = MSELoss::new();

        let x = Tensor::new(vec![1.0, 2.0], (1, 2));
        let y = Tensor::new(vec![3.0], (1, 1));

        let pred_0 = model.forward(&x);
        let loss_0 = loss_fn.forward(&pred_0, &y);

        let grad = loss_fn.backward(&pred_0, &y);
        model.backward(&grad);
        optimizer.step(&mut model);
        model.zero_grad();

        let pred_1 = model.forward(&x);
        let loss_1 = loss_fn.forward(&pred_1, &y);

        assert!(loss_1 < loss_0, "SGD step should decrease loss");
    }

    #[test]
    fn test_adam_single_step_analytical_math() {
        // Test exact theoretical calculation for step 1 of Adam
        let w = Tensor::new(vec![1.0], (1, 1));
        let b = Tensor::new(vec![0.5], (1, 1));
        let mut layer = Linear::from_weights(w, b);

        // Manually assign gradients
        layer.d_weights = Some(Tensor::new(vec![2.0], (1, 1)));
        layer.d_bias = Some(Tensor::new(vec![-3.0], (1, 1)));

        // Adam: lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8
        let lr = 0.1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        layer.update_adam(lr, beta1, beta2, eps);

        // At t = 1:
        // Weight:
        // m = (1 - 0.9) * 2.0 = 0.2
        // m_hat = 0.2 / (1 - 0.9) = 2.0
        // v = (1 - 0.999) * 4.0 = 0.004
        // v_hat = 0.004 / (1 - 0.999) = 4.0
        // sqrt(v_hat) = 2.0
        // delta = 0.1 * 2.0 / (2.0 + 1e-8) ≈ 0.1
        // new_w = 1.0 - 0.1 = 0.9
        assert!((layer.weights.data[0] - 0.9).abs() < 1e-5);

        // Bias:
        // m = (1 - 0.9) * (-3.0) = -0.3
        // m_hat = -0.3 / 0.1 = -3.0
        // v = (1 - 0.999) * 9.0 = 0.009
        // v_hat = 0.009 / 0.001 = 9.0
        // sqrt(v_hat) = 3.0
        // delta = 0.1 * (-3.0) / (3.0 + 1e-8) ≈ -0.1
        // new_b = 0.5 - (-0.1) = 0.6
        assert!((layer.bias.data[0] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_adam_training_convergence() {
        let x = Tensor::new(
            vec![
                0.0, 0.0,
                0.0, 1.0,
                1.0, 0.0,
                1.0, 1.0,
            ],
            (4, 2),
        );
        let y = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], (4, 1));

        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(Linear::new(2, 8)),
            Box::new(Sigmoid::new()),
            Box::new(Linear::new(8, 1)),
            Box::new(Sigmoid::new()),
        ];
        let mut model = Sequential::new(layers);
        let optimizer = Adam::new(0.05);
        let loss_fn = MSELoss::new();

        // With Adam, XOR learns rapidly and reliably!
        let mut loss = 1.0;
        for _ in 0..1500 {
            let pred = model.forward(&x);
            loss = loss_fn.forward(&pred, &y);

            let grad = loss_fn.backward(&pred, &y);
            model.backward(&grad);
            optimizer.step(&mut model);
            model.zero_grad();
        }

        assert!(loss < 0.05, "Adam failed to converge on XOR, loss = {}", loss);
    }
}
