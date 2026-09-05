use crate::layer::Layer;
use crate::tensor::Tensor;

pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut iter = self.layers.iter_mut();

        let first_layer = match iter.next() {
            Some(layer) => layer.forward(input),
            None => return input.clone(),
        };

        iter.fold(first_layer, |acc, layer| layer.forward(&acc))
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut iter = self.layers.iter_mut().rev();

        let first_layer = match iter.next() {
            Some(layer) => layer.backward(grad_output),
            None => return grad_output.clone(),
        };

        iter.fold(first_layer, |acc, layer| layer.backward(&acc))
    }
    pub fn update(&mut self, lr: f32) {
        for layer in self.layers.iter_mut() {
            layer.update(lr);
        }
    }

    pub fn zero_grad(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_grad();
        }
    }
}
#[cfg(test)]
mod test {
    use std::{assert_eq, println, vec};

    use super::*;
    use crate::layer::{Linear, ReLU, Sigmoid};
    use crate::loss::{Loss, MSELoss};

    #[test]
    fn test_sequential_forward_shape() {
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(Linear::new(2, 4)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(4, 1)),
            Box::new(Sigmoid::new()),
        ];
        let mut model = Sequential::new(layers);

        let x = Tensor::ones((5, 2));
        let out = model.forward(&x);

        assert_eq!(out.shape(), (5, 1));
    }

    #[test]
    fn test_sequential_train_step_decreases_loss() {
        // Simple fitting: 1 sample
        let layers: Vec<Box<dyn Layer>> =
            vec![Box::new(Linear::new(2, 2)), Box::new(Linear::new(2, 1))];
        let mut model = Sequential::new(layers);
        let loss_fn = MSELoss::new();

        let x = Tensor::new(vec![1.0, 2.0], (1, 2));
        let y = Tensor::new(vec![5.0], (1, 1));

        let initial_pred = model.forward(&x);
        let initial_loss = loss_fn.forward(&initial_pred, &y);

        // Run 5 gradient descent steps with conservative learning rate to prevent overshoot
        for _ in 0..5 {
            let pred = model.forward(&x);
            let loss_grad = loss_fn.backward(&pred, &y);
            model.backward(&loss_grad);
            model.update(0.01);
            model.zero_grad();
        }

        let final_pred = model.forward(&x);
        let final_loss = loss_fn.forward(&final_pred, &y);

        assert!(
            final_loss < initial_loss,
            "Loss did not decrease! initial = {}, final = {}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_xor_milestone() {
        // XOR Truth Table:
        // [0, 0] -> 0
        // [0, 1] -> 1
        // [1, 0] -> 1
        // [1, 1] -> 0
        let x = Tensor::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], (4, 2));
        let y = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], (4, 1));

        // Architecture: Linear(2 -> 8) -> Sigmoid -> Linear(8 -> 1) -> Sigmoid
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(Linear::new(2, 8)),
            Box::new(Sigmoid::new()),
            Box::new(Linear::new(8, 1)),
            Box::new(Sigmoid::new()),
        ];
        let mut model = Sequential::new(layers);
        let loss_fn = MSELoss::new();

        let lr = 1.0_f32;
        let epochs = 3000;

        let mut final_loss = 1.0;
        for _ in 0..epochs {
            let pred = model.forward(&x);
            final_loss = loss_fn.forward(&pred, &y);

            let loss_grad = loss_fn.backward(&pred, &y);
            model.backward(&loss_grad);
            model.update(lr);
            model.zero_grad();
        }

        println!("XOR Training complete! Final Loss: {}", final_loss);
        assert!(
            final_loss < 0.05,
            "XOR failed to converge! Final loss = {}",
            final_loss
        );

        // Verify predictions
        let pred = model.forward(&x);
        // [0, 0] -> ~0
        assert!(
            pred.get(0, 0) < 0.25,
            "Expected ~0 for [0, 0], got {}",
            pred.get(0, 0)
        );
        // [0, 1] -> ~1
        assert!(
            pred.get(1, 0) > 0.75,
            "Expected ~1 for [0, 1], got {}",
            pred.get(1, 0)
        );
        // [1, 0] -> ~1
        assert!(
            pred.get(2, 0) > 0.75,
            "Expected ~1 for [1, 0], got {}",
            pred.get(2, 0)
        );
        // [1, 1] -> ~0
        assert!(
            pred.get(3, 0) < 0.25,
            "Expected ~0 for [1, 1], got {}",
            pred.get(3, 0)
        );
    }
}
