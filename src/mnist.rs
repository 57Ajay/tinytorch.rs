use crate::tensor::Tensor;
use std::{assert_eq, fs, println, vec};

pub fn load_mnist_images(path: &str) -> Tensor {
    //    --- data layout ---
    //
    //    - Header: 16 bytes
    //      - bytes 0..4:   magic number (2051 in Big-Endian)
    //      - bytes 4..8:   number of images (Big-Endian u32)
    //      - bytes 8..12:  number of rows (28 in Big-Endian u32)
    //      - bytes 12..16: number of cols (28 in Big-Endian u32)
    //    - Payload: num_images * 28 * 28 bytes

    let data = fs::read(path).unwrap();
    let magic_num = u32::from_be_bytes(data[0..4].try_into().unwrap());
    assert!(magic_num == 2051);
    let num_img = u32::from_be_bytes(data[4..8].try_into().unwrap());

    let num_r = u32::from_be_bytes(data[8..12].try_into().unwrap());
    let num_c = u32::from_be_bytes(data[12..16].try_into().unwrap());
    assert!(num_r == 28 && num_c == 28);
    let payload_size = (num_img * num_r * num_c) as usize;
    assert!(payload_size == data[16..].len());

    let mut d = Vec::<f32>::with_capacity(payload_size);
    for &b in data[16..].iter() {
        d.push(f32::from(b) / 255.0);
    }

    Tensor::new(d, (num_img as usize, (num_r * num_c) as usize))
}

pub fn load_mnist_labels(path: &str) -> Tensor {
    //    - Header: 8 bytes
    //      - bytes 0..4: magic number (2049 in Big-Endian)
    //      - bytes 4..8: number of labels (Big-Endian u32)
    //    - Payload: num_labels bytes (each in 0..=9)

    let bytes = std::fs::read(path).unwrap();
    let magic_num = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert!(magic_num == 2049);

    let num_labels = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;

    let mut data = vec![0.0; num_labels * 10];
    for row in 0..num_labels {
        let label = bytes[8 + row] as usize;
        data[row * 10 + label] = 1.0;
    }

    Tensor::new(data, (num_labels, 10))
}

pub fn render_ascii(image: &[f32]) {
    assert_eq!(image.len(), 784);
    let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    for r in 0..28 {
        let mut line = String::with_capacity(28);
        for c in 0..28 {
            let pixel = image[r * 28 + c]; // 0.0 to 1.0
            let idx = (pixel * (chars.len() - 1) as f32).round() as usize;
            line.push(chars[idx]);
        }
        println!("{}", line);
    }
}

pub fn accuracy(pred: &Tensor, target: &Tensor) -> f32 {
    assert_eq!(pred.shape().0, target.shape().0);
    let n = pred.shape().0;
    let num_classes = pred.shape().1;

    let mut correct = 0;
    for i in 0..n {
        let mut max_pred_val = f32::NEG_INFINITY;
        let mut pred_class = 0;
        let mut max_target_val = f32::NEG_INFINITY;
        let mut target_class = 0;

        for c in 0..num_classes {
            let p = pred.get(i, c);
            if p > max_pred_val {
                max_pred_val = p;
                pred_class = c;
            }

            let t = target.get(i, c);
            if t > max_target_val {
                max_target_val = t;
                target_class = c;
            }
        }

        if pred_class == target_class {
            correct += 1;
        }
    }

    (correct as f32 / n as f32) * 100.0
}

#[cfg(test)]
mod test {
    use std::{assert_eq, vec};

    use super::*;

    #[test]
    fn test_load_mnist_train_dataset() {
        let images = load_mnist_images("data/train-images-idx3-ubyte");
        let labels = load_mnist_labels("data/train-labels-idx1-ubyte");

        assert_eq!(images.shape(), (60000, 784));
        assert_eq!(labels.shape(), (60000, 10));

        // Check normalization: all pixel values must be in [0.0, 1.0]
        assert!(images.data.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // Check one-hot: each row of labels must sum to 1.0
        for r in 0..100 {
            let mut sum = 0.0;
            for c in 0..10 {
                sum += labels.get(r, c);
            }
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_accuracy_metric() {
        // 3 samples, 3 classes
        let pred = Tensor::new(
            vec![
                0.9, 0.1, 0.0, // predicts class 0
                0.1, 0.2, 0.7, // predicts class 2
                0.8, 0.1, 0.1, // predicts class 0
            ],
            (3, 3),
        );
        let target = Tensor::new(
            vec![
                1.0, 0.0, 0.0, // class 0 -> MATCH
                0.0, 0.0, 1.0, // class 2 -> MATCH
                0.0, 1.0, 0.0, // class 1 -> MISMATCH
            ],
            (3, 3),
        );

        // 2 out of 3 correct = 66.66667%
        let acc = accuracy(&pred, &target);
        assert!((acc - 66.66667).abs() < 1e-3);
    }
}
