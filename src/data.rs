use std::assert_eq;

use rand::{rng, seq::SliceRandom};

use crate::tensor::Tensor;

pub struct DataLoader {
    pub x: Tensor,
    pub y: Tensor,
    pub batch_size: usize,
    pub shuffle: bool,
}

impl DataLoader {
    pub fn new(x: Tensor, y: Tensor, batch_size: usize, shuffle: bool) -> Self {
        assert_eq!(
            x.shape().0,
            y.shape().0,
            "X and Y must have same number of samples"
        );
        Self {
            x,
            y,
            batch_size,
            shuffle,
        }
    }

    pub fn get_batches(&self) -> Vec<(Tensor, Tensor)> {
        let n = self.x.shape.0;

        let (x_source, y_source) = if self.shuffle {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng());
            (self.x.get_rows(&indices), self.y.get_rows(&indices))
        } else {
            (self.x.clone(), self.y.clone())
        };

        let mut batches = Vec::new();
        let n = self.x.shape().0;
        let mut start = 0;

        while start < n {
            let end = (start + self.batch_size).min(n);
            let batch_x = x_source.slice_rows(start, end);
            let batch_y = y_source.slice_rows(start, end);
            batches.push((batch_x, batch_y));
            start = end;
        }

        batches
    }
}

#[cfg(test)]
mod test {
    use std::vec;

    use super::*;

    #[test]
    fn test_dataloader_batch_shapes_and_remainder() {
        // 10 samples total, batch_size = 4 -> 3 batches (4, 4, 2)
        let x_data: Vec<f32> = (0..20).map(|i| i as f32).collect(); // 10 samples, 2 features
        let y_data: Vec<f32> = (0..10).map(|i| i as f32).collect(); // 10 samples, 1 target

        let x = Tensor::new(x_data, (10, 2));
        let y = Tensor::new(y_data, (10, 1));

        let loader = DataLoader::new(x, y, 4, false);
        let batches = loader.get_batches();

        assert_eq!(batches.len(), 3);

        // Batch 0
        assert_eq!(batches[0].0.shape(), (4, 2));
        assert_eq!(batches[0].1.shape(), (4, 1));

        // Batch 1
        assert_eq!(batches[1].0.shape(), (4, 2));
        assert_eq!(batches[1].1.shape(), (4, 1));

        // Batch 2 (remainder)
        assert_eq!(batches[2].0.shape(), (2, 2));
        assert_eq!(batches[2].1.shape(), (2, 1));
    }

    #[test]
    fn test_dataloader_without_shuffle_order() {
        let x = Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], (6, 1));
        let y = Tensor::new(vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0], (6, 1));

        let loader = DataLoader::new(x, y, 2, false);
        let batches = loader.get_batches();

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].0.data, vec![0.0, 1.0]);
        assert_eq!(batches[0].1.data, vec![0.0, 10.0]);

        assert_eq!(batches[1].0.data, vec![2.0, 3.0]);
        assert_eq!(batches[1].1.data, vec![20.0, 30.0]);

        assert_eq!(batches[2].0.data, vec![4.0, 5.0]);
        assert_eq!(batches[2].1.data, vec![40.0, 50.0]);
    }

    #[test]
    fn test_dataloader_with_shuffle_conservation() {
        let x = Tensor::new((0..50).map(|i| i as f32).collect(), (50, 1));
        let y = Tensor::new((0..50).map(|i| (i * 2) as f32).collect(), (50, 1));

        let loader = DataLoader::new(x, y, 10, true);
        let batches = loader.get_batches();

        assert_eq!(batches.len(), 5);

        // Collect all target items across all batches
        let mut seen_y = Vec::new();
        for (_, batch_y) in batches {
            seen_y.extend(batch_y.data);
        }

        // All 50 original elements must be present
        seen_y.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected: Vec<f32> = (0..50).map(|i| (i * 2) as f32).collect();
        assert_eq!(seen_y, expected);
    }
}
