use selfnn::data::DataLoader;
use selfnn::layer::{Linear, ReLU, Softmax};
use selfnn::loss::{CrossEntropyLoss, Loss};
use selfnn::mnist::{accuracy, load_mnist_images, load_mnist_labels, render_ascii};
use selfnn::model::Sequential;
use selfnn::optim::Adam;
use std::time::Instant;
use std::{println, vec};

fn main() {
    println!("============================================================");
    println!("         Training MNIST Digit Classifier on selfnn          ");
    println!("============================================================");

    // -> Loading Data
    println!("Loading MNIST binary datasets...");
    let start_load = Instant::now();
    let train_x = load_mnist_images("data/train-images-idx3-ubyte");
    let train_y = load_mnist_labels("data/train-labels-idx1-ubyte");
    let test_x = load_mnist_images("data/t10k-images-idx3-ubyte");
    let test_y = load_mnist_labels("data/t10k-labels-idx1-ubyte");
    println!("   Loaded in {:.2?}", start_load.elapsed());

    // -> Build Model Architecture: 784 -> 128 -> 64 -> 10 -> Softmax
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(784, 128)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(128, 64)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(64, 10)),
        Box::new(Softmax::new()),
    ]);

    let loss_fn = CrossEntropyLoss::new();
    let optimizer = Adam::new(0.001);
    let batch_size = 64;
    let epochs = 5;

    let dataloader = DataLoader::new(train_x, train_y, batch_size, true);

    // -> Training Loop
    println!(
        "\nStarting training ({} epochs, batch_size = {})...",
        epochs, batch_size
    );
    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;
        let batches = dataloader.get_batches();
        let num_batches = batches.len();

        for (batch_x, batch_y) in &batches {
            let pred = model.forward(batch_x);
            let loss = loss_fn.forward(&pred, batch_y);
            total_loss += loss;

            let grad = loss_fn.backward(&pred, batch_y);
            model.backward(&grad);
            optimizer.step(&mut model);
            model.zero_grad();
        }

        let avg_loss = total_loss / num_batches as f32;
        println!(
            "Epoch [{}/{}] - Avg Loss: {:.4} - Time: {:.2?}",
            epoch,
            epochs,
            avg_loss,
            epoch_start.elapsed()
        );
    }

    // -> Test Evaluation
    println!("\nEvaluating on 10,000 unseen test images...");
    let test_pred = model.forward(&test_x);
    let test_acc = accuracy(&test_pred, &test_y);
    println!("Test Accuracy: {:.2}%!", test_acc);

    // -> Save Weights
    let weights_path = "data/mnist_weights.bin";
    model
        .save_weights(weights_path)
        .expect("Failed to save weights");
    println!("Saved trained weights to: {}", weights_path);

    // -> Visual Verification on 3 Sample Images
    println!("\nVisualizing predictions on sample test images:");
    for sample_idx in [0, 1, 2] {
        println!("\n───────────────────────────────────────────────");
        println!("Sample Test Image #{}", sample_idx);

        let sample_pixels = &test_x.slice_rows(sample_idx, sample_idx + 1).data;
        render_ascii(sample_pixels);

        let single_pred = model.forward(&test_x.slice_rows(sample_idx, sample_idx + 1));

        // predicted digit
        let mut best_digit = 0;
        let mut best_prob = 0.0;
        for c in 0..10 {
            let prob = single_pred.get(0, c);
            if prob > best_prob {
                best_prob = prob;
                best_digit = c;
            }
        }

        // actual digit
        let mut actual_digit = 0;
        for c in 0..10 {
            if test_y.get(sample_idx, c) == 1.0 {
                actual_digit = c;
            }
        }

        println!(
            "Model Prediction: {} (Confidence: {:.1}%) | Actual Label: {}",
            best_digit,
            best_prob * 100.0,
            actual_digit
        );
    }
}
