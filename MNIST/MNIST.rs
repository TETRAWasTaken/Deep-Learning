//! ```cargo
//! [dependencies]
//! NeuralFlow = { path = "/Users/anshumaansoni/RustroverProjects/NeuralFlow" }
//! ```

use NeuralFlow::prelude::*;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn load_mnist(
    image_path: &str,
    label_path: &str,
    max_samples: Option<usize>,
) -> std::io::Result<TensorDataset> {
    let mut img_file = File::open(image_path)?;
    let mut lbl_file = File::open(label_path)?;

    let mut img_buf = Vec::new();
    let mut lbl_buf = Vec::new();

    img_file.read_to_end(&mut img_buf)?;
    lbl_file.read_to_end(&mut lbl_buf)?;

    let total_images = u32::from_be_bytes(img_buf[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(img_buf[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(img_buf[12..16].try_into().unwrap()) as usize;
    let image_pixels = rows * cols;

    let total_labels = u32::from_be_bytes(lbl_buf[4..8].try_into().unwrap()) as usize;
    assert_eq!(total_images, total_labels);

    let count = match max_samples {
        Some(m) => m.min(total_images),
        None => total_images,
    };

    let mut all_images = Vec::with_capacity(count);
    let mut all_labels = Vec::with_capacity(count);

    for i in 0..count {
        let img_start = 16 + i * image_pixels;
        let img_end = img_start + image_pixels;
        let pixels: Vec<f32> = img_buf[img_start..img_end]
            .iter()
            .map(|&b| (b as f32) / 255.0)
            .collect();

        let label = lbl_buf[8 + i] as f32;

        all_images.push(pixels);
        all_labels.push(vec![label]);
    }

    Ok(TensorDataset::new(all_images, all_labels))
}

#[allow(non_snake_case)]
fn MNIST() -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(784, 256)),
        Box::new(ReLu),
        Box::new(Dropout::new(0.2)),
        Box::new(Linear::new(256, 128)),
        Box::new(ReLu),
        Box::new(Dropout::new(0.2)),
        Box::new(Linear::new(128, 10)),
    ])
}

fn count_parameters(model: &Sequential) -> usize {
    model.parameters().iter().map(|p| p.0.borrow().data.len()).sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (train_images_path, train_labels_path, test_images_path, test_labels_path) =
        if std::path::Path::new("data/mnist/train-images-idx3-ubyte").exists() {
            (
                "data/mnist/train-images-idx3-ubyte",
                "data/mnist/train-labels-idx1-ubyte",
                "data/mnist/t10k-images-idx3-ubyte",
                "data/mnist/t10k-labels-idx1-ubyte",
            )
        } else {
            (
                "/Users/anshumaansoni/RustroverProjects/NeuralFlow/data/mnist/train-images-idx3-ubyte",
                "/Users/anshumaansoni/RustroverProjects/NeuralFlow/data/mnist/train-labels-idx1-ubyte",
                "/Users/anshumaansoni/RustroverProjects/NeuralFlow/data/mnist/t10k-images-idx3-ubyte",
                "/Users/anshumaansoni/RustroverProjects/NeuralFlow/data/mnist/t10k-labels-idx1-ubyte",
            )
        };

    let load_start = Instant::now();
    let train_data = load_mnist(train_images_path, train_labels_path, None)?;
    let test_data = load_mnist(test_images_path, test_labels_path, None)?;
    println!(
        "Loaded {} train samples, {} test samples in {:.2}s",
        train_data.len(),
        test_data.len(),
        load_start.elapsed().as_secs_f32()
    );

    let batch_size = 128;
    let epochs = 50;
    let learning_rate = 0.001;

    let mut train_loader = Dataloader::new(&train_data, batch_size, true);
    let mut test_loader = Dataloader::new(&test_data, batch_size, false);

    let model = MNIST();
    let total_params = count_parameters(&model);
    println!("Model initialized with {} trainable parameters", total_params);
    
    let mut optimizer = Adam::new(learning_rate, 0.9, 0.999, 1e-8, 0.0);

    println!("{:-<75}", "");
    println!(
        "{:<6} | {:<10} | {:<9} | {:<10} | {:<8} | {:<12}",
        "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Throughput"
    );
    println!("{:-<75}", "");

    let overall_start = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();

        model.train();
        let mut total_train_loss = 0.0;
        let mut train_correct = 0;
        let mut train_samples = 0;

        for (x_batch, y_batch) in train_loader.iter_batches() {
            let current_b = x_batch.shape().0;

            model.zero_grad();
            
            let logits = model.forward(&x_batch);
            let loss = cross_entropy_loss(&logits, &y_batch);

            loss.backward();
            optimizer.step(&model.parameters());

            let loss_val = loss.0.borrow().data[0];
            total_train_loss += loss_val * current_b as f32;

            let acc = Metrics::accuracy_logits(&logits.0.borrow().data, &y_batch.0.borrow().data, 10);
            train_correct += (acc * current_b as f32).round() as usize;
            train_samples += current_b;
        }

        let train_acc = train_correct as f32 / train_samples as f32;
        let avg_train_loss = total_train_loss / train_samples as f32;

        model.eval();
        let _guard = no_grad();
        let mut total_val_loss = 0.0;
        let mut val_correct = 0;
        let mut val_samples = 0;

        for (x_batch, y_batch) in test_loader.iter_batches() {
            let current_b = x_batch.shape().0;

            let logits = model.forward(&x_batch);
            let loss = cross_entropy_loss(&logits, &y_batch);

            let loss_val = loss.0.borrow().data[0];
            total_val_loss += loss_val * current_b as f32;

            let acc = Metrics::accuracy_logits(&logits.0.borrow().data, &y_batch.0.borrow().data, 10);
            val_correct += (acc * current_b as f32).round() as usize;
            val_samples += current_b;
        }

        let val_acc = val_correct as f32 / val_samples as f32;
        let avg_val_loss = total_val_loss / val_samples as f32;

        let epoch_duration = epoch_start.elapsed().as_secs_f32();
        let throughput = train_samples as f32 / epoch_duration;

        println!(
            "{:<6} | {:<10.4} | {:<9.4} | {:<10.4} | {:<8.4} | {:<12.0}",
            epoch,
            avg_train_loss,
            train_acc,
            avg_val_loss,
            val_acc,
            throughput
        );
    }

    let total_duration = overall_start.elapsed().as_secs_f32();
    println!("{:-<75}", "");
    println!("Training completed in {:.2}s", total_duration);

    Ok(())
}