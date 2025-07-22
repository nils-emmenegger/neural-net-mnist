use anyhow::{Context, Result};
use neural_net_mnist::{
    multi_layer_perceptron::MultiLayerPerceptron,
    training::{GradientDescentResult, TrainingData, stochastic_gradient_descent},
    value::Value,
};
use std::fs::File;
use std::io::{self, BufRead, Read, Write};

fn load_training_data() -> Result<Vec<TrainingData>> {
    let file_path = "mnist_train.csv";
    let file = File::open(file_path).with_context(|| {
        format!(
            "Failed to open {file_path}, \
            download it from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"
        )
    })?;
    let reader = io::BufReader::new(file);

    let mut data = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.context("Failed to read line")?;
        let mut tokens = line.split(',');

        let label = tokens
            .next()
            .context("Expected label column")?
            .parse::<u8>()
            .context("Failed to parse label column into u8")?;
        assert!(label <= 9);

        let expected_output = (0..10)
            .map(|i| if i == label { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();

        let input = tokens
            .map(|token| {
                Ok(token
                    .parse::<u8>()
                    .context("Failed to parse pixel column into u8")? as f64
                    / 255.0)
            })
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(input.len(), 784);

        data.push(TrainingData::new(input, expected_output));
    }

    assert_eq!(data.len(), 60_000);

    Ok(data)
}

fn write_model_to_file(model: &MultiLayerPerceptron, file: &File) -> Result<()> {
    let mut writer = io::BufWriter::new(file);

    for param in model.parameters() {
        writer
            .write_all(&param.data().to_le_bytes())
            .context("Failed to write model to file")?;
    }
    writer.flush().context("Failed to flush model to file")?;

    Ok(())
}

fn load_model_from_file(model: &MultiLayerPerceptron, file: &File) -> Result<()> {
    let mut reader = io::BufReader::new(file);
    let mut bytes = [0u8; 8];

    for mut param in model.parameters() {
        reader.read_exact(&mut bytes)?;
        param.set_data(f64::from_le_bytes(bytes));
    }

    if reader.read(&mut bytes)? != 0 {
        anyhow::bail!("Model file has extra unread bytes");
    }

    Ok(())
}

fn loss_function(output: &[Value], expected_output: &[f64]) -> Value {
    output
        .iter()
        .zip(expected_output.iter().copied())
        .map(|(o, e)| (o - &Value::new(e)).powf(2.0))
        .fold(Value::new(0.0), |acc, cur| &acc + &cur)
}

fn accuracy_function(output: &[Value], expected_output: &[f64]) -> bool {
    let mut max_output_index = 0;

    for (i, o) in output.iter().enumerate() {
        if o.data() > output[max_output_index].data() {
            max_output_index = i;
        }
    }

    let max_expected_index = expected_output.iter().position(|e| *e == 1.0).unwrap();

    max_output_index == max_expected_index
}

fn linearly_interpolate(start: f64, end: f64, iterations: usize) -> impl Fn(usize) -> f64 {
    let delta = end - start;
    let step = if iterations <= 1 {
        0.0
    } else {
        delta / (iterations - 1) as f64
    };
    move |iter| start + step * iter.min(iterations - 1) as f64
}

fn main() -> Result<()> {
    let data = load_training_data()?;
    let model = MultiLayerPerceptron::new(784, &[32, 16], 10);
    let batch_size = 100;
    let learning_rate = linearly_interpolate(0.5, 0.1, 150);

    let model_file = "model.bin";
    if let Ok(file) = File::open(model_file) {
        load_model_from_file(&model, &file)?;
    }

    let handle = std::thread::spawn(move || {
        let mut stdio = io::stdin().lock();
        let mut bytes = [0u8; 1];
        let _ = stdio.read(&mut bytes);
        println!("Received input, quitting...");
    });

    for iteration in 0.. {
        let GradientDescentResult {
            avg_loss,
            avg_accuracy,
        } = stochastic_gradient_descent(
            &model,
            &data,
            batch_size,
            iteration,
            loss_function,
            accuracy_function,
            &learning_rate,
        );

        let max_param = model
            .parameters()
            .map(|p| p.data().abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        println!(
            "Iteration {iteration:>4}: loss = {avg_loss:>8.5}, accuracy = {avg_accuracy:>8.5}, \
        maximum parameter = {max_param:>8.5}, learning rate = {:>8.5}",
            learning_rate(iteration)
        );

        if handle.is_finished() {
            break;
        }
    }

    write_model_to_file(&model, &File::create(model_file)?)?;

    Ok(())
}
