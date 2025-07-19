use anyhow::{Context, Result};
use neural_net_mnist::multi_layer_perceptron::MultiLayerPerceptron;
use neural_net_mnist::training::{TrainingData, gradient_descent};
use neural_net_mnist::value::Value;
use std::fs::File;
use std::io::{self, BufRead};

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

fn per_iteration_callback(iter: usize, loss: f64, accuracy: f64) {
    println!("Iteration {iter}: loss = {loss}, accuracy = {accuracy}");
}

fn main() -> Result<()> {
    let data = load_training_data()?;
    let model = MultiLayerPerceptron::new(784, &[50, 10]);
    let iterations = 50;
    let learning_rate = |iter| 1.0 - iter as f64 / iterations as f64;

    gradient_descent(
        &model,
        &data[..1000],
        loss_function,
        accuracy_function,
        iterations,
        learning_rate,
        per_iteration_callback,
    );

    Ok(())
}
