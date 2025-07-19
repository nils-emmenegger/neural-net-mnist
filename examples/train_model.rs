use anyhow::{Context, Result};
use neural_net_mnist::training::TrainingData;
use std::fs::File;
use std::io::{self, BufRead};

fn load_training_data() -> Result<Vec<TrainingData>> {
    let file_path = "mnist_train.csv";
    let file = File::open(file_path).context("Failed to open file, download from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")?;
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

fn main() -> Result<()> {
    // check if mnist dataset has already been downloaded, throw error if not
    // check if model file exists, recreate if not
    // preferably this would display/graph/log some information about the loss function

    // Download dataset from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

    let data = load_training_data()?;

    Ok(())
}
