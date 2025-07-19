use crate::{multi_layer_perceptron::MultiLayerPerceptron, value::Value};

pub struct TrainingData {
    pub input: Vec<f64>,
    pub expected_output: Vec<f64>,
}

impl TrainingData {
    pub fn new(input: Vec<f64>, expected_output: Vec<f64>) -> Self {
        Self {
            input,
            expected_output,
        }
    }
}

struct Output {
    output: Vec<Value>,
    loss: Value,
}

pub fn gradient_descent(
    model: &MultiLayerPerceptron,
    training_data: &[TrainingData],
    mut loss_function: impl FnMut(&[Value], &[f64]) -> Value,
    iterations: usize,
    learning_rate: impl FnMut(usize) -> f64,
    per_iteration_callback: impl FnMut(usize, f64, f64),
) {
    for iter in 0..iterations {
        let outputs = training_data
            .iter()
            .map(
                |TrainingData {
                     input,
                     expected_output,
                 }| {
                    let output =
                        model.forward(&input.iter().copied().map(Value::new).collect::<Vec<_>>());
                    let loss = loss_function(&output, &expected_output);

                    Output { output, loss }
                },
            )
            .collect::<Vec<_>>();
    }
}
