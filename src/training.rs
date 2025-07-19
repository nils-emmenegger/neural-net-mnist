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

pub fn gradient_descent(
    model: &MultiLayerPerceptron,
    training_data: &[TrainingData],
    mut loss_function: impl FnMut(&[Value], &[f64]) -> Value,
    mut accuracy_function: impl FnMut(&[Value], &[f64]) -> bool,
    iterations: usize,
    mut learning_rate: impl FnMut(usize) -> f64,
    mut per_iteration_callback: impl FnMut(usize, &MultiLayerPerceptron, f64, f64),
) {
    for iter in 0..iterations {
        struct Acc {
            total_loss: Value,
            num_accurate: usize,
        }

        let Acc {
            total_loss,
            num_accurate,
        } = training_data.iter().fold(
            Acc {
                total_loss: Value::new(0.0),
                num_accurate: 0,
            },
            |mut acc,
             TrainingData {
                 input,
                 expected_output,
             }| {
                let output =
                    model.forward(&input.iter().copied().map(Value::new).collect::<Vec<_>>());

                let loss = loss_function(&output, expected_output);
                acc.total_loss = &acc.total_loss + &loss;

                if accuracy_function(&output, expected_output) {
                    acc.num_accurate += 1;
                }

                acc
            },
        );

        let mut avg_loss = &total_loss / &Value::new(training_data.len() as f64);
        let avg_accuracy = num_accurate as f64 / (training_data.len() as f64);

        per_iteration_callback(iter, model, avg_loss.data(), avg_accuracy);

        avg_loss.backward();

        let learning_rate = learning_rate(iter);

        for mut param in model.parameters() {
            param.add_data(-param.grad() * learning_rate);
        }
    }
}
