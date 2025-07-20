use crate::{multi_layer_perceptron::MultiLayerPerceptron, value::Value};
use rand::{distr::Uniform, prelude::*};

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

pub fn gradient_descent<'a>(
    model: &MultiLayerPerceptron,
    training_data: impl Iterator<Item = &'a TrainingData>,
    mut loss_function: impl FnMut(&[Value], &[f64]) -> Value,
    mut accuracy_function: impl FnMut(&[Value], &[f64]) -> bool,
    iteration: usize,
    mut learning_rate: impl FnMut(usize) -> f64,
    mut per_iteration_callback: impl FnMut(usize, &MultiLayerPerceptron, f64, f64),
) {
    struct Acc {
        total_loss: Value,
        num_accurate: usize,
        batch_size: usize,
    }

    let Acc {
        total_loss,
        num_accurate,
        batch_size,
    } = training_data.fold(
        Acc {
            total_loss: Value::new(0.0),
            num_accurate: 0,
            batch_size: 0,
        },
        |mut acc,
         TrainingData {
             input,
             expected_output,
         }| {
            let output = model.forward(&input.iter().copied().map(Value::new).collect::<Vec<_>>());

            let loss = loss_function(&output, expected_output);
            acc.total_loss = &acc.total_loss + &loss;

            if accuracy_function(&output, expected_output) {
                acc.num_accurate += 1;
            }

            acc.batch_size += 1;

            acc
        },
    );

    let mut avg_loss = &total_loss / &Value::new(batch_size as f64);
    let avg_accuracy = num_accurate as f64 / (batch_size as f64);

    per_iteration_callback(iteration, model, avg_loss.data(), avg_accuracy);

    avg_loss.backward();

    let learning_rate = learning_rate(iteration);

    for mut param in model.parameters() {
        param.set_data(param.data() - param.grad() * learning_rate);
    }
}

struct RandomSampleIterator<'a> {
    data: &'a [TrainingData],
    generated: usize,
    batch_size: usize,
    rng: ThreadRng,
    distribution: Uniform<usize>,
}

impl<'a> RandomSampleIterator<'a> {
    fn new(
        data: &'a [TrainingData],
        batch_size: usize,
    ) -> Result<Self, rand::distr::uniform::Error> {
        Ok(Self {
            data,
            generated: 0,
            batch_size,
            rng: rand::rng(),
            distribution: Uniform::new(0, data.len())?,
        })
    }
}

impl<'a> Iterator for RandomSampleIterator<'a> {
    type Item = &'a TrainingData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.generated == self.batch_size {
            return None;
        }

        self.generated += 1;
        let index = self.distribution.sample(&mut self.rng);
        Some(&self.data[index])
    }
}

pub fn stochastic_gradient_descent(
    model: &MultiLayerPerceptron,
    training_data: &[TrainingData],
    batch_size: usize,
    iterations: usize,
    mut loss_function: impl FnMut(&[Value], &[f64]) -> Value,
    mut accuracy_function: impl FnMut(&[Value], &[f64]) -> bool,
    mut learning_rate: impl FnMut(usize) -> f64,
    mut per_iteration_callback: impl FnMut(usize, &MultiLayerPerceptron, f64, f64),
) {
    for iteration in 0..iterations {
        gradient_descent(
            model,
            RandomSampleIterator::new(training_data, batch_size).unwrap(),
            &mut loss_function,
            &mut accuracy_function,
            iteration,
            &mut learning_rate,
            &mut per_iteration_callback,
        );
    }
}
