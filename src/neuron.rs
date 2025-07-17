use crate::value::Value;
use rand::{distr::Uniform, prelude::*};

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(num_inputs: usize) -> Self {
        let rng = rand::rng();
        let dist = Uniform::new_inclusive(-1.0, 1.0).unwrap();

        Self {
            weights: dist
                .sample_iter(rng)
                .take(num_inputs)
                .map(|val| Value::new(val))
                .collect::<Vec<_>>(),
            bias: Value::new(0.0),
        }
    }

    pub fn forward(&self, activations: &[Value]) -> Value {
        assert_eq!(activations.len(), self.weights.len());
        activations
            .iter()
            .zip(&self.weights)
            .fold(self.bias.clone(), |acc, (activation, weight)| {
                &acc + &(activation * weight)
            })
            .tanh()
    }
}
