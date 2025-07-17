use crate::value::Value;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(num_inputs: usize) -> Self {
        Self {
            weights: vec![Value::new(0.0); num_inputs],
            bias: Value::new(0.0),
        }
    }

    pub fn forward(&self, activation: &[Value]) -> Value {
        assert_eq!(activation.len(), self.weights.len());
        activation
            .iter()
            .fold(self.bias.clone(), |acc, cur| &acc + cur)
            .tanh()
    }
}
