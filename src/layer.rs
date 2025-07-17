use crate::{neuron::Neuron, value::Value};

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_neurons: usize) -> Self {
        Self {
            neurons: (0..num_neurons)
                .map(|_| Neuron::new(num_inputs))
                .collect::<Vec<_>>(),
        }
    }

    pub fn forward(&self, activations: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(activations))
            .collect::<Vec<_>>()
    }
}
