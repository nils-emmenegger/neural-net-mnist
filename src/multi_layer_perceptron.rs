use std::iter;

use crate::{layer::Layer, value::Value};

pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    pub fn new(num_inputs: usize, hidden_layer_sizes: &[usize], num_outputs: usize) -> Self {
        let mut layers = Vec::with_capacity(hidden_layer_sizes.len() + 1);

        let mut last_size = num_inputs;
        for layer_size in hidden_layer_sizes
            .iter()
            .copied()
            .chain(iter::once(num_outputs))
        {
            layers.push(Layer::new(last_size, layer_size));
            last_size = layer_size;
        }

        MultiLayerPerceptron { layers }
    }

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        self.layers
            .iter()
            .fold(Vec::from(inputs), |acc, layer| layer.forward(&acc))
    }

    pub fn parameters(&self) -> impl Iterator<Item = Value> {
        self.layers.iter().flat_map(|layer| layer.paramters())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        let mlp = MultiLayerPerceptron::new(10, &[9, 5, 10], 1);
        assert_eq!(mlp.parameters().count(), 220);
    }
}
