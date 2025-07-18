use crate::{layer::Layer, value::Value};

pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    pub fn new(num_inputs: usize, layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::with_capacity(layer_sizes.len());

        let mut last_size = num_inputs;
        for layer_size in layer_sizes.iter().copied() {
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
}
