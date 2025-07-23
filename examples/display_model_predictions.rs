use iced::{
    Element, Font,
    widget::{button, column, row, text},
};
use std::{
    fs::File,
    io::{self, BufRead, Read},
};

const WIDTH: u32 = 28;
const HEIGHT: u32 = 28;

struct Neuron {
    bias: f64,
    weights: Vec<f64>,
}

impl Neuron {
    fn forward(&self, activations: &[f64]) -> f64 {
        assert_eq!(activations.len(), self.weights.len());

        (self.bias
            + activations
                .iter()
                .zip(self.weights.iter())
                .map(|(input, weight)| input * weight)
                .sum::<f64>())
        .tanh()
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn forward(&self, activations: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(activations))
            .collect()
    }
}

struct Model {
    layers: Vec<Layer>,
}

impl Model {
    fn new(layer_sizes: &[usize], data: &[f64]) -> Self {
        let mut layers = Vec::new();

        let mut data_iter = data.iter();

        for window in layer_sizes.windows(2) {
            let &[prv_layer_size, cur_layer_size] = window.try_into().unwrap();
            let mut neurons = Vec::with_capacity(cur_layer_size);

            for _ in 0..cur_layer_size {
                let weights = data_iter
                    .by_ref()
                    .take(prv_layer_size)
                    .copied()
                    .collect::<Vec<_>>();
                assert_eq!(weights.len(), prv_layer_size);
                let bias = *data_iter.next().unwrap();
                neurons.push(Neuron { bias, weights });
            }

            layers.push(Layer { neurons });
        }

        assert!(data_iter.next().is_none());

        Model { layers }
    }

    fn forward(&self, activations: &[f64]) -> Vec<f64> {
        self.layers
            .iter()
            .fold(activations.to_vec(), |activations, layer| {
                layer.forward(&activations)
            })
    }

    fn get_prediction(&self, activations: &[f64]) -> u8 {
        let output = self.forward(activations);
        assert_eq!(output.len(), 10);

        let mut max_output_index = 0;

        for (i, o) in output.iter().copied().enumerate() {
            if o > output[max_output_index] {
                max_output_index = i;
            }
        }

        max_output_index as u8
    }
}

fn load_model_from_file(layers: &[usize], file: &File) -> Model {
    let mut reader = io::BufReader::new(file);
    let mut bytes = Vec::new();

    reader
        .read_to_end(&mut bytes)
        .expect("Failed to read model file");

    let data = bytes
        .chunks(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();

    Model::new(layers, &data)
}

struct Data {
    label: u8,
    image_data: Vec<u8>,
}

fn load_training_data() -> Vec<Data> {
    let file_path = "mnist_train.csv";
    let file = File::open(file_path)
        .expect("Failed to open file, download it from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv");
    let reader = io::BufReader::new(file);

    let mut data = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.expect("Failed to read line");
        let mut tokens = line.split(',');

        let label = tokens
            .next()
            .expect("Expected label column")
            .parse::<u8>()
            .expect("Failed to parse label column into u8");
        assert!(label <= 9);

        let image_data = tokens
            .map(|token| {
                token
                    .parse::<u8>()
                    .expect("Failed to parse pixel column into u8")
            })
            .collect::<Vec<_>>();
        assert_eq!(image_data.len(), 784);

        data.push(Data { label, image_data });
    }

    assert_eq!(data.len(), 60_000);

    data
}

struct Viewer {
    model: Model,
    data: Vec<Data>,
    data_index: usize,
}

#[derive(Debug, Clone)]
enum Message {
    Next,
    Previous,
}

impl Default for Viewer {
    fn default() -> Self {
        let model = load_model_from_file(&[784, 40, 10], &File::open("model.bin").unwrap());
        let data = load_training_data();

        Self {
            model,
            data,
            data_index: 0,
        }
    }
}

impl Viewer {
    fn update(&mut self, message: Message) {
        match message {
            Message::Next => self.data_index = (self.data_index + 1) % self.data.len(),
            Message::Previous => {
                self.data_index = (self.data_index + self.data.len() - 1) % self.data.len()
            }
        }
    }

    fn view(&self) -> Element<Message> {
        let image_data = self.data[self.data_index]
            .image_data
            .iter()
            .flat_map(|pixel| [*pixel, *pixel, *pixel, 255])
            .collect::<Vec<_>>();

        let label = self.data[self.data_index].label;

        let prediction = self.model.get_prediction(
            &self.data[self.data_index]
                .image_data
                .iter()
                .copied()
                .map(|p| p as f64 / 255.0)
                .collect::<Vec<_>>(),
        );

        row![
            button("Previous").on_press(Message::Previous),
            column![
                iced::widget::image::viewer(iced::advanced::image::Handle::from_rgba(
                    WIDTH, HEIGHT, image_data,
                ))
                .width((WIDTH * 10) as u16)
                .height((HEIGHT * 10) as u16),
                text(format!("Label: {}", label)),
                text(format!("Prediction: {}", prediction)).color(if &label == &prediction {
                    iced::Color::from_rgb(1.0, 1.0, 1.0)
                } else {
                    iced::Color::from_rgb(1.0, 0.0, 0.0)
                },),
            ],
            button("Next").on_press(Message::Next),
        ]
        .spacing(10)
        .into()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    iced::application("Display Model Predictions", Viewer::update, Viewer::view)
        .default_font(Font::MONOSPACE)
        .run()?;

    Ok(())
}
