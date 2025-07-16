mod backprop;
mod base;
mod ops;

use base::Op;
pub use base::Value;

// general process I want to support is:
// create model
// loop:
//   calculate loss as a new Value that calculates the loss function across the last layer of the neural net
//   backpropagate from the loss Value
//   modify neural net accordingly (check associated grad and modify data)
