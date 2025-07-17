use std::{cell::RefCell, rc::Rc};

pub struct Value(Rc<RefCell<InnerValue>>);

struct InnerValue {
    data: f64,
    grad: f64,
    prev: Option<Op>,
}

#[derive(Clone)]
pub(super) enum Op {
    Add(Value, Value),
    Neg(Value),
    Mul(Value, Value),
    Pow { base: Value, exp: f64 },
    Tanh(Value),
}

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(InnerValue {
            data,
            grad: 0.0,
            prev: None,
        })))
    }

    pub(super) fn with_op(data: f64, op: Op) -> Value {
        Value(Rc::new(RefCell::new(InnerValue {
            data,
            grad: 0.0,
            prev: Some(op),
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn add_data(&mut self, val: f64) {
        self.0.borrow_mut().data += val;
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub(super) fn reset_grad(&mut self) {
        self.0.borrow_mut().grad = 0.0;
    }

    pub(super) fn add_grad(&mut self, val: f64) {
        self.0.borrow_mut().grad += val;
    }

    pub(super) fn prev(&self) -> Option<Op> {
        self.0.borrow().prev.clone()
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(Rc::clone(&self.0))
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(Rc::as_ptr(&self.0) as usize);
    }
}
