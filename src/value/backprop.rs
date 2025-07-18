use super::Op;
use std::collections::{HashMap, hash_map::Entry};

use super::Value;

impl Value {
    fn get_children(&self) -> Vec<Value> {
        match self.prev() {
            Some(Op::Add(x, y)) => vec![x, y],
            Some(Op::Neg(x)) => vec![x],
            Some(Op::Mul(x, y)) => vec![x, y],
            Some(Op::Pow { base, exp: _ }) => vec![base],
            Some(Op::Tanh(x)) => vec![x],
            None => Vec::new(),
        }
    }

    fn topological_sort(&self) -> Vec<Value> {
        #[allow(clippy::mutable_key_type)]
        let mut in_degree: HashMap<Value, usize> = HashMap::new();
        let mut stack = Vec::new();

        // First pass to initialize in_degree
        in_degree.insert(self.clone(), 0);
        stack.push(self.clone());
        while let Some(val) = stack.pop() {
            for child in val.get_children().into_iter() {
                match in_degree.entry(child.clone()) {
                    Entry::Occupied(mut occupied_entry) => *(occupied_entry.get_mut()) += 1,
                    Entry::Vacant(vacant_entry) => {
                        vacant_entry.insert(1);
                        stack.push(child);
                    }
                }
            }
        }

        // Should be 0, otherwise there is a loop
        assert_eq!(in_degree.get(self), Some(&0));

        // Second pass to determine sorting
        let mut sorting = Vec::new();
        stack.push(self.clone());
        while let Some(val) = stack.pop() {
            for child in val.get_children().into_iter() {
                let ind = in_degree.get_mut(&child).unwrap();
                *ind -= 1;
                if *ind == 0 {
                    stack.push(child);
                }
            }
            sorting.push(val);
        }

        // Assert that everything has been processed, otherwise there is a loop
        assert_eq!(sorting.len(), in_degree.len());

        sorting
    }

    pub fn backward(&mut self) {
        let mut topological_sorting = self.topological_sort();

        // Set all gradients to 0
        for val in topological_sorting.iter_mut() {
            val.reset_grad();
        }

        // Set root gradient to 1.0
        self.add_grad(1.0);

        // Backpropagate
        for val in topological_sorting.into_iter() {
            match val.prev() {
                Some(Op::Add(mut x, mut y)) => {
                    x.add_grad(val.grad());
                    y.add_grad(val.grad());
                }
                Some(Op::Neg(mut x)) => {
                    x.add_grad(-val.grad());
                }
                Some(Op::Mul(mut x, mut y)) => {
                    x.add_grad(val.grad() * y.data());
                    y.add_grad(val.grad() * x.data());
                }
                Some(Op::Pow { mut base, exp }) => {
                    base.add_grad(val.grad() * exp * base.data().powf(exp - 1.0));
                }
                Some(Op::Tanh(mut x)) => {
                    x.add_grad(val.grad() * (1.0 - x.data().tanh().powi(2)));
                }
                None => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        let a = Value::new(3.0);
        let b = Value::new(7.0);
        let c = Value::new(11.0);
        let mut d = &a * &(&b + &c);
        d.backward();
        assert_eq!(a.grad(), 18.0);
        assert_eq!(b.grad(), 3.0);
        assert_eq!(c.grad(), 3.0);
        assert_eq!(d.grad(), 1.0);
    }

    #[test]
    fn test2() {
        let a = Value::new(3.0);
        let b = Value::new(7.0);
        let c = Value::new(11.0);
        let mut d = (&(&a * &b) + &(-&c)).powf(5.0);
        d.backward();
        assert_eq!(a.grad(), 350000.0);
        assert_eq!(b.grad(), 150000.0);
        assert_eq!(c.grad(), -50000.0);
        assert_eq!(d.grad(), 1.0);
    }

    #[test]
    fn test3() {
        let a = Value::new(3.0);
        let mut b = &(&a * &a) + &a;
        b.backward();
        assert_eq!(a.grad(), 7.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn test4() {
        let a = Value::new(0.5);
        let b = Value::new(0.25);
        let mut c = &b * &(&(&a * &b) + &a).tanh();
        c.backward();
        assert_eq!(a.grad(), 0.2163809837406213);
        assert_eq!(b.grad(), 0.6411521158456308);
        assert_eq!(c.grad(), 1.0);
    }
}
