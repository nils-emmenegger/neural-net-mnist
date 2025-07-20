use super::Op;
use std::collections::{HashMap, hash_map::Entry};

use super::Value;

impl Value {
    fn children(&self) -> impl Iterator<Item = Value> {
        struct ValueChildrenIterator {
            children: [Option<Value>; 2],
            index: usize,
        }

        impl ValueChildrenIterator {
            fn new(op: Option<Op>) -> Self {
                let mut children = [None, None];

                match op {
                    Some(Op::Add(x, y)) => {
                        children[0] = Some(x);
                        children[1] = Some(y);
                    }
                    Some(Op::Mul(x, y)) => {
                        children[0] = Some(x);
                        children[1] = Some(y);
                    }
                    Some(Op::Pow { base, exp: _ }) => {
                        children[0] = Some(base);
                    }
                    Some(Op::Tanh(x)) => {
                        children[0] = Some(x);
                    }
                    None => {}
                }

                Self { children, index: 0 }
            }
        }

        impl Iterator for ValueChildrenIterator {
            type Item = Value;

            fn next(&mut self) -> Option<Self::Item> {
                let child = self.children.get_mut(self.index).and_then(|x| x.take());
                self.index += 1;
                child
            }
        }

        ValueChildrenIterator::new(self.prev())
    }

    fn topological_sort(&self) -> Vec<Value> {
        #[allow(clippy::mutable_key_type)]
        let mut in_degree: HashMap<Value, usize> = HashMap::new();
        let mut stack = Vec::new();

        // First pass to initialize in_degree
        in_degree.insert(self.clone(), 0);
        stack.push(self.clone());
        while let Some(val) = stack.pop() {
            for child in val.children() {
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
            for child in val.children() {
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
            val.set_grad(0.0);
        }

        // Set root gradient to 1.0
        self.set_grad(1.0);

        // Backpropagate
        for val in topological_sorting.into_iter() {
            match val.prev() {
                Some(Op::Add(mut x, mut y)) => {
                    x.set_grad(x.grad() + val.grad());
                    y.set_grad(y.grad() + val.grad());
                }
                Some(Op::Mul(mut x, mut y)) => {
                    x.set_grad(x.grad() + val.grad() * y.data());
                    y.set_grad(y.grad() + val.grad() * x.data());
                }
                Some(Op::Pow { mut base, exp }) => {
                    base.set_grad(base.grad() + val.grad() * exp * base.data().powf(exp - 1.0));
                }
                Some(Op::Tanh(mut x)) => {
                    x.set_grad(x.grad() + val.grad() * (1.0 - x.data().tanh().powi(2)));
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
