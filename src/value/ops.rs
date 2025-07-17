use super::{Op, Value};

impl std::ops::Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        Value::with_op(self.data() + rhs.data(), Op::Add(self.clone(), rhs.clone()))
    }
}

impl std::ops::Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value::with_op(-self.data(), Op::Neg(self.clone()))
    }
}

impl std::ops::Mul for &Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value::with_op(self.data() * rhs.data(), Op::Mul(self.clone(), rhs.clone()))
    }
}

impl Value {
    pub fn powf(&self, exp: f64) -> Value {
        Value::with_op(
            self.data().powf(exp),
            Op::Pow {
                base: self.clone(),
                exp,
            },
        )
    }

    pub fn tanh(&self) -> Value {
        Value::with_op(self.data().tanh(), Op::Tanh(self.clone()))
    }
}
