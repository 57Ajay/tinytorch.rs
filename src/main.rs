use std::println;

pub mod tensor;

fn main() {
    let t1 = tensor::Tensor::new(vec![2.0, 4.9, 1.2], (1, 3));
    let t2 = tensor::Tensor::new(vec![2.1, 4.1, 2.2], (3, 1));

    let t3 = t1.mul(&t2);

    println!("t3: {:?}", t3.data);
}
