#[macro_use]
extern crate rhai;
use rhai::{Engine, Type, RegisterTypeFn};

#[derive(Clone, Debug)]
struct TestStruct {
    x: i64,
}

impl TestStruct {
    fn update(&mut self) {
        self.x += 1000;
    }

    fn new() -> TestStruct {
        TestStruct { x: 1 }
    }
}

fn main() {
    let mut engine = Engine::new();

    register_type!(engine, TestStruct,
        functions: new, update
    );

    println!("{:?}", engine.eval::<TestStruct>("let x = TestStruct::new(); x.update(); x"));
    println!("{:?}", engine.eval::<TestStruct>("let x = [TestStruct::new()]; x[0].update(); x[0]"));
}
