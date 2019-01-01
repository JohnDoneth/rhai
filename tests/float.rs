#[macro_use]
extern crate rhai;

use rhai::{Engine, Type, RegisterTypeFn};

#[test]
fn test_float() {
    let mut engine = Engine::new();

    let result = engine.eval::<bool>("let x = 0.0; let y = 1.0; x < y");
    assert!(result.unwrap());

    let result = engine.eval::<bool>("let x = 0.0; let y = 1.0; x > y");
    assert!(!result.unwrap());

    let result = engine.eval::<f64>("let x = 9.9999; x");
    assert_eq!(result.unwrap(), 9.9999);
}

#[test]
fn struct_with_float() {
    #[derive(Clone)]
    struct TestStruct {
        x: f64,
    }

    impl TestStruct {
        fn update(&mut self) {
            self.x += 5.789_f64;
        }

        fn new() -> TestStruct {
            TestStruct { x: 1.0 }
        }
    }

    let mut engine = Engine::new();
    register_type!(engine, TestStruct,
        fields: x;
        functions: new, update
    );

    let result = engine.eval::<f64>("let ts = TestStruct::new(); ts.update(); ts.x");
    assert_eq!(result.unwrap(), 6.789);

    let result = engine.eval::<f64>("let ts = TestStruct::new(); ts.x = 10.1001; ts.x");
    assert_eq!(result.unwrap(), 10.1001);
}
