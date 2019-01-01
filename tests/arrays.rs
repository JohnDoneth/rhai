#[macro_use]
extern crate rhai;

use rhai::{Engine, Type, RegisterTypeFn};

#[test]
fn test_arrays() {
    let mut engine = Engine::new();

    let result = engine.eval::<i64>("let x = [1, 2, 3]; x[1]");
    assert_eq!(result.unwrap(), 2);

    let result = engine.eval::<i64>("let y = [1, 2, 3]; y[1] = 5; y[1]");
    assert_eq!(result.unwrap(), 5);
}

#[test]
fn test_array_with_structs() {
    #[derive(Clone)]
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

    let mut engine = Engine::new();

    register_type!(engine, TestStruct,
        fields: x;
        functions: new, update
    );

    let result = engine.eval::<i64>("let a = [TestStruct::new()]; a[0].x");
    assert_eq!(result.unwrap(), 1);

    let result = engine.eval::<i64>("let a = [TestStruct::new()]; a[0].x = 100; a[0].update(); \
                                            a[0].x");
    assert_eq!(result.unwrap(), 1100);
}
