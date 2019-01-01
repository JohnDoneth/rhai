#[macro_use]
extern crate rhai;

use rhai::{Engine, Type, RegisterTypeFn};

#[test]
fn test_method_call() {
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

    let result = engine.eval::<TestStruct>("let x = TestStruct::new(); x.update(); x");
    assert_eq!(result.unwrap().x, 1001);

}
