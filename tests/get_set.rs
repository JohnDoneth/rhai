#[macro_use]
extern crate rhai;

use rhai::{Engine, Type, RegisterTypeFn};

#[test]
fn test_get_set() {
    #[derive(Clone)]
    struct TestStruct {
        x: i64,
    }

    impl TestStruct {
        fn new() -> TestStruct {
            TestStruct { x: 1 }
        }
    }

    let mut engine = Engine::new();

    register_type!(engine, TestStruct,
        fields: x;
        functions: new
    );

    let result = engine.eval::<i64>("let a = TestStruct::new(); a.x = 500; a.x");
    assert_eq!(result.unwrap(), 500);
}

#[test]
fn test_big_get_set() {
    #[derive(Clone)]
    struct TestChild {
        x: i64,
    }

    impl TestChild {
        fn new() -> TestChild {
            TestChild { x: 1 }
        }
    }

    #[derive(Clone)]
    struct TestParent {
        child: TestChild,
    }

    impl TestParent {
        fn new() -> TestParent {
            TestParent { child: TestChild::new() }
        }
    }

    let mut engine = Engine::new();

    register_type!(engine, TestChild,
        fields: x;
        functions: new
    );

    register_type!(engine, TestParent,
        fields: child;
        functions: new
    );

    assert_eq!(engine.eval::<i64>("let a = TestParent::new(); a.child.x = 500; a.child.x"), Ok(500));
}
