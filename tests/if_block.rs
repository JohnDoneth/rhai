extern crate rhai;

use rhai::Engine;

#[test]
fn test_if() {
    let mut engine = Engine::new();

    if let Ok(result) = engine.eval::<i64>("if true { 55 }") {
        assert_eq!(result, 55);
    } else {
        assert!(false);
    }

    if let Ok(result) = engine.eval::<i64>("if false { 55 } else { 44 }") {
        assert_eq!(result, 44);
    } else {
        assert!(false);
    }

    if let Ok(result) = engine.eval::<i64>("if true { 55 } else { 44 }") {
        assert_eq!(result, 55);
    } else {
        assert!(false);
    }

    if let Ok(result) = engine.eval::<i64>("let x = if true { 55 } else { 44 }; x") {
        assert_eq!(result, 55);
    } else {
        assert!(false);
    }
}
