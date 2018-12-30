extern crate rhai;

use rhai::Engine;

#[test]
fn test_loop() {
	let mut engine = Engine::new();

	assert!(
		engine.eval::<bool>("
			let x = 0;
			let i = 0;

			loop {
				if i < 10 {
					x = x + i;
					i = i + 1;
				}
				else {
					break;
				}
			}

			x == 45
		").unwrap()
	);

	assert_eq!(
		engine.eval::<i64>("
			let i = 0;

			let r = loop {
				if i < 10 {
					i = i + 1;
				}
				else {
					break 5;
				}
			}

			r
		").unwrap(), 5
	)
}
