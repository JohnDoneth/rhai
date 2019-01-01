use std::any::TypeId;
use std::sync::Arc;

use any::Any;
use engine::{Engine, EvalAltResult, FnSpec};

pub trait RegisterFn<FN, ARGS, RET> {
    fn register_fn(&mut self, name: &str, f: FN);
}

pub trait RegisterTypeFn<FN, ARGS, RET> {
    fn add_function(mut self, name: &str, f: FN) -> Self;
}