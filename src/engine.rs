use std::any::TypeId;
use std::iter::once;
use std::borrow::Borrow;
use std::cmp::{PartialEq, PartialOrd};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::{Add, BitAnd, BitOr, BitXor, Deref, Div, Mul, Neg, Rem, Shl, Shr, Sub};
use std::sync::Arc;

use any::{Any, AnyExt};
use call::FunArgs;
use fn_register::{RegisterFn, RegisterTypeFn};
use parser::{lex, parse, Expr, FnDef, Stmt};

#[macro_export]
macro_rules! register_type {
    ($engine:expr, $name:ident) => {
        $engine.register_type(
            Type::new::<$name>(stringify!($name))
        );
    };
    ($engine:expr, $name:ident, fields: $($field:ident),+ ; functions: $($function:ident),+) => {
        $engine.register_type(
            Type::new::<$name>(stringify!($name))
            $(
                .add_field(stringify!($field),
                    Box::new(|s: &$name|s.$field),
                    Some(Box::new(|s: &mut $name, v|{s.$field = v}))
                )
            )+
            $(
                .add_function(stringify!($function), $name::$function)
            )+
        );
    };
    ($engine:expr, $name:ident, fields: $($field:ident),+) => {
        $engine.register_type(
            Type::new::<$name>(stringify!($name))
            $(
                .add_field(stringify!($field),
                    Box::new(|s: &$name|s.$field),
                    Some(Box::new(|s: &mut $name, v|{s.$field = v}))
                )
            )+
        );
    };
    ($engine:expr, $name:ident, functions: $($function:ident),+) => {
        $engine.register_type(
            Type::new::<$name>(stringify!($name))
            $(
                .add_function(stringify!($function), $name::$function)
            )+
        );
    };
}

#[derive(Debug)]
pub enum EvalAltResult {
    ErrorFunctionNotFound(String),
    ErrorFunctionArgMismatch,
    ErrorIndexMismatch,
    ErrorIfGuardMismatch,
    ErrorVariableNotFound(String),
    ErrorAssignmentToUnknownLHS,
    ErrorUnregisteredType(String),
    ErrorMismatchOutputType(String),
    ErrorCantOpenScriptFile,
    InternalErrorMalformedDotExpression,
    InternalErrorMalformedDoubleColonCall,
    LoopBreak(Box<Any>),
    Return(Box<Any>),
}

impl EvalAltResult {
    fn as_str(&self) -> Option<&str> {
        match *self {
            EvalAltResult::ErrorVariableNotFound(ref s) => Some(s.as_str()),
            EvalAltResult::ErrorFunctionNotFound(ref s) => Some(s.as_str()),
            EvalAltResult::ErrorMismatchOutputType(ref s) => Some(s.as_str()),
            EvalAltResult::ErrorUnregisteredType(ref s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl PartialEq for EvalAltResult {
    fn eq(&self, other: &Self) -> bool {
        use EvalAltResult::*;

        match (self, other) {
            (&ErrorFunctionNotFound(ref a), &ErrorFunctionNotFound(ref b)) => a == b,
            (&ErrorFunctionArgMismatch, &ErrorFunctionArgMismatch) => true,
            (&ErrorIndexMismatch, &ErrorIndexMismatch) => true,
            (&ErrorIfGuardMismatch, &ErrorIfGuardMismatch) => true,
            (&ErrorVariableNotFound(ref a), &ErrorVariableNotFound(ref b)) => a == b,
            (&ErrorAssignmentToUnknownLHS, &ErrorAssignmentToUnknownLHS) => true,
            (&ErrorMismatchOutputType(ref a), &ErrorMismatchOutputType(ref b)) => a == b,
            (&ErrorUnregisteredType(ref a), &ErrorUnregisteredType(ref b)) => a == b,
            (&ErrorCantOpenScriptFile, &ErrorCantOpenScriptFile) => true,
            (&InternalErrorMalformedDotExpression, &InternalErrorMalformedDotExpression) => true,
            (&InternalErrorMalformedDoubleColonCall, &InternalErrorMalformedDoubleColonCall) => true,
            _ => false,
        }
    }
}

impl Error for EvalAltResult {
    fn description(&self) -> &str {
        match *self {
            EvalAltResult::ErrorFunctionNotFound(_) => "Function not found",
            EvalAltResult::ErrorFunctionArgMismatch => "Function argument types do not match",
            EvalAltResult::ErrorIndexMismatch => "Index does not match array",
            EvalAltResult::ErrorIfGuardMismatch => "If guards expect boolean expression",
            EvalAltResult::ErrorVariableNotFound(_) => "Variable not found",
            EvalAltResult::ErrorAssignmentToUnknownLHS => {
                "Assignment to an unsupported left-hand side"
            }
            EvalAltResult::ErrorMismatchOutputType(_) => "Cast of output failed",
            EvalAltResult::ErrorUnregisteredType(_) => "Unknown type",
            EvalAltResult::ErrorCantOpenScriptFile => "Cannot open script file",
            EvalAltResult::InternalErrorMalformedDotExpression => {
                "[Internal error] Unexpected expression in dot expression"
            }
            EvalAltResult::InternalErrorMalformedDoubleColonCall => {
                "[Internal error] Unexpected expression in double colon expression"
            }
            EvalAltResult::LoopBreak(_) => "Loop broken before completion (not an error)",
            EvalAltResult::Return(_) => "Function returned value (not an error)",
        }
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

impl fmt::Display for EvalAltResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(s) = self.as_str() {
            write!(f, "{}: {}", self.description(), s)
        } else {
            write!(f, "{}", self.description())
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct FnSpec {
    ident: String,
    args: Option<Vec<TypeId>>,
}

impl FnSpec {
    pub fn new(ident: String, args: Option<Vec<TypeId>>) -> FnSpec {
        FnSpec { ident, args }
    }
}

#[derive(Clone)]
pub struct TypeSpec {
    id: TypeId,
    name: String,
    fields: HashMap<String, (Arc<Box<FnGetter>>, Option<Arc<Box<FnSetter>>>)>,
    functions: HashMap<FnSpec, Arc<Box<FnAny>>>,
}

impl TypeSpec {
    pub fn new(id: TypeId, name: String, fields: HashMap<String, (Arc<Box<FnGetter>>, Option<Arc<Box<FnSetter>>>)>,
               functions: HashMap<FnSpec, Arc<Box<FnAny>>>) -> TypeSpec {

        TypeSpec { id, name, fields, functions }
    }
}

/// Rhai's engine type. This is what you use to run Rhai scripts
///
/// ```rust
/// extern crate rhai;
/// use rhai::Engine;
///
/// fn main() {
///     let mut engine = Engine::new();
///
///     if let Ok(result) = engine.eval::<i64>("40 + 2") {
///         println!("Answer: {}", result);  // prints 42
///     }
/// }
/// ```
#[derive(Clone)]
pub struct Engine {
    /// A hashmap containing all functions known to the engine
    pub fns: HashMap<FnSpec, Arc<FnIntExt>>,
    pub type_names: HashMap<TypeId, String>,
    pub types: HashMap<String, Arc<TypeSpec>>,
}

pub enum FnIntExt {
    Ext(Box<FnAny>),
    Int(FnDef),
}

pub type FnGetter = Fn(&Any) -> Result<Box<Any>, EvalAltResult>;
pub type FnSetter = Fn(&mut Any, Box<Any>) -> Result<(), EvalAltResult>;
pub type FnAny = Fn(Vec<&mut Any>) -> Result<Box<Any>, EvalAltResult>;

/// A type containing information about current scope.
/// Useful for keeping state between `Engine` runs
///
/// ```rust
/// use rhai::{Engine, Scope};
///
/// let mut engine = Engine::new();
/// let mut my_scope = Scope::new();
///
/// assert!(engine.eval_with_scope::<()>(&mut my_scope, "let x = 5;").is_ok());
/// assert_eq!(engine.eval_with_scope::<i64>(&mut my_scope, "x + 1").unwrap(), 6);
/// ```
///
/// Between runs, `Engine` only remembers functions when not using own `Scope`.
pub type Scope = Vec<(String, Box<Any>)>;

pub struct Type {
    id: TypeId,
    name: String,
    fields: HashMap<String, (Arc<Box<FnGetter>>, Option<Arc<Box<FnSetter>>>)>,
    functions: HashMap<FnSpec, Arc<Box<FnAny>>>
}

impl Type {
    pub fn new<T: Any>(name: &str) -> Type {
        Type {
            id: TypeId::of::<T>(),
            name: name.to_string(),
            fields: HashMap::new(),
            functions: HashMap::new()
        }
    }

    pub fn add_field<T, U, G, S>(mut self, name: &str, getter: Box<G>, setter: Option<Box<S>>) -> Type
        where
            T: Clone + Any,
            U: Clone + Any,
            G: 'static + Fn(&T) -> U,
            S: 'static + Fn(&mut T, U) -> ()
    {
        let getter = Arc::new(Box::new(move |this_ptr: &Any| {
            if let Some(this_ptr) = this_ptr.downcast_ref::<T>() {
                Ok(Box::new(getter(this_ptr)) as Box<Any>)
            } else {
                Err(EvalAltResult::ErrorFunctionArgMismatch)
            }
        }) as Box<FnGetter>);
        let setter = if let Some(setter) = setter {
            Some(Arc::new(Box::new(move |this_ptr: &mut Any, new_val: Box<Any>| {
                let mut this_ptr = this_ptr.downcast_mut::<T>();
                if let Ok(new_val) = new_val.clone().downcast::<U>() {
                    setter(this_ptr.unwrap(), *new_val);
                    Ok(())
                } else {
                    Err(EvalAltResult::ErrorFunctionArgMismatch)
                }
            }) as Box<FnSetter>))
        } else {
            None
        };
        self.fields.insert(name.to_string(), (getter, setter));
        self
    }
}

impl Engine {
    pub fn call_fn<'a, I, A, T>(&self, ident: I, args: A) -> Result<T, EvalAltResult>
    where
        I: Into<String>,
        A: FunArgs<'a>,
        T: Any + Clone,
    {
        let mut scope = Scope::new();
        self.call_fn_raw(ident.into(), args.into_vec(), &mut scope)
            .and_then(|b| {
                b.downcast()
                    .map(|b| *b)
                    .map_err(|a| EvalAltResult::ErrorMismatchOutputType(self.format_type(a.type_id())))
            })
    }

    /// Universal method for calling functions, that are either
    /// registered with the `Engine` or written in Rhai
    pub fn call_fn_raw(
        &self,
        ident: String,
        args: Vec<&mut Any>,
        scope: &mut Scope,
    ) -> Result<Box<Any>, EvalAltResult> {
        debug_println!(
            "Trying to call function {:?} with args {:?}",
            ident,
            args.iter().map(|x| (&**x).type_id()).collect::<Vec<_>>()
        );

        let prev_len = scope.len();
        let res = match *self.find_fn(&ident, &args)? {
            FnIntExt::Ext(ref f) => f(args),
            FnIntExt::Int(ref f) => {
                scope.extend(
                    f.params
                        .iter()
                        .cloned()
                        .zip(args.iter().map(|x| (&**x).box_clone())),
                );

                match self.eval_stmt(scope, &*f.body) {
                    Err(EvalAltResult::Return(x)) => Ok(x),
                    other => other,
                }
            }
        };
        while scope.len() > prev_len {
            scope.pop();
        }
        res
    }

    pub fn register_fn_raw(&mut self, ident: String, args: Option<Vec<TypeId>>, f: Box<FnAny>) {
        debug_println!("Register; {:?} with args {:?}", ident, args);

        let spec = FnSpec::new(ident, args);
        self.fns.insert(spec, Arc::new(FnIntExt::Ext(f)));
    }

    /// Register a type
    pub fn register_type(&mut self, ty: Type) {
        let Type {id, name, fields, functions} = ty;
        self.type_names.insert(id.clone(), name.clone());
        self.types.insert(name.clone(), Arc::new(TypeSpec {
            id, name, fields, functions
        }));
    }

    fn find_type(&self, tid: &TypeId) -> Result<Arc<TypeSpec>, EvalAltResult> {
        if let Some(type_name) = self.type_names.get(tid) {
            if let Some(ty) = self.types.get(type_name).cloned() {
                Ok(ty)
            } else {
                Err(EvalAltResult::ErrorUnregisteredType(format!("{}", type_name)))
            }
        } else {
            Err(EvalAltResult::ErrorUnregisteredType(format!("<unknown> {:?}", tid)))
        }
    }

    fn find_getter(&self, tid: &TypeId, name: &String) -> Result<Arc<Box<FnGetter>>, EvalAltResult> {
        let ty = self.find_type(tid)?;

        if let Some((g, s)) = ty.fields.get(name) {
            Ok(g.clone())
        } else {
            Err(EvalAltResult::ErrorVariableNotFound(format!("{}.{}", ty.name, name)))
        }
    }

    fn find_setter(&self, tid: &TypeId, name: &String) -> Result<Arc<Box<FnSetter>>, EvalAltResult> {
        let ty = self.find_type(tid)?;

        if let Some((g, s)) = ty.fields.get(name) {
            if let Some(s) = s {
                return Ok(s.clone());
            }
        }
        Err(EvalAltResult::ErrorVariableNotFound(format!("{}.{}", ty.name, name)))
    }

    fn find_fn(&self, name: &String, args: &Vec<&mut Any>) -> Result<Arc<FnIntExt>, EvalAltResult> {
        let spec = FnSpec::new(
            name.clone(),
            Some(args.iter().map(|a| Any::type_id(&**a)).collect())
        );
        if let Some(f) = self.fns.get(&spec) {
            Ok(f.clone())
        } else {
            let typenames = args
                .iter()
                .map(|x| self.format_type((&**x).type_id()))
                .collect::<Vec<_>>();
            Err(EvalAltResult::ErrorFunctionNotFound(format!("{}({})", name, typenames.join(","))))
        }
    }

    fn find_type_fn(&self, tid: &TypeId, name: &String, args: &Vec<&mut Any>) -> Result<Arc<Box<FnAny>>, EvalAltResult> {
        let ty = self.find_type(tid)?;

        let spec = FnSpec::new(
            name.clone(),
            Some(args.iter().map(|a| Any::type_id(&**a)).collect())
        );
        if let Some(f) = ty.functions.get(&spec) {
            Ok(f.clone())
        } else {
            let typenames = args
                .iter()
                .map(|x| self.format_type((&**x).type_id()))
                .skip(1)
                .collect::<Vec<_>>();
            Err(EvalAltResult::ErrorFunctionNotFound(format!("{}.{}({})", ty.name, name, typenames.join(","))))
        }
    }

    fn get_dot_val_helper(
        &self,
        scope: &mut Scope,
        this_ptr: &mut Any,
        dot_rhs: &Expr,
    ) -> Result<Box<Any>, EvalAltResult> {
        match *dot_rhs {
            Expr::FnCall(ref fn_name, ref args) => {
                let type_id = Any::type_id(this_ptr);

                let mut args: Vec<Box<Any>> = args
                    .iter()
                    .map(|arg| self.eval_expr(scope, arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let args: Vec<&mut Any> = once(this_ptr)
                    .chain(args.iter_mut().map(|b| b.as_mut()))
                    .collect();

                match self.find_type_fn(&type_id, fn_name, &args) {
                    Ok(f) => f(args),
                    Err(e) => match e {
                        EvalAltResult::ErrorUnregisteredType(s) => {
                            self.call_fn_raw(fn_name.to_owned(), args, scope)
                        },
                        other => Err(other)
                    }
                }
            }
            Expr::Identifier(ref id) => {
                let getter: Arc<Box<FnGetter>> = self.find_getter(&Any::type_id(this_ptr), id)?;
                getter(this_ptr)
            }
            Expr::Index(ref id, ref idx_raw) => {
                let getter: Arc<Box<FnGetter>> = self.find_getter(&Any::type_id(this_ptr), id)?;
                let mut val = getter(this_ptr)?;
                let idx = self.eval_expr(scope, idx_raw)?;
                ((*val).downcast_mut() as Option<&mut Vec<Box<Any>>>)
                    .and_then(|arr| idx.downcast_ref::<i64>().map(|idx| (arr, *idx as usize)))
                    .map(|(arr, idx)| arr[idx].clone())
                    .ok_or(EvalAltResult::ErrorIndexMismatch)
            }
            Expr::Dot(ref inner_lhs, ref inner_rhs) => match **inner_lhs {
                Expr::Identifier(ref id) => {
                    let type_id = Any::type_id(this_ptr);

                    let getter: Arc<Box<FnGetter>> = self.find_getter(&type_id, id)?;
                    let mut val = getter(this_ptr)?;
                    self.get_dot_val_helper(scope, val.as_mut(), inner_rhs)
                }
                _ => Err(EvalAltResult::InternalErrorMalformedDotExpression),
            },
            _ => Err(EvalAltResult::InternalErrorMalformedDotExpression),
        }
    }

    fn search_scope<'a, F, T>(
        scope: &'a mut Scope,
        id: &str,
        map: F,
    ) -> Result<(usize, T), EvalAltResult>
    where
        F: FnOnce(&'a mut Any) -> Result<T, EvalAltResult>,
    {
        scope
            .iter_mut()
            .enumerate()
            .rev()
            .find(|&(_, &mut (ref name, _))| *id == *name)
            .ok_or_else(|| EvalAltResult::ErrorVariableNotFound(id.to_owned()))
            .and_then(move |(idx, &mut (_, ref mut val))| map(val.as_mut()).map(|val| (idx, val)))
    }

    fn array_value(
        &self,
        scope: &mut Scope,
        id: &str,
        idx: &Expr,
    ) -> Result<(usize, usize, Box<Any>), EvalAltResult> {
        let idx_boxed = self
            .eval_expr(scope, idx)?
            .downcast::<i64>()
            .map_err(|_| EvalAltResult::ErrorIndexMismatch)?;
        let idx = *idx_boxed as usize;
        let (idx_sc, val) = Self::search_scope(scope, id, |val| {
            ((*val).downcast_mut() as Option<&mut Vec<Box<Any>>>)
                .map(|arr| arr[idx].clone())
                .ok_or(EvalAltResult::ErrorIndexMismatch)
        })?;

        Ok((idx_sc, idx, val))
    }

    fn call_fn_double_colon(
        &self,
        scope: &mut Scope,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<Box<Any>, EvalAltResult> {
        match *lhs {
            Expr::Identifier(ref id) => {
                if let Some(ty) = self.types.get(id) {
                    match *rhs {
                        Expr::FnCall(ref fn_name, ref args) => {
                            let mut args: Vec<Box<Any>> = args
                                .iter()
                                .map(|arg| self.eval_expr(scope, arg))
                                .collect::<Result<Vec<_>, _>>()?;

                            let args = args.iter_mut().map(|b| b.as_mut()).collect::<Vec<_>>();

                            let f = self.find_type_fn(&ty.id, fn_name, &args)?;
                            f(args)
                        },
                        _ => Err(EvalAltResult::InternalErrorMalformedDoubleColonCall)
                    }
                } else {
                    Err(EvalAltResult::ErrorUnregisteredType(id.clone()))
                }
            }
            _ => Err(EvalAltResult::InternalErrorMalformedDoubleColonCall),
        }
    }

    fn get_dot_val(
        &self,
        scope: &mut Scope,
        dot_lhs: &Expr,
        dot_rhs: &Expr,
    ) -> Result<Box<Any>, EvalAltResult> {
        match *dot_lhs {
            Expr::Identifier(ref id) => {
                let (sc_idx, mut target) = Self::search_scope(scope, id, |x| Ok(x.box_clone()))?;
                let value = self.get_dot_val_helper(scope, target.as_mut(), dot_rhs);

                // In case the expression mutated `target`, we need to reassign it because
                // of the above `clone`.
                scope[sc_idx].1 = target;

                value
            }
            Expr::Index(ref id, ref idx_raw) => {
                let (sc_idx, idx, mut target) = self.array_value(scope, id, idx_raw)?;
                let value = self.get_dot_val_helper(scope, target.as_mut(), dot_rhs);

                // In case the expression mutated `target`, we need to reassign it because
                // of the above `clone`.
                scope[sc_idx].1.downcast_mut::<Vec<Box<Any>>>().unwrap()[idx] = target;

                value
            }
            _ => Err(EvalAltResult::InternalErrorMalformedDotExpression),
        }
    }

    fn set_dot_val_helper(
        &self,
        this_ptr: &mut Any,
        dot_rhs: &Expr,
        mut source_val: Box<Any>,
        scope: &mut Scope,
    ) -> Result<Box<Any>, EvalAltResult> {
        match *dot_rhs {
            Expr::Identifier(ref id) => {
                let type_id = Any::type_id(this_ptr);
                let setter = self.find_setter(&type_id, id)?;
                setter(this_ptr, source_val)?;
                Ok(Box::new(()))
            }
            Expr::Dot(ref inner_lhs, ref inner_rhs) => match **inner_lhs {
                Expr::Identifier(ref id) => {
                    let type_id = Any::type_id(this_ptr);
                    let getter = self.find_getter(&type_id, id)?;
                    let mut val = getter(this_ptr)?;
                    self.set_dot_val_helper(val.as_mut(), inner_rhs, source_val, scope)?;
                    let setter = self.find_setter(&type_id, id)?;
                    setter(this_ptr, val)?;
                    Ok(Box::new(()))
                }
                _ => Err(EvalAltResult::InternalErrorMalformedDotExpression),
            },
            _ => Err(EvalAltResult::InternalErrorMalformedDotExpression),
        }
    }

    fn set_dot_val(
        &self,
        scope: &mut Scope,
        dot_lhs: &Expr,
        dot_rhs: &Expr,
        source_val: Box<Any>,
    ) -> Result<Box<Any>, EvalAltResult> {
        match *dot_lhs {
            Expr::Identifier(ref id) => {
                let (sc_idx, mut target) = Self::search_scope(scope, id, |x| Ok(x.box_clone()))?;
                let value = self.set_dot_val_helper(target.as_mut(), dot_rhs, source_val, scope);

                // In case the expression mutated `target`, we need to reassign it because
                // of the above `clone`.
                scope[sc_idx].1 = target;

                value
            }
            Expr::Index(ref id, ref idx_raw) => {
                let (sc_idx, idx, mut target) = self.array_value(scope, id, idx_raw)?;
                let value = self.set_dot_val_helper(target.as_mut(), dot_rhs, source_val, scope);

                // In case the expression mutated `target`, we need to reassign it because
                // of the above `clone`.
                scope[sc_idx].1.downcast_mut::<Vec<Box<Any>>>().unwrap()[idx] = target;

                value
            }
            _ => Err(EvalAltResult::InternalErrorMalformedDotExpression),
        }
    }

    fn eval_expr(&self, scope: &mut Scope, expr: &Expr) -> Result<Box<Any>, EvalAltResult> {
        match *expr {
            Expr::IntConst(i) => Ok(Box::new(i)),
            Expr::FloatConst(i) => Ok(Box::new(i)),
            Expr::StringConst(ref s) => Ok(Box::new(s.clone())),
            Expr::CharConst(ref c) => Ok(Box::new(*c)),
            Expr::Identifier(ref id) => {
                for &mut (ref name, ref mut val) in &mut scope.iter_mut().rev() {
                    if *id == *name {
                        return Ok(val.clone());
                    }
                }
                Err(EvalAltResult::ErrorVariableNotFound(id.clone()))
            },
            Expr::IfElse(ref guard, ref body, ref else_body) => {
                let guard_result = self.eval_expr(scope, guard)?;
                match guard_result.downcast::<bool>() {
                    Ok(g) => {
                        if *g {
                            self.eval_stmt(scope, body)
                        } else {
                            self.eval_stmt(scope, else_body)
                        }
                    }
                    Err(_) => Err(EvalAltResult::ErrorIfGuardMismatch),
                }
            },
            Expr::Loop(ref body) => loop {
                match self.eval_stmt(scope, body) {
                    Err(EvalAltResult::LoopBreak(x)) => return Ok(x),
                    Err(x) => return Err(x),
                    _ => (),
                }
            },
            Expr::Index(ref id, ref idx_raw) => {
                self.array_value(scope, id, idx_raw).map(|(_, _, x)| x)
            },
            Expr::Assignment(ref id, ref rhs) => {
                let rhs_val = self.eval_expr(scope, rhs)?;

                match **id {
                    Expr::Identifier(ref n) => {
                        for &mut (ref name, ref mut val) in &mut scope.iter_mut().rev() {
                            if *n == *name {
                                *val = rhs_val;

                                return Ok(Box::new(()));
                            }
                        }
                        Err(EvalAltResult::ErrorVariableNotFound(n.clone()))
                    }
                    Expr::Index(ref id, ref idx_raw) => {
                        let idx = self.eval_expr(scope, idx_raw)?;
                        for &mut (ref name, ref mut val) in &mut scope.iter_mut().rev() {
                            if *id == *name {
                                if let Some(i) = idx.downcast_ref::<i64>() {
                                    if let Some(arr_typed) =
                                        (*val).downcast_mut() as Option<&mut Vec<Box<Any>>>
                                    {
                                        arr_typed[*i as usize] = rhs_val;
                                        return Ok(Box::new(()));
                                    } else {
                                        return Err(EvalAltResult::ErrorIndexMismatch);
                                    }
                                } else {
                                    return Err(EvalAltResult::ErrorIndexMismatch);
                                }
                            }
                        }

                        Err(EvalAltResult::ErrorVariableNotFound(id.clone()))
                    }
                    Expr::Dot(ref dot_lhs, ref dot_rhs) => {
                        self.set_dot_val(scope, dot_lhs, dot_rhs, rhs_val)
                    }
                    _ => Err(EvalAltResult::ErrorAssignmentToUnknownLHS),
                }
            },
            Expr::Dot(ref lhs, ref rhs) => self.get_dot_val(scope, lhs, rhs),
            Expr::DoubleColon(ref lhs, ref rhs) => self.call_fn_double_colon(scope, lhs, rhs),
            Expr::Array(ref contents) => {
                let mut arr = Vec::new();

                for item in &(*contents) {
                    let arg = self.eval_expr(scope, item)?;
                    arr.push(arg);
                }

                Ok(Box::new(arr))
            },
            Expr::FnCall(ref fn_name, ref args) => self.call_fn_raw(
                fn_name.to_owned(),
                args.iter()
                    .map(|ex| self.eval_expr(scope, ex))
                    .collect::<Result<Vec<Box<Any>>, _>>()?
                    .iter_mut()
                    .map(|b| b.as_mut())
                    .collect(),
                scope,
            ),
            Expr::True => Ok(Box::new(true)),
            Expr::False => Ok(Box::new(false)),
            Expr::Unit => Ok(Box::new(())),
        }
    }

    fn eval_stmt(&self, scope: &mut Scope, stmt: &Stmt) -> Result<Box<Any>, EvalAltResult> {
        match *stmt {
            Stmt::Expr(ref e) => self.eval_expr(scope, e),
            Stmt::Block(ref b) => {
                let prev_len = scope.len();
                let mut last_result: Result<Box<Any>, EvalAltResult> = Ok(Box::new(()));

                for s in b.iter() {
                    last_result = self.eval_stmt(scope, s);
                    if let Err(x) = last_result {
                        last_result = Err(x);
                        break;
                    }
                }

                while scope.len() > prev_len {
                    scope.pop();
                }

                last_result
            }
            Stmt::If(ref guard, ref body) => {
                let guard_result = self.eval_expr(scope, guard)?;
                match guard_result.downcast::<bool>() {
                    Ok(g) => {
                        if *g {
                            self.eval_stmt(scope, body)
                        } else {
                            Ok(Box::new(()))
                        }
                    }
                    Err(_) => Err(EvalAltResult::ErrorIfGuardMismatch),
                }
            }
            Stmt::IfElse(ref guard, ref body, ref else_body) => {
                let guard_result = self.eval_expr(scope, guard)?;
                match guard_result.downcast::<bool>() {
                    Ok(g) => {
                        if *g {
                            self.eval_stmt(scope, body)
                        } else {
                            self.eval_stmt(scope, else_body)
                        }
                    }
                    Err(_) => Err(EvalAltResult::ErrorIfGuardMismatch),
                }
            }
            Stmt::While(ref guard, ref body) => loop {
                let guard_result = self.eval_expr(scope, guard)?;
                match guard_result.downcast::<bool>() {
                    Ok(g) => {
                        if *g {
                            match self.eval_stmt(scope, body) {
                                Err(EvalAltResult::LoopBreak(x)) => return Ok(x),
                                Err(x) => return Err(x),
                                _ => (),
                            }
                        } else {
                            return Ok(Box::new(()));
                        }
                    }
                    Err(_) => return Err(EvalAltResult::ErrorIfGuardMismatch),
                }
            },
            Stmt::Loop(ref body) => loop {
                match self.eval_stmt(scope, body) {
                    Err(EvalAltResult::LoopBreak(x)) => return Ok(x),
                    Err(x) => return Err(x),
                    _ => (),
                }
            },
            Stmt::Break => Err(EvalAltResult::LoopBreak(Box::new(()))),
            Stmt::BreakWithVal(ref a) => {
                let result = self.eval_expr(scope, a)?;
                Err(EvalAltResult::LoopBreak(result))
            }
            Stmt::Return => Err(EvalAltResult::Return(Box::new(()))),
            Stmt::ReturnWithVal(ref a) => {
                let result = self.eval_expr(scope, a)?;
                Err(EvalAltResult::Return(result))
            }
            Stmt::Var(ref name, ref init) => {
                match *init {
                    Some(ref v) => {
                        let i = self.eval_expr(scope, v)?;
                        scope.push((name.clone(), i));
                    }
                    None => scope.push((name.clone(), Box::new(()))),
                };
                Ok(Box::new(()))
            }
        }
    }

    fn format_type(&self, tid: TypeId) -> String {
        if let Some(name) = self.type_names.get(&tid) {
            name.to_string()
        } else {
            format!("<unknown> {:?}", tid)
        }
    }

    /// Evaluate a file
    pub fn eval_file<T: Any + Clone>(&mut self, fname: &str) -> Result<T, EvalAltResult> {
        use std::fs::File;
        use std::io::prelude::*;

        if let Ok(mut f) = File::open(fname) {
            let mut contents = String::new();

            if f.read_to_string(&mut contents).is_ok() {
                self.eval::<T>(&contents)
            } else {
                Err(EvalAltResult::ErrorCantOpenScriptFile)
            }
        } else {
            Err(EvalAltResult::ErrorCantOpenScriptFile)
        }
    }

    /// Evaluate a string
    pub fn eval<T: Any + Clone>(&mut self, input: &str) -> Result<T, EvalAltResult> {
        let mut scope: Scope = Vec::new();

        self.eval_with_scope(&mut scope, input)
    }

    /// Evaluate with own scope
    pub fn eval_with_scope<T: Any + Clone>(
        &mut self,
        scope: &mut Scope,
        input: &str,
    ) -> Result<T, EvalAltResult> {
        let tokens = lex(input);

        let mut peekables = tokens.peekable();
        let tree = parse(&mut peekables);

        match tree {
            Ok((ref os, ref fns)) => {
                let mut x: Result<Box<Any>, EvalAltResult> = Ok(Box::new(()));

                for f in fns {
                    let local_f = f.clone();
                    let spec = FnSpec::new(f.name.clone(), None);

                    self.fns.insert(spec, Arc::new(FnIntExt::Int(local_f)));
                }

                for o in os {
                    x = match self.eval_stmt(scope, o) {
                        Ok(v) => Ok(v),
                        Err(e) => return Err(e),
                    }
                }

                let x = x?;

                match x.downcast::<T>() {
                    Ok(out) => Ok(*out),
                    Err(a) => Err(EvalAltResult::ErrorMismatchOutputType(
                        self.format_type(Any::type_id(&*a)),
                    )),
                }
            }
            Err(_) => Err(EvalAltResult::ErrorFunctionArgMismatch),
        }
    }

    /// Evaluate a file, but only return errors, if there are any.
    /// Useful for when you don't need the result, but still need
    /// to keep track of possible errors
    pub fn consume_file(&mut self, fname: &str) -> Result<(), EvalAltResult> {
        use std::fs::File;
        use std::io::prelude::*;

        if let Ok(mut f) = File::open(fname) {
            let mut contents = String::new();

            if f.read_to_string(&mut contents).is_ok() {
                if let e @ Err(_) = self.consume(&contents) {
                    e
                } else {
                    Ok(())
                }
            } else {
                Err(EvalAltResult::ErrorCantOpenScriptFile)
            }
        } else {
            Err(EvalAltResult::ErrorCantOpenScriptFile)
        }
    }

    /// Evaluate a string, but only return errors, if there are any.
    /// Useful for when you don't need the result, but still need
    /// to keep track of possible errors
    pub fn consume(&mut self, input: &str) -> Result<(), EvalAltResult> {
        self.consume_with_scope(&mut Scope::new(), input)
    }

    /// Evaluate a string with own scoppe, but only return errors, if there are any.
    /// Useful for when you don't need the result, but still need
    /// to keep track of possible errors
    pub fn consume_with_scope(
        &mut self,
        scope: &mut Scope,
        input: &str,
    ) -> Result<(), EvalAltResult> {
        let tokens = lex(input);

        let mut peekables = tokens.peekable();
        let tree = parse(&mut peekables);

        match tree {
            Ok((ref os, ref fns)) => {
                for f in fns {
                    if f.params.len() > 6 {
                        return Ok(());
                    }
                    let local_f = f.clone();
                    let spec = FnSpec::new(f.name.clone(), None);

                    self.fns.insert(spec, Arc::new(FnIntExt::Int(local_f)));
                }

                for o in os {
                    if let Err(e) = self.eval_stmt(scope, o) {
                        return Err(e);
                    }
                }

                Ok(())
            }
            Err(_) => Err(EvalAltResult::ErrorFunctionArgMismatch),
        }
    }

    /// Register the default library. That means, numberic types, char, bool
    /// String, arithmetics and string concatenations.
    pub fn register_default_lib(engine: &mut Engine) {
        register_type!(engine, i32);
        register_type!(engine, u32);
        register_type!(engine, i64);
        register_type!(engine, u64);
        register_type!(engine, usize);
        register_type!(engine, f32);
        register_type!(engine, f64);
        register_type!(engine, String);
        register_type!(engine, char);
        register_type!(engine, bool);
        engine.register_type(Type::new::<Vec<Box<Any>>>("array"));

        macro_rules! reg_op {
            ($engine:expr, $x:expr, $op:expr, $( $y:ty ),*) => (
                $(
                    $engine.register_fn($x, ($op as fn(x: $y, y: $y)->$y));
                )*
            )
        }

        macro_rules! reg_un {
            ($engine:expr, $x:expr, $op:expr, $( $y:ty ),*) => (
                $(
                    $engine.register_fn($x, ($op as fn(x: $y)->$y));
                )*
            )
        }

        macro_rules! reg_cmp {
            ($engine:expr, $x:expr, $op:expr, $( $y:ty ),*) => (
                $(
                    $engine.register_fn($x, ($op as fn(x: $y, y: $y)->bool));
                )*
            )
        }

        fn add<T: Add>(x: T, y: T) -> <T as Add>::Output { x + y }
        fn sub<T: Sub>(x: T, y: T) -> <T as Sub>::Output { x - y }
        fn mul<T: Mul>(x: T, y: T) -> <T as Mul>::Output { x * y }
        fn div<T: Div>(x: T, y: T) -> <T as Div>::Output { x / y }
        fn neg<T: Neg>(x: T) -> <T as Neg>::Output { -x }
        fn lt<T: PartialOrd>(x: T, y: T) -> bool { x < y }
        fn lte<T: PartialOrd>(x: T, y: T) -> bool { x <= y }
        fn gt<T: PartialOrd>(x: T, y: T) -> bool { x > y }
        fn gte<T: PartialOrd>(x: T, y: T) -> bool { x >= y }
        fn eq<T: PartialEq>(x: T, y: T) -> bool { x == y }
        fn ne<T: PartialEq>(x: T, y: T) -> bool { x != y }
        fn and(x: bool, y: bool) -> bool { x && y }
        fn or(x: bool, y: bool) -> bool { x || y }
        fn not(x: bool) -> bool { !x }
        fn concat(x: String, y: String) -> String { x + &y }
        fn binary_and<T: BitAnd>(x: T, y: T) -> <T as BitAnd>::Output { x & y }
        fn binary_or<T: BitOr>(x: T, y: T) -> <T as BitOr>::Output { x | y }
        fn binary_xor<T: BitXor>(x: T, y: T) -> <T as BitXor>::Output { x ^ y }
        fn left_shift<T: Shl<T>>(x: T, y: T) -> <T as Shl<T>>::Output { x.shl(y) }
        fn right_shift<T: Shr<T>>(x: T, y: T) -> <T as Shr<T>>::Output { x.shr(y) }
        fn modulo<T: Rem<T>>(x: T, y: T) -> <T as Rem<T>>::Output { x % y }
        fn pow_i64_i64(x: i64, y: i64) -> i64 { x.pow(y as u32) }
        fn pow_f64_f64(x: f64, y: f64) -> f64 { x.powf(y) }
        fn pow_f64_i64(x: f64, y: i64) -> f64 { x.powi(y as i32) }
        fn unit_eq(a: (), b: ()) -> bool { true }

        reg_op!(engine, "+", add, i32, i64, u32, u64, f32, f64);
        reg_op!(engine, "-", sub, i32, i64, u32, u64, f32, f64);
        reg_op!(engine, "*", mul, i32, i64, u32, u64, f32, f64);
        reg_op!(engine, "/", div, i32, i64, u32, u64, f32, f64);

        reg_cmp!(engine, "<", lt, i32, i64, u32, u64, String, f64);
        reg_cmp!(engine, "<=", lte, i32, i64, u32, u64, String, f64);
        reg_cmp!(engine, ">", gt, i32, i64, u32, u64, String, f64);
        reg_cmp!(engine, ">=", gte, i32, i64, u32, u64, String, f64);
        reg_cmp!(engine, "==", eq, i32, i64, u32, u64, bool, String, f64);
        reg_cmp!(engine, "!=", ne, i32, i64, u32, u64, bool, String, f64);

        reg_op!(engine, "||", or, bool);
        reg_op!(engine, "&&", and, bool);
        reg_op!(engine, "|", binary_or, i32, i64, u32, u64);
        reg_op!(engine, "|", or, bool);
        reg_op!(engine, "&", binary_and, i32, i64, u32, u64);
        reg_op!(engine, "&", and, bool);
        reg_op!(engine, "^", binary_xor, i32, i64, u32, u64);
        reg_op!(engine, "<<", left_shift, i32, i64, u32, u64);
        reg_op!(engine, ">>", right_shift, i32, i64, u32, u64);
        reg_op!(engine, "%", modulo, i32, i64, u32, u64);
        engine.register_fn("~", pow_i64_i64);
        engine.register_fn("~", pow_f64_f64);
        engine.register_fn("~", pow_f64_i64);

        reg_un!(engine, "-", neg, i32, i64, f32, f64);
        reg_un!(engine, "!", not, bool);

        engine.register_fn("+", concat);
        engine.register_fn("==", unit_eq);

        // engine.register_fn("[]", idx);
        // FIXME?  Registering array lookups are a special case because we want to return boxes
        // directly let ent = engine.fns.entry("[]".to_string()).or_insert_with(Vec::new);
        // (*ent).push(FnType::ExternalFn2(Box::new(idx)));
    }

    /// Make a new engine
    pub fn new() -> Engine {
        let mut engine = Engine {
            fns: HashMap::new(),
            type_names: HashMap::new(),
            types: HashMap::new(),
        };

        Engine::register_default_lib(&mut engine);

        engine
    }
}

pub struct Ref<A>(A);
pub struct Mut<A>(A);

macro_rules! count_args {
    () => {0usize};
    ($head:ident $($tail:ident)*) => {1usize + count_args!($($tail)*)};
}

macro_rules! def_register {
    () => {
        def_register!(imp);
    };
    (imp $($par:ident => $mark:ty => $param:ty => $clone:expr),*) => {
        impl<$($par,)* FN, RET> RegisterFn<FN, ($($mark,)*), RET> for Engine
        where
            $($par: Any + Clone,)*
            FN: Fn($($param),*) -> RET + 'static,
            RET: Any,
        {
            fn register_fn(&mut self, name: &str, f: FN) {
                let fun = move |mut args: Vec<&mut Any>| {
                    // Check for length at the beginning to avoid
                    // per-element bound checks.
                    if args.len() != count_args!($($par)*) {
                        return Err(EvalAltResult::ErrorFunctionArgMismatch);
                    }

                    let mut drain = args.drain(..);
                    $(
                    // Downcast every element, return in case of a type mismatch
                    let $par = ((*drain.next().unwrap()).downcast_mut() as Option<&mut $par>)
                        .ok_or(EvalAltResult::ErrorFunctionArgMismatch)?;
                    )*

                    // Call the user-supplied function using ($clone) to
                    // potentially clone the value, otherwise pass the reference.
                    Ok(Box::new(f($(($clone)($par)),*)) as Box<Any>)
                };
                self.register_fn_raw(name.to_owned(), Some(vec![$(TypeId::of::<$par>()),*]), Box::new(fun));
            }
        }

        impl<$($par,)* FN, RET> RegisterTypeFn<FN, ($($mark,)*), RET> for Type
        where
            $($par: Any + Clone,)*
            FN: Fn($($param),*) -> RET + 'static,
            RET: Any,
        {

            fn add_function(mut self, name: &str, f: FN) -> Type {
                let fun = move |mut args: Vec<&mut Any>| {
                    // Check for length at the beginning to avoid
                    // per-element bound checks.
                    if args.len() != count_args!($($par)*) {
                        return Err(EvalAltResult::ErrorFunctionArgMismatch);
                    }

                    let mut drain = args.drain(..);
                    $(
                    // Downcast every element, return in case of a type mismatch
                    let $par = ((*drain.next().unwrap()).downcast_mut() as Option<&mut $par>)
                        .ok_or(EvalAltResult::ErrorFunctionArgMismatch)?;
                    )*

                    // Call the user-supplied function using ($clone) to
                    // potentially clone the value, otherwise pass the reference.
                    Ok(Box::new(f($(($clone)($par)),*)) as Box<Any>)
                };
                let spec = FnSpec::new(name.to_string(), Some(vec![$(TypeId::of::<$par>()),*]));
                self.functions.insert(spec, Arc::new(Box::new(fun)));
                self
            }
        }

        //def_register!(imp_pop $($par => $mark => $param),*);
    };
    ($p0:ident $(, $p:ident)*) => {
        def_register!(imp $p0 => $p0 => $p0 => Clone::clone $(, $p => $p => $p => Clone::clone)*);
        def_register!(imp $p0 => Ref<$p0> => &$p0 => |x| { x } $(, $p => $p => $p => Clone::clone)*);
        def_register!(imp $p0 => Mut<$p0> => &mut $p0 => |x| { x } $(, $p => $p => $p => Clone::clone)*);

        def_register!($($p),*);
    };
//    (imp_pop) => {};
//    (imp_pop $head:ident => $head_mark:ty => $head_param:ty $(,$tail:ident => $tail_mark:ty => $tp:ty)*) => {
//        def_register!(imp $($tail => $tail_mark => $tp),*);
//    };
}

#[cfg_attr(rustfmt, rustfmt_skip)]
def_register!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S);