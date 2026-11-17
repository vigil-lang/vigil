//! this module implements the interpreter of `comptime` functions and
//! `comptime:` blocks, over the HIR and its inference results. The arithmetic
//! is the one of eBPF, because the value replaces one computed at run time:
//! integers wrap, `x // 0` is `0`, `x % 0` is `x`, shifts are modulo the
//! width. `FUEL` and `MAX_DEPTH` bound an evaluation, so a loop that never
//! ends gives an error and does not hang the compiler

use la_arena::ArenaMap;

use crate::db::{ConstValue, FnSig};
use crate::hir::*;
use crate::infer::{InferenceResult, Res};
use crate::resolve::{Builtin, Math};
use crate::ty::{IntTy, TyId, TyKind, Types};
use vigil_diagnostics::Span;

/// a value computed at compile time
#[derive(Clone, Debug, PartialEq)]
pub enum CValue {
    /// an integer of any type, already wrapped to it
    Int(i128),
    Bool(bool),
    Float(f64),
    Array(Vec<CValue>),
    /// the fields of a struct, in declaration order
    Struct(Vec<CValue>),
    Option(Option<Box<CValue>>),
    /// a variant of an enum, by index, with its fields
    Enum(u32, Vec<CValue>),
    /// what a function without a return value gives
    Unit,
}

/// the steps an evaluation may take
const FUEL: u64 = 10_000_000;
/// how deep the calls of an evaluation may nest
const MAX_DEPTH: u32 = 64;

/// how a statement ends: go to the next one, or leave the loop or function
enum Flow {
    Next,
    Return(CValue),
    Break,
    Continue,
}

/// the result of a step of an evaluation, or why it failed
type Eval<T> = Result<T, EvalError>;

/// why an evaluation failed, and where
#[derive(Clone, Debug)]
pub(crate) struct EvalError {
    pub message: String,
    /// the failing expression or statement, then the call of each function
    /// it is in, innermost first, each with the function it is written in
    pub places: Vec<(FnId, Span)>,
    /// whether the place in the current function is recorded already
    located: bool,
    /// why the rule exists, when it is not obvious
    pub note: Option<String>,
    /// the constant whose value was missing
    pub missing: Option<ConstId>,
}

impl EvalError {
    fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// recording where the error is in the current function, unless an
    /// inner expression did already
    fn at(mut self, function: FnId, span: Option<Span>) -> Self {
        if let (false, Some(span)) = (self.located, span) {
            self.places.push((function, span));
            self.located = true;
        }
        self
    }
}

impl From<String> for EvalError {
    fn from(message: String) -> Self {
        EvalError {
            message,
            places: Vec::new(),
            located: false,
            note: None,
            missing: None,
        }
    }
}

impl From<&str> for EvalError {
    fn from(message: &str) -> Self {
        message.to_string().into()
    }
}

/// the interpreter, with what it reads from the analysis and the fuel and
/// depth left
pub(crate) struct Interp<'a> {
    pub package: &'a Package,
    pub bodies: &'a ArenaMap<FnId, InferenceResult>,
    pub types: &'a Types,
    pub consts: &'a ArenaMap<ConstId, (TyId, Option<ConstValue>)>,
    pub comptime_consts: &'a ArenaMap<ConstId, CValue>,
    /// the signatures of functions, for the default values of parameters
    pub fn_sigs: &'a ArenaMap<FnId, FnSig>,
    fuel: u64,
    depth: u32,
}

/// wrapping an integer to the width of a type, as eBPF does: the low bits,
/// read as signed or not
pub fn wrap(value: i128, int: IntTy) -> i128 {
    let bits = int.bits();
    let mask = (1u128 << bits) - 1;
    let raw = (value as u128) & mask;
    if int.signed() && raw >> (bits - 1) == 1 {
        raw as i128 - (1i128 << bits)
    } else {
        raw as i128
    }
}

/// the activation of a function: its body, its inference result and the
/// values of its locals (`None` until assigned)
struct Frame<'b> {
    function: FnId,
    body: &'b Body,
    spans: &'b BodySpans,
    res: &'b InferenceResult,
    locals: Vec<Option<CValue>>,
}

impl<'a> Interp<'a> {
    /// creating an interpreter with full fuel. The fuel is never refilled, so
    /// it bounds everything this interpreter runs.
    pub fn new(
        package: &'a Package,
        bodies: &'a ArenaMap<FnId, InferenceResult>,
        types: &'a Types,
        consts: &'a ArenaMap<ConstId, (TyId, Option<ConstValue>)>,
        comptime_consts: &'a ArenaMap<ConstId, CValue>,
        fn_sigs: &'a ArenaMap<FnId, FnSig>,
    ) -> Self {
        Interp {
            package,
            bodies,
            types,
            consts,
            comptime_consts,
            fn_sigs,
            fuel: FUEL,
            depth: 0,
        }
    }

    /// the default values of the parameters of `f`, without `self` for a
    /// method
    fn defaults(&self, f: FnId, method: bool) -> Vec<Option<CValue>> {
        let Some(sig) = self.fn_sigs.get(f) else {
            return Vec::new();
        };
        let defaults = if method {
            sig.call_defaults()
        } else {
            &sig.defaults[..]
        };
        defaults
            .iter()
            .map(|d| match *d {
                Some(ConstValue::Int(v)) => Some(CValue::Int(v)),
                Some(ConstValue::Bool(b)) => Some(CValue::Bool(b)),
                None => None,
            })
            .collect()
    }

    /// taking one unit of fuel, or failing when there is none left
    fn step(&mut self) -> Eval<()> {
        if self.fuel == 0 {
            return Err(
                EvalError::from("the evaluation does not end (or takes too long)").with_note(
                    format!("the compiler stops a compile-time evaluation after {FUEL} steps"),
                ),
            );
        }
        self.fuel -= 1;
        Ok(())
    }

    /// calling a function, with the arguments in parameter order
    pub fn call(&mut self, function: FnId, args: Vec<CValue>) -> Eval<CValue> {
        Ok(self.call_keeping_self(function, args)?.0)
    }

    /// calling a function, returning also the final value of the first
    /// parameter (what a `@mut` method leaves in `self`)
    pub(crate) fn call_keeping_self(
        &mut self,
        function: FnId,
        args: Vec<CValue>,
    ) -> Eval<(CValue, CValue)> {
        let data = &self.package.fns[function];
        // here we stop at a helper reached through a run-time function, a
        // direct call from comptime code is a type error already
        if data
            .decorators
            .iter()
            .any(|d| d.name.to_string() == "helper")
        {
            return Err(EvalError::from(format!(
                "`{}` cannot be called at compile time",
                data.name.text
            ))
            .with_note("it is a kernel helper, which only exists in the running program"));
        }
        let Some(res) = self.bodies.get(function) else {
            return Err(format!("`{}` cannot be evaluated at compile time", data.name.text).into());
        };
        if self.depth >= MAX_DEPTH {
            return Err(EvalError::from("too many nested calls")
                .with_note(format!("compile-time calls nest at most {MAX_DEPTH} deep")));
        }
        self.depth += 1;
        let mut frame = Frame {
            function,
            body: &data.body,
            spans: &data.spans,
            res,
            locals: vec![None; res.locals.len()],
        };
        for (i, arg) in args.into_iter().enumerate() {
            frame.locals[i] = Some(arg);
        }
        // an error leaving the function is placed again at the call
        let flow = self.block(&mut frame, &data.body.root).map_err(|mut e| {
            e.located = false;
            e
        });
        self.depth -= 1;
        let first = frame
            .locals
            .first()
            .cloned()
            .flatten()
            .unwrap_or(CValue::Unit);
        match flow? {
            Flow::Return(value) => Ok((value, first)),
            _ => Ok((CValue::Unit, first)),
        }
    }

    /// running the `comptime:` blocks of a run-time function in order, in one
    /// frame, so a block sees what the blocks before it assigned. It returns
    /// the values of the variables after each block, or the failed block and
    /// the message.
    pub fn run_blocks(
        &mut self,
        function: FnId,
        blocks: &[StmtId],
    ) -> Result<crate::BlockValues, (StmtId, EvalError)> {
        let data = &self.package.fns[function];
        let Some(res) = self.bodies.get(function) else {
            return Ok(Vec::new());
        };
        let mut frame = Frame {
            function,
            body: &data.body,
            spans: &data.spans,
            res,
            locals: vec![None; res.locals.len()],
        };
        let mut out = Vec::new();
        for &block in blocks {
            self.stmt(&mut frame, block).map_err(|e| (block, e))?;
            let values = frame
                .locals
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    let id = la_arena::Idx::from_raw(la_arena::RawIdx::from_u32(i as u32));
                    v.clone().map(|v| (id, v))
                })
                .collect();
            out.push((block, values));
        }
        Ok(out)
    }

    /// running the statements of a block until one leaves it
    fn block(&mut self, frame: &mut Frame, block: &Block) -> Eval<Flow> {
        for &stmt in block {
            match self.stmt(frame, stmt)? {
                Flow::Next => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Next)
    }

    /// running a statement, placing an error at it
    fn stmt(&mut self, frame: &mut Frame, id: StmtId) -> Eval<Flow> {
        self.stmt_here(frame, id)
            .map_err(|e| e.at(frame.function, frame.spans.stmts.get(id).copied()))
    }

    fn stmt_here(&mut self, frame: &mut Frame, id: StmtId) -> Eval<Flow> {
        self.step()?;
        let body = frame.body;
        match &body.stmts[id] {
            Stmt::Missing | Stmt::Pass => {}
            Stmt::Expr(e) => {
                self.expr(frame, *e)?;
            }
            Stmt::Assign { target, value, .. } => {
                let Some(value) = value else {
                    return Ok(Flow::Next);
                };
                let value = self.expr(frame, *value)?;
                match frame.res.decl_locals.get(id) {
                    Some(&local) => {
                        frame.locals[local.into_raw().into_u32() as usize] = Some(value)
                    }
                    None => self.assign(frame, *target, value)?,
                }
            }
            Stmt::AugAssign { target, op, value } => {
                let current = self.expr(frame, *target)?;
                let rhs = self.expr(frame, *value)?;
                let result = match frame.res.aug_operators.get(id) {
                    Some(&method) => self.call(method, vec![current, rhs])?,
                    None => {
                        let ty = frame.res.expr_types[*target];
                        self.binary(*op, current, rhs, ty)?
                    }
                };
                self.assign(frame, *target, result)?;
            }
            Stmt::Return(value) => {
                let value = match value {
                    Some(v) => self.expr(frame, *v)?,
                    None => CValue::Unit,
                };
                return Ok(Flow::Return(value));
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                if self.boolean(frame, *cond)? {
                    return self.block(frame, then);
                } else if let Some(otherwise) = otherwise {
                    return self.block(frame, otherwise);
                }
            }
            Stmt::For {
                iter,
                body: loop_body,
                ..
            } => {
                let local = frame.res.for_locals[id].into_raw().into_u32() as usize;
                let items: Vec<CValue> = match &body.exprs[*iter] {
                    Expr::Call { callee, args }
                        if frame.res.res.get(*callee) == Some(&Res::Builtin(Builtin::Range)) =>
                    {
                        let bounds: Vec<i128> = args
                            .iter()
                            .map(|a| self.int(frame, a.value))
                            .collect::<Eval<_>>()?;
                        let (start, end) = match bounds.as_slice() {
                            [end] => (0, *end),
                            [start, end] => (*start, *end),
                            _ => return Err("invalid range".into()),
                        };
                        // the items are built before the loop runs, so here we
                        // stop a range longer than the fuel before allocating it
                        if end - start > FUEL as i128 {
                            return Err(EvalError::from(format!(
                                "the range has {} items",
                                end - start
                            ))
                            .with_note(format!(
                                "the compiler stops a compile-time evaluation after {FUEL} steps"
                            ))
                            .at(frame.function, frame.spans.exprs.get(*iter).copied()));
                        }
                        (start..end.max(start)).map(CValue::Int).collect()
                    }
                    _ => match self.expr(frame, *iter)? {
                        CValue::Array(items) => items,
                        _ => return Err("cannot iterate over this value".into()),
                    },
                };
                for item in items {
                    frame.locals[local] = Some(item);
                    match self.block(frame, loop_body)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        Flow::Next | Flow::Continue => {}
                    }
                }
            }
            Stmt::While {
                cond,
                body: loop_body,
                ..
            } => {
                // `@bounded(N)` stops the loop after N iterations, as the
                // compiled code does
                let bound = frame.res.loop_bounds.get(id).copied();
                let mut count = 0u64;
                while bound.is_none_or(|b| count < b) && self.boolean(frame, *cond)? {
                    count += 1;
                    self.step()?;
                    match self.block(frame, loop_body)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        Flow::Next | Flow::Continue => {}
                    }
                }
            }
            Stmt::Break => return Ok(Flow::Break),
            Stmt::Continue => return Ok(Flow::Continue),
            Stmt::Match { scrutinee, arms } => {
                let value = self.expr(frame, *scrutinee)?;
                for arm in arms {
                    if !self.matches(frame, arm.pat, &value)? {
                        continue;
                    }
                    if let Some(guard) = arm.guard
                        && !self.boolean(frame, guard)?
                    {
                        continue;
                    }
                    return self.block(frame, &arm.body);
                }
            }
            Stmt::Comptime(block) => return self.block(frame, block),
        }
        Ok(Flow::Next)
    }

    /// whether `value` matches the pattern, binding its names on the way. A
    /// string or float pattern is an error
    fn matches(&mut self, frame: &mut Frame, pat: PatId, value: &CValue) -> Eval<bool> {
        let body = frame.body;
        Ok(match &body.pats[pat] {
            Pat::Missing | Pat::Wild => true,
            Pat::Bind(_) => {
                let local = frame.res.pat_locals[pat].into_raw().into_u32() as usize;
                frame.locals[local] = Some(value.clone());
                true
            }
            Pat::Int { value: v, negative } => {
                let v = i128::try_from(*v).map_err(|_| "integer too large".to_string())?;
                *value == CValue::Int(if *negative { -v } else { v })
            }
            Pat::Bool(b) => *value == CValue::Bool(*b),
            Pat::None => *value == CValue::Option(None),
            Pat::Str | Pat::Float => return Err("this pattern cannot match at compile time".into()),
            Pat::Path(_) => match frame.res.pat_res.get(pat).copied() {
                Some(Res::Variant(_, i)) => matches!(value, CValue::Enum(tag, _) if *tag == i),
                Some(Res::Const(c)) => *value == self.constant(c)?,
                _ => return Err("invalid pattern".into()),
            },
            Pat::TupleStruct { fields, .. } => match frame.res.pat_res.get(pat).copied() {
                Some(Res::Builtin(Builtin::Some)) => match (value, fields.as_slice()) {
                    (CValue::Option(Some(inner)), [field]) => {
                        let inner = (**inner).clone();
                        self.matches(frame, *field, &inner)?
                    }
                    _ => false,
                },
                Some(Res::Variant(_, i)) => match value {
                    CValue::Enum(tag, values) if *tag == i => {
                        let values = values.clone();
                        for (field, v) in fields.iter().zip(&values) {
                            if !self.matches(frame, *field, v)? {
                                return Ok(false);
                            }
                        }
                        true
                    }
                    _ => false,
                },
                _ => return Err("invalid pattern".into()),
            },
            Pat::Or(alternatives) => {
                for &alternative in alternatives {
                    if self.matches(frame, alternative, value)? {
                        return Ok(true);
                    }
                }
                false
            }
        })
    }

    /// the value of a constant, computed by a comptime function or written as
    /// an integer or boolean
    fn constant(&self, c: ConstId) -> Eval<CValue> {
        match (self.comptime_consts.get(c), self.consts.get(c)) {
            (Some(value), _) => Ok(value.clone()),
            (None, Some((_, Some(ConstValue::Int(v))))) => Ok(CValue::Int(*v)),
            (None, Some((_, Some(ConstValue::Bool(b))))) => Ok(CValue::Bool(*b)),
            _ => {
                let mut error = EvalError::from(format!(
                    "`{}` has no value yet",
                    self.package.consts[c].name.text
                ));
                error.missing = Some(c);
                Err(error)
            }
        }
    }

    /// the value of an expression that must be a `bool`
    fn boolean(&mut self, frame: &mut Frame, e: ExprId) -> Eval<bool> {
        match self.expr(frame, e)? {
            CValue::Bool(b) => Ok(b),
            _ => Err("expected a `bool`".into()),
        }
    }

    /// the value of an expression that must be an integer
    fn int(&mut self, frame: &mut Frame, e: ExprId) -> Eval<i128> {
        match self.expr(frame, e)? {
            CValue::Int(v) => Ok(v),
            _ => Err("expected an integer".into()),
        }
    }

    /// the local and the path of field and element indices of an assignable
    /// expression. Indices are checked here, before anything is assigned.
    fn place(&mut self, frame: &mut Frame, e: ExprId) -> Eval<(usize, Vec<usize>)> {
        match &frame.body.exprs[e] {
            Expr::Name(_) => match frame.res.res.get(e) {
                Some(Res::Local(local)) => Ok((local.into_raw().into_u32() as usize, Vec::new())),
                _ => Err("not assignable".into()),
            },
            Expr::Field { base, .. } => {
                let Some(Res::Field(i)) = frame.res.res.get(e).copied() else {
                    return Err("not assignable".into());
                };
                let (local, mut path) = self.place(frame, *base)?;
                path.push(i as usize);
                Ok((local, path))
            }
            Expr::Index { base, indices } => {
                let index = self.int(frame, *indices.first().ok_or("an index is missing")?)?;
                let (local, mut path) = self.place(frame, *base)?;
                let len = match self.types.kind(frame.res.expr_types[*base]) {
                    TyKind::Array(_, len) => i128::from(*len),
                    _ => 0,
                };
                if !(0..len).contains(&index) {
                    return Err(format!("index {index} is out of bounds (length {len})").into());
                }
                path.push(index as usize);
                Ok((local, path))
            }
            _ => Err("not assignable".into()),
        }
    }

    /// writing a value to an assignable expression
    fn assign(&mut self, frame: &mut Frame, target: ExprId, value: CValue) -> Eval<()> {
        let (local, path) = self
            .place(frame, target)
            .map_err(|e| e.at(frame.function, frame.spans.exprs.get(target).copied()))?;
        let mut slot = frame.locals[local]
            .as_mut()
            .ok_or_else(|| "use of an unassigned variable".to_string())?;
        for index in path {
            slot = match slot {
                CValue::Array(items) | CValue::Struct(items) => items
                    .get_mut(index)
                    .ok_or_else(|| "index out of bounds".to_string())?,
                _ => return Err("not assignable".into()),
            };
        }
        *slot = value;
        Ok(())
    }

    /// the value of an expression, placing an error at it
    fn expr(&mut self, frame: &mut Frame, e: ExprId) -> Eval<CValue> {
        self.expr_here(frame, e)
            .map_err(|err| err.at(frame.function, frame.spans.exprs.get(e).copied()))
    }

    fn expr_here(&mut self, frame: &mut Frame, e: ExprId) -> Eval<CValue> {
        self.step()?;
        let body = frame.body;
        let ty = frame.res.expr_types.get(e).copied();
        let value = match &body.exprs[e] {
            Expr::Int(v) => {
                let v = i128::try_from(*v).map_err(|_| "integer too large".to_string())?;
                match ty.map(|t| self.types.kind(t)) {
                    Some(TyKind::Int(int)) => CValue::Int(wrap(v, *int)),
                    _ => CValue::Int(v),
                }
            }
            Expr::Float(v) => CValue::Float(*v),
            Expr::Bool(b) => CValue::Bool(*b),
            Expr::None => CValue::Option(None),
            Expr::Name(_) | Expr::Field { .. } => match frame.res.res.get(e).copied() {
                Some(Res::Local(local)) => frame.locals[local.into_raw().into_u32() as usize]
                    .clone()
                    .ok_or_else(|| "use of an unassigned variable".to_string())?,
                Some(Res::Const(c)) => self.constant(c)?,
                Some(Res::ConstValue(v)) => CValue::Int(v),
                Some(Res::Variant(_, i)) => CValue::Enum(i, Vec::new()),
                Some(Res::Field(i)) => {
                    let Expr::Field { base, .. } = body.exprs[e] else {
                        return Err("invalid field access".into());
                    };
                    match self.expr(frame, base)? {
                        CValue::Struct(fields) => fields[i as usize].clone(),
                        _ => return Err("not a struct".into()),
                    }
                }
                _ => return Err("this value is not available at compile time".into()),
            },
            Expr::Index { base, indices } => {
                let base = self.expr(frame, *base)?;
                let index = self.int(frame, *indices.first().ok_or("an index is missing")?)?;
                match base {
                    CValue::Array(items) => usize::try_from(index)
                        .ok()
                        .and_then(|i| items.get(i).cloned())
                        .ok_or_else(|| {
                            EvalError::from(format!(
                                "index {index} is out of bounds (length {})",
                                items.len()
                            ))
                        })?,
                    _ => return Err("not an array".into()),
                }
            }
            Expr::Array(elements) => {
                let elements = elements.clone();
                CValue::Array(
                    elements
                        .iter()
                        .map(|&x| self.expr(frame, x))
                        .collect::<Eval<_>>()?,
                )
            }
            Expr::Unary { op, operand } => {
                if let Some(Res::Operator(method)) = frame.res.res.get(e).copied() {
                    let value = self.expr(frame, *operand)?;
                    return self.call(method, vec![value]);
                }
                let value = self.expr(frame, *operand)?;
                match (op, value) {
                    (UnaryOp::Not, CValue::Bool(b)) => CValue::Bool(!b),
                    (UnaryOp::Neg, CValue::Float(f)) => CValue::Float(-f),
                    (UnaryOp::Neg, CValue::Int(v)) => self.int_result(-v, ty),
                    (UnaryOp::BitNot, CValue::Int(v)) => self.int_result(!v, ty),
                    _ => return Err("invalid operand".into()),
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                match frame.res.res.get(e).copied() {
                    Some(Res::Operator(method)) => {
                        let (a, b) = (self.expr(frame, lhs)?, self.expr(frame, rhs)?);
                        return self.call(method, vec![a, b]);
                    }
                    Some(Res::Repeat(n)) => {
                        let Expr::Array(elements) = &body.exprs[lhs] else {
                            return Err("invalid repetition".into());
                        };
                        let element = self.expr(frame, elements[0])?;
                        return Ok(CValue::Array(vec![element; n as usize]));
                    }
                    _ => {}
                }
                match op {
                    BinaryOp::And => {
                        CValue::Bool(self.boolean(frame, lhs)? && self.boolean(frame, rhs)?)
                    }
                    BinaryOp::Or => {
                        CValue::Bool(self.boolean(frame, lhs)? || self.boolean(frame, rhs)?)
                    }
                    _ => {
                        let a = self.expr(frame, lhs)?;
                        let b = self.expr(frame, rhs)?;
                        let operand_ty = frame.res.expr_types[lhs];
                        let value = self.binary(op, a, b, operand_ty)?;
                        return Ok(value);
                    }
                }
            }
            Expr::Call { callee, args } => return self.call_expr(frame, *callee, args),
            Expr::Try(_) => return Err("`?` is not supported in comptime functions yet".into()),
            Expr::Str(_) | Expr::FStr(_) => return Err("strings only exist in `print`".into()),
            Expr::Missing | Expr::Ellipsis => return Err("invalid expression".into()),
        };
        Ok(value)
    }

    /// wrapping an integer to the type of an expression, when it is an
    /// integer type
    fn int_result(&self, value: i128, ty: Option<TyId>) -> CValue {
        match ty.map(|t| self.types.kind(t)) {
            Some(TyKind::Int(int)) => CValue::Int(wrap(value, *int)),
            _ => CValue::Int(value),
        }
    }

    /// a binary operation, where `ty` is the type of the left operand and sets
    /// the width of the result. If it is not an integer type we use `i64`.
    fn binary(&mut self, op: BinaryOp, a: CValue, b: CValue, ty: TyId) -> Eval<CValue> {
        use BinaryOp::*;
        Ok(match (a, b) {
            (CValue::Int(x), CValue::Int(y)) => {
                let int = match self.types.kind(ty) {
                    TyKind::Int(int) => *int,
                    _ => IntTy::I64,
                };
                let shift = |y: i128| (y as u32) % int.bits();
                let value = match op {
                    Add => x.wrapping_add(y),
                    Sub => x.wrapping_sub(y),
                    Mul => x.wrapping_mul(y),
                    FloorDiv if y == 0 => 0,
                    FloorDiv => x / y,
                    Rem if y == 0 => x,
                    Rem => x % y,
                    BitAnd => x & y,
                    BitOr => x | y,
                    BitXor => x ^ y,
                    Shl => x.wrapping_shl(shift(y)),
                    Shr if int.signed() => x >> shift(y),
                    Shr => ((x as u128 & ((1u128 << int.bits()) - 1)) >> shift(y)) as i128,
                    Eq => return Ok(CValue::Bool(x == y)),
                    Ne => return Ok(CValue::Bool(x != y)),
                    Lt => return Ok(CValue::Bool(x < y)),
                    Le => return Ok(CValue::Bool(x <= y)),
                    Gt => return Ok(CValue::Bool(x > y)),
                    Ge => return Ok(CValue::Bool(x >= y)),
                    Div | And | Or => return Err("invalid operator".into()),
                };
                CValue::Int(wrap(value, int))
            }
            (CValue::Float(x), CValue::Float(y)) => match op {
                Add => CValue::Float(x + y),
                Sub => CValue::Float(x - y),
                Mul => CValue::Float(x * y),
                Div => CValue::Float(x / y),
                Eq => CValue::Bool(x == y),
                Ne => CValue::Bool(x != y),
                Lt => CValue::Bool(x < y),
                Le => CValue::Bool(x <= y),
                Gt => CValue::Bool(x > y),
                Ge => CValue::Bool(x >= y),
                _ => return Err("invalid operator on `f64`".into()),
            },
            (CValue::Bool(x), CValue::Bool(y)) => match op {
                Eq => CValue::Bool(x == y),
                Ne => CValue::Bool(x != y),
                _ => return Err("invalid operator on `bool`".into()),
            },
            _ => return Err("invalid operands".into()),
        })
    }

    /// the value of a call to a function, a method, a constructor or a builtin
    fn call_expr(&mut self, frame: &mut Frame, callee: ExprId, args: &[Arg]) -> Eval<CValue> {
        // the name of the callee as written, for the errors
        let callee_name = match &frame.body.exprs[callee] {
            Expr::Name(name) => name.clone(),
            Expr::Field { name, .. } => name.text.clone(),
            _ => String::new(),
        };
        let cannot_run = || -> EvalError {
            if callee_name.is_empty() {
                "this call cannot run at compile time".into()
            } else {
                format!("`{callee_name}` cannot run at compile time").into()
            }
        };
        // `print` takes strings, which only exist in the running program
        if frame.res.res.get(callee) == Some(&Res::Builtin(Builtin::Print)) {
            return Err(cannot_run());
        }
        let mut values = Vec::new();
        for arg in args {
            values.push(self.expr(frame, arg.value)?);
        }
        match frame.res.res.get(callee).copied() {
            Some(Res::Fn(f)) => {
                let names: Vec<String> = self.package.fns[f]
                    .params
                    .iter()
                    .map(|p| p.name.text.clone())
                    .collect();
                let ordered = order(&names, &self.defaults(f, false), args, values)?;
                self.call(f, ordered)
            }
            Some(Res::Method(f)) => {
                let Expr::Field { base, .. } = frame.body.exprs[callee] else {
                    return Err("invalid method call".into());
                };
                let receiver = self.expr(frame, base)?;
                let names: Vec<String> = self.package.fns[f]
                    .params
                    .iter()
                    .skip(1)
                    .map(|p| p.name.text.clone())
                    .collect();
                let mut ordered = vec![receiver];
                ordered.extend(order(&names, &self.defaults(f, true), args, values)?);
                let changes_self = self.package.fns[f]
                    .decorators
                    .iter()
                    .any(|d| d.name.to_string() == "mut");
                if !changes_self {
                    return self.call(f, ordered);
                }
                // a `@mut` method changes `self` in place, so we write its
                // final value back to the receiver
                let (result, receiver) = self.call_keeping_self(f, ordered)?;
                self.assign(frame, base, receiver)?;
                Ok(result)
            }
            Some(Res::StructCtor(s)) => {
                let names: Vec<String> = self.package.structs[s]
                    .fields
                    .iter()
                    .map(|f| f.name.text.clone())
                    .collect();
                Ok(CValue::Struct(order(&names, &[], args, values)?))
            }
            Some(Res::Variant(_, i)) => Ok(CValue::Enum(i, values)),
            Some(Res::Builtin(builtin @ (Builtin::SaturatingAdd | Builtin::SaturatingSub))) => {
                let int = match args
                    .first()
                    .map(|a| self.types.kind(frame.res.expr_types[a.value]))
                {
                    Some(TyKind::Int(int)) => *int,
                    _ => return Err("invalid operands".into()),
                };
                let [CValue::Int(a), CValue::Int(b)] = values.as_slice() else {
                    return Err("invalid operands".into());
                };
                let exact = if builtin == Builtin::SaturatingAdd {
                    a + b
                } else {
                    a - b
                };
                Ok(CValue::Int(exact.clamp(int.min(), int.max())))
            }
            Some(Res::Builtin(builtin)) => match self.builtin(builtin, values) {
                Err(e) if e.message == UNAVAILABLE_BUILTIN => Err(cannot_run()),
                result => result,
            },
            _ => Err(cannot_run()),
        }
    }

    /// the value of a call to a builtin. Converting an `f64` to an integer
    /// truncates it and fails when the result does not fit.
    fn builtin(&mut self, builtin: Builtin, values: Vec<CValue>) -> Eval<CValue> {
        let one = || {
            values
                .first()
                .cloned()
                .ok_or_else(|| "missing argument".to_string())
        };
        Ok(match builtin {
            Builtin::Some => CValue::Option(Some(Box::new(one()?))),
            Builtin::Int(int) => match one()? {
                CValue::Int(v) => CValue::Int(wrap(v, int)),
                CValue::Bool(b) => CValue::Int(i128::from(b)),
                CValue::Float(f) => {
                    if !f.is_finite()
                        || f.trunc() < int.min() as f64
                        || f.trunc() > int.max() as f64
                    {
                        return Err(format!("{f} does not fit in `{}`", int.name()).into());
                    }
                    CValue::Int(f.trunc() as i128)
                }
                _ => return Err("invalid conversion".into()),
            },
            Builtin::F64 => match one()? {
                CValue::Int(v) => CValue::Float(v as f64),
                CValue::Float(f) => CValue::Float(f),
                _ => return Err("invalid conversion".into()),
            },
            Builtin::Math(math) => {
                let CValue::Float(x) = one()? else {
                    return Err("expected an `f64`".into());
                };
                CValue::Float(match math {
                    Math::Exp => x.exp(),
                    Math::Log => x.ln(),
                    Math::Sqrt => x.sqrt(),
                    Math::Tanh => x.tanh(),
                    Math::Floor => x.floor(),
                    Math::Round => x.round(),
                })
            }
            Builtin::Min | Builtin::Max => {
                let (Some(a), Some(b)) = (values.first(), values.get(1)) else {
                    return Err("missing argument".into());
                };
                let lt = match (a, b) {
                    (CValue::Int(x), CValue::Int(y)) => x < y,
                    (CValue::Float(x), CValue::Float(y)) => x < y,
                    _ => return Err("invalid operands".into()),
                };
                let pick_a = if builtin == Builtin::Min { lt } else { !lt };
                if pick_a { a.clone() } else { b.clone() }
            }
            Builtin::Clamp => match values.as_slice() {
                [CValue::Int(x), CValue::Int(lo), CValue::Int(hi)] => {
                    CValue::Int(*x.max(lo).min(hi))
                }
                _ => return Err("invalid operands".into()),
            },
            _ => return Err(UNAVAILABLE_BUILTIN.into()),
        })
    }
}

/// the error of a builtin that only exists in the running program. The
/// caller replaces it by an error naming the builtin
const UNAVAILABLE_BUILTIN: &str = "this builtin is not available at compile time";

/// the arguments of a call in parameter order: positional ones by position,
/// keyword ones by name, and the defaults for the rest
fn order(
    names: &[String],
    defaults: &[Option<CValue>],
    args: &[Arg],
    values: Vec<CValue>,
) -> Eval<Vec<CValue>> {
    let mut slots: Vec<Option<CValue>> = vec![None; names.len()];
    for (i, (arg, value)) in args.iter().zip(values).enumerate() {
        let slot = match &arg.name {
            Some(name) => names
                .iter()
                .position(|n| *n == name.text)
                .ok_or_else(|| "unknown argument".to_string())?,
            None => i,
        };
        *slots
            .get_mut(slot)
            .ok_or_else(|| "too many arguments".to_string())? = Some(value);
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            s.or_else(|| defaults.get(i).cloned().flatten())
                .ok_or_else(|| "missing argument".into())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_like_the_target() {
        assert_eq!(wrap(256, IntTy::U8), 0);
        assert_eq!(wrap(200, IntTy::I8), -56);
        assert_eq!(wrap(-1, IntTy::U32), u32::MAX as i128);
    }
}
