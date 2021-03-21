use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1},
    combinator::{map_res, opt, recognize},
    multi::{fold_many0, many0},
    number::complete::double,
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};
use std::fs::File;
use std::io::prelude::*;
use std::{cell::RefCell, collections::HashMap, env};

#[derive(Debug, PartialEq, Clone)]
enum Statement<'a> {
    Comment(&'a str),
    VarDecl(&'a str, Option<Expression<'a>>),
    FnDecl(&'a str, Vec<&'a str>, Vec<Statement<'a>>),
    Expression(Expression<'a>),
    Loop(Vec<Statement<'a>>),
    While(Expression<'a>, Vec<Statement<'a>>),
    For(&'a str, Expression<'a>, Expression<'a>, Vec<Statement<'a>>),
    Break,
}

#[derive(Debug, PartialEq, Clone)]
enum Expression<'a> {
    NumLiteral(f64),
    Variable(&'a str),
    VarAssign(&'a str, Box<Expression<'a>>),
    FnInvoke(&'a str, Vec<Expression<'a>>),
    Add(Box<Expression<'a>>, Box<Expression<'a>>),
    Sub(Box<Expression<'a>>, Box<Expression<'a>>),
    Mult(Box<Expression<'a>>, Box<Expression<'a>>),
    Div(Box<Expression<'a>>, Box<Expression<'a>>),
    LT(Box<Expression<'a>>, Box<Expression<'a>>),
    GT(Box<Expression<'a>>, Box<Expression<'a>>),
    Conditional(
        Box<Expression<'a>>,
        Vec<Statement<'a>>,
        Option<Vec<Statement<'a>>>,
    ),
    Brace(Vec<Statement<'a>>),
}

fn comment(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace0(input)?;
    delimited(tag("/*"), take_until("*/"), tag("*/"))(r).map(|(r, s)| (r, Statement::Comment(s)))
}

pub fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn ident_space(input: &str) -> IResult<&str, &str> {
    delimited(multispace0, identifier, multispace0)(input)
}

fn var_ref(input: &str) -> IResult<&str, Expression> {
    let (r, res) = ident_space(input)?;
    Ok((r, Expression::Variable(res)))
}

fn var_decl(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace1(tag("var")(multispace0(input)?.0)?.0)?;
    let (r, ident) = identifier(r)?;
    let (r, initializer) = opt(delimited(
        delimited(multispace0, tag("="), multispace0),
        full_expression,
        multispace0,
    ))(r)?;
    let (r, _) = char(';')(multispace0(r)?.0)?;
    Ok((r, Statement::VarDecl(ident, initializer)))
}

fn numeric_literal_expression(input: &str) -> IResult<&str, Expression> {
    let (r, val) = double(multispace0(input)?.0)?;
    Ok((multispace0(r)?.0, Expression::NumLiteral(val)))
}

// We parse any expr surrounded by parens, ignoring all whitespaces around those
fn parens(i: &str) -> IResult<&str, Expression> {
    delimited(
        multispace0,
        delimited(tag("("), conditional_expr, tag(")")),
        multispace0,
    )(i)
}

fn func_invoke(i: &str) -> IResult<&str, Expression> {
    let (r, ident) = delimited(multispace0, identifier, multispace0)(i)?;
    // println!("func_invoke ident: {}", ident);
    let (r, args) = delimited(
        multispace0,
        delimited(
            tag("("),
            many0(delimited(
                multispace0,
                expr,
                delimited(multispace0, opt(tag(",")), multispace0),
            )),
            tag(")"),
        ),
        multispace0,
    )(r)?;
    Ok((r, Expression::FnInvoke(ident, args)))
}

// We transform an double string into a Expression::NumLiteral
// on failure, we fallback to the parens parser defined above
fn factor(i: &str) -> IResult<&str, Expression> {
    alt((
        numeric_literal_expression,
        func_invoke,
        var_ref,
        parens,
        brace_expr,
    ))(i)
}

// We read an initial factor and for each time we find
// a * or / operator followed by another factor, we do
// the math by folding everything
fn term(i: &str) -> IResult<&str, Expression> {
    let (i, init) = factor(i)?;

    fold_many0(
        pair(alt((char('*'), char('/'))), factor),
        init,
        |acc, (op, val): (char, Expression)| {
            if op == '*' {
                Expression::Mult(Box::new(acc), Box::new(val))
            } else {
                Expression::Div(Box::new(acc), Box::new(val))
            }
        },
    )(i)
}

fn expr(i: &str) -> IResult<&str, Expression> {
    let (i, init) = term(i)?;

    fold_many0(
        pair(alt((char('+'), char('-'))), term),
        init,
        |acc, (op, val): (char, Expression)| {
            if op == '+' {
                Expression::Add(Box::new(acc), Box::new(val))
            } else {
                Expression::Sub(Box::new(acc), Box::new(val))
            }
        },
    )(i)
}

fn cmp(i: &str) -> IResult<&str, Expression> {
    let (i, lhs) = expr(i)?;

    let (i, (op, val)) = pair(alt((char('<'), char('>'))), expr)(i)?;
    Ok((
        i,
        if op == '<' {
            Expression::LT(Box::new(lhs), Box::new(val))
        } else {
            Expression::GT(Box::new(lhs), Box::new(val))
        },
    ))
}

fn conditional(i: &str) -> IResult<&str, Expression> {
    let (r, _) = delimited(multispace0, tag("if"), multispace0)(i)?;
    let (r, cond) = cmp_expr(r)?;
    let (r, true_branch) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    let (r, false_branch) = opt(preceded(
        delimited(multispace0, tag("else"), multispace0),
        alt((
            delimited(
                delimited(multispace0, tag("{"), multispace0),
                source,
                delimited(multispace0, tag("}"), multispace0),
            ),
            map_res(
                conditional,
                |v| -> Result<Vec<Statement>, nom::error::Error<&str>> {
                    Ok(vec![Statement::Expression(v)])
                },
            ),
        )),
    ))(r)?;
    Ok((
        r,
        Expression::Conditional(Box::new(cond), true_branch, false_branch),
    ))
}

fn var_assign(input: &str) -> IResult<&str, Expression> {
    let (r, res) = tuple((ident_space, char('='), cmp_expr))(input)?;
    Ok((r, Expression::VarAssign(res.0, Box::new(res.2))))
}

fn cmp_expr(i: &str) -> IResult<&str, Expression> {
    alt((cmp, expr))(i)
}

fn assign_expr(i: &str) -> IResult<&str, Expression> {
    alt((var_assign, cmp_expr))(i)
}

fn conditional_expr(i: &str) -> IResult<&str, Expression> {
    alt((conditional, assign_expr))(i)
}

fn brace_expr(input: &str) -> IResult<&str, Expression> {
    let (r, v) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(input)?;
    Ok((r, Expression::Brace(v)))
}

fn full_expression(input: &str) -> IResult<&str, Expression> {
    conditional_expr(input)
}

fn expression_statement(input: &str) -> IResult<&str, Statement> {
    let (r, val) = full_expression(input)?;
    Ok((r, Statement::Expression(val)))
}

fn func_decl(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace1(tag("fn")(multispace0(input)?.0)?.0)?;
    let (r, ident) = identifier(r)?;
    let (r, args) = delimited(
        multispace0,
        delimited(
            tag("("),
            many0(delimited(
                multispace0,
                identifier,
                delimited(multispace0, opt(tag(",")), multispace0),
            )),
            tag(")"),
        ),
        multispace0,
    )(r)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::FnDecl(ident, args, stmts)))
}

fn loop_stmt(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace0(tag("loop")(multispace0(input)?.0)?.0)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::Loop(stmts)))
}

fn while_stmt(input: &str) -> IResult<&str, Statement> {
    let (r, _) = multispace0(tag("while")(multispace0(input)?.0)?.0)?;
    let (r, cond) = cmp_expr(r)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::While(cond, stmts)))
}

fn for_stmt(input: &str) -> IResult<&str, Statement> {
    let (r, _) = delimited(multispace0, tag("for"), multispace1)(input)?;
    let (r, iter) = identifier(r)?;
    let (r, _) = delimited(multispace0, tag("in"), multispace0)(r)?;
    let (r, from) = expr(r)?;
    let (r, _) = delimited(multispace0, tag(".."), multispace0)(r)?;
    let (r, to) = expr(r)?;
    let (r, stmts) = delimited(
        delimited(multispace0, tag("{"), multispace0),
        source,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    Ok((r, Statement::For(iter, from, to, stmts)))
}

fn break_stmt(input: &str) -> IResult<&str, Statement> {
    let (r, _) = delimited(multispace0, tag("break"), multispace0)(input)?;
    Ok((r, Statement::Break))
}

fn general_statement<'a>(last: bool) -> impl Fn(&'a str) -> IResult<&'a str, Statement> {
    let terminator = move |i| -> IResult<&str, ()> {
        let mut semicolon = pair(tag(";"), multispace0);
        if last {
            Ok((opt(semicolon)(i)?.0, ()))
        } else {
            Ok((semicolon(i)?.0, ()))
        }
    };
    move |input: &str| {
        alt((
            var_decl,
            func_decl,
            loop_stmt,
            while_stmt,
            for_stmt,
            terminated(break_stmt, terminator),
            terminated(expression_statement, terminator),
            comment,
        ))(input)
    }
}

fn last_statement(input: &str) -> IResult<&str, Statement> {
    general_statement(true)(input)
}

fn statement(input: &str) -> IResult<&str, Statement> {
    general_statement(false)(input)
}

fn source(input: &str) -> IResult<&str, Vec<Statement>> {
    let (r, mut v) = many0(statement)(input)?;
    let (r, last) = opt(last_statement)(r)?;
    if let Some(last) = last {
        v.push(last);
    }
    Ok((r, v))
}

macro_rules! unwrap_run {
    ($e:expr) => {
        match $e {
            RunResult::Yield(v) => v,
            RunResult::Break => return RunResult::Break,
        }
    };
}

fn eval<'a, 'b>(e: &'b Expression<'a>, ctx: &mut EvalContext<'a, 'b, '_, '_>) -> RunResult {
    match e {
        Expression::NumLiteral(val) => RunResult::Yield(*val),
        Expression::Variable(str) => RunResult::Yield(
            ctx.get_var(str)
                .expect(&format!("Variable {} not found in scope", str)),
        ),
        Expression::VarAssign(str, rhs) => {
            let value = unwrap_run!(eval(rhs, ctx));
            let mut search_ctx: Option<&EvalContext> = Some(ctx);
            while let Some(c) = search_ctx {
                if let None = c.variables.borrow().get(str) {
                    search_ctx = c.super_context;
                    continue;
                }
                c.variables.borrow_mut().insert(str, value);
                break;
            }
            if search_ctx.is_none() {
                panic!(format!("Variable \"{}\" was not declared!", str));
            }
            RunResult::Yield(value)
        }
        Expression::FnInvoke(str, args) => {
            let args = args.iter().map(|v| eval(v, ctx)).collect::<Vec<_>>();
            let mut subctx = EvalContext::push_stack(ctx);
            let func = ctx
                .get_fn(*str)
                .expect(&format!("function {} is not defined.", str));
            match func {
                FuncDef::Code(func) => {
                    for (k, v) in func.args.iter().zip(&args) {
                        subctx.variables.borrow_mut().insert(k, unwrap_run!(*v));
                    }
                    let run_result = run(func.stmts, &mut subctx).unwrap();
                    match run_result {
                        RunResult::Yield(v) => RunResult::Yield(v),
                        RunResult::Break => panic!("break in function toplevel"),
                    }
                }
                FuncDef::Native(native) => RunResult::Yield(native(
                    &args
                        .iter()
                        .map(|e| if let RunResult::Yield(v) = e { *v } else { 0. })
                        .collect::<Vec<_>>(),
                )),
            }
        }
        Expression::Add(lhs, rhs) => {
            RunResult::Yield(unwrap_run!(eval(lhs, ctx)) + unwrap_run!(eval(rhs, ctx)))
        }
        Expression::Sub(lhs, rhs) => {
            RunResult::Yield(unwrap_run!(eval(lhs, ctx)) - unwrap_run!(eval(rhs, ctx)))
        }
        Expression::Mult(lhs, rhs) => {
            RunResult::Yield(unwrap_run!(eval(lhs, ctx)) * unwrap_run!(eval(rhs, ctx)))
        }
        Expression::Div(lhs, rhs) => {
            RunResult::Yield(unwrap_run!(eval(lhs, ctx)) / unwrap_run!(eval(rhs, ctx)))
        }
        Expression::LT(lhs, rhs) => {
            if unwrap_run!(eval(lhs, ctx)) < unwrap_run!(eval(rhs, ctx)) {
                RunResult::Yield(1.)
            } else {
                RunResult::Yield(0.)
            }
        }
        Expression::GT(lhs, rhs) => {
            if unwrap_run!(eval(lhs, ctx)) > unwrap_run!(eval(rhs, ctx)) {
                RunResult::Yield(1.)
            } else {
                RunResult::Yield(0.)
            }
        }
        Expression::Conditional(cond, true_branch, false_branch) => {
            if unwrap_run!(eval(cond, ctx)) != 0. {
                run(true_branch, ctx).unwrap()
            } else if let Some(ast) = false_branch {
                run(ast, ctx).unwrap()
            } else {
                RunResult::Yield(0.)
            }
        }
        Expression::Brace(stmts) => {
            let mut subctx = EvalContext::push_stack(ctx);
            run(stmts, &mut subctx).unwrap()
        }
    }
}

fn s_print(vals: &[f64]) -> f64 {
    if let [val, ..] = vals {
        println!("print: {}", val);
    }
    0.
}

#[derive(Clone)]
struct FuncCode<'src, 'ast> {
    args: &'ast Vec<&'src str>,
    stmts: &'ast Vec<Statement<'src>>,
}

#[derive(Clone)]
enum FuncDef<'src, 'ast, 'native> {
    Code(FuncCode<'src, 'ast>),
    Native(&'native dyn Fn(&[f64]) -> f64),
}

/// A context stat for evaluating a script.
///
/// It has 2 lifetime arguments, one for the source code ('src) and the other for
/// the AST ('ast), because usually AST is created after the source.
#[derive(Clone)]
struct EvalContext<'src, 'ast, 'native, 'ctx> {
    /// RefCell to allow mutation in super context
    variables: RefCell<HashMap<&'src str, f64>>,
    /// Function names are owned strings because it can be either from source or native.
    functions: HashMap<String, FuncDef<'src, 'ast, 'native>>,
    super_context: Option<&'ctx EvalContext<'src, 'ast, 'native, 'ctx>>,
}

impl<'src, 'ast, 'native, 'ctx> EvalContext<'src, 'ast, 'native, 'ctx> {
    fn new() -> Self {
        let mut functions = HashMap::new();
        functions.insert("print".to_string(), FuncDef::Native(&s_print));
        Self {
            variables: RefCell::new(HashMap::new()),
            functions,
            super_context: None,
        }
    }

    fn push_stack(super_ctx: &'ctx Self) -> Self {
        Self {
            variables: RefCell::new(HashMap::new()),
            functions: HashMap::new(),
            super_context: Some(super_ctx),
        }
    }

    fn get_var(&self, name: &str) -> Option<f64> {
        if let Some(val) = self.variables.borrow_mut().get(name) {
            Some(*val)
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_var(name)
        } else {
            None
        }
    }

    fn get_fn(&self, name: &str) -> Option<&FuncDef<'src, 'ast, 'native>> {
        if let Some(val) = self.functions.get(name) {
            Some(val)
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_fn(name)
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq)]
enum RunResult {
    Yield(f64),
    Break,
}

macro_rules! unwrap_break {
    ($e:expr) => {
        match $e {
            RunResult::Yield(v) => v,
            RunResult::Break => break,
        }
    };
}

fn run<'src, 'ast>(
    stmts: &'ast Vec<Statement<'src>>,
    ctx: &mut EvalContext<'src, 'ast, '_, '_>,
) -> Result<RunResult, ()> {
    let mut res = RunResult::Yield(0.);
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var, initializer) => {
                let init_val = if let Some(init_expr) = initializer {
                    unwrap_break!(eval(init_expr, ctx))
                } else {
                    0.
                };
                ctx.variables.borrow_mut().insert(*var, init_val);
            }
            Statement::FnDecl(var, args, stmts) => {
                ctx.functions
                    .insert(var.to_string(), FuncDef::Code(FuncCode { args, stmts }));
            }
            Statement::Expression(e) => {
                res = eval(&e, ctx);
                if let RunResult::Break = res {
                    return Ok(res);
                }
                // println!("Expression evaluates to: {:?}", res);
            }
            Statement::Loop(e) => loop {
                res = match run(e, ctx)? {
                    RunResult::Yield(v) => RunResult::Yield(v),
                    RunResult::Break => break,
                };
            },
            Statement::While(cond, e) => loop {
                match eval(cond, ctx) {
                    RunResult::Yield(v) => {
                        if v == 0. {
                            break;
                        }
                    }
                    RunResult::Break => break,
                }
                res = match run(e, ctx)? {
                    RunResult::Yield(v) => RunResult::Yield(v),
                    RunResult::Break => break,
                };
            },
            Statement::For(iter, from, to, e) => {
                let from_res = unwrap_break!(eval(from, ctx)) as isize;
                let to_res = unwrap_break!(eval(to, ctx)) as isize;
                for i in from_res..to_res {
                    ctx.variables.borrow_mut().insert(iter, i as f64);
                    res = match run(e, ctx)? {
                        RunResult::Yield(v) => RunResult::Yield(v),
                        RunResult::Break => break,
                    };
                }
            }
            Statement::Break => {
                return Ok(RunResult::Break);
            }
            _ => {}
        }
    }
    Ok(res)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut contents = String::new();
    let code = if 1 < args.len() {
        if let Ok(mut file) = File::open(&args[1]) {
            file.read_to_string(&mut contents)?;
            &contents
        } else {
            &args[1]
        }
    } else {
        r"var x;
  /* This is a block comment. */
  var y;
  123;
  123 + 456;
  "
    };
    if let Ok(result) = source(code) {
        println!("Match: {:?}", result.1);
        run(&result.1, &mut EvalContext::new()).expect("Error in run()");
    } else {
        println!("failed");
    }
    Ok(())
}

mod test;
