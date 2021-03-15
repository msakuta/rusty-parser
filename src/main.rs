use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1},
    combinator::{opt, recognize},
    multi::{fold_many0, many0},
    number::complete::double,
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};
use std::fs::File;
use std::io::prelude::*;
use std::{collections::HashMap, env};

#[derive(Debug, PartialEq, Clone)]
enum Statement<'a> {
    Comment(&'a str),
    VarDecl(&'a str),
    FnDecl(&'a str, Vec<&'a str>, Vec<Statement<'a>>),
    Expression(Expression<'a>),
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
        Box<Expression<'a>>,
        Option<Box<Expression<'a>>>,
    ),
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
    let (r, _) = char(';')(multispace0(r)?.0)?;
    Ok((r, Statement::VarDecl(ident)))
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
    println!("func_invoke ident: {}", ident);
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
    alt((numeric_literal_expression, func_invoke, var_ref, parens))(i)
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
        conditional_expr,
        delimited(multispace0, tag("}"), multispace0),
    )(r)?;
    let (r, false_branch) = opt(preceded(
        delimited(multispace0, tag("else"), multispace0),
        alt((
            delimited(
                delimited(multispace0, tag("{"), multispace0),
                conditional_expr,
                delimited(multispace0, tag("}"), multispace0),
            ),
            conditional,
        )),
    ))(r)?;
    Ok((
        r,
        Expression::Conditional(
            Box::new(cond),
            Box::new(true_branch),
            false_branch.map(|e| Box::new(e)),
        ),
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

fn expression_statement(input: &str) -> IResult<&str, Statement> {
    let (r, val) = conditional_expr(input)?;
    Ok((char(';')(r)?.0, Statement::Expression(val)))
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

fn source(input: &str) -> IResult<&str, Vec<Statement>> {
    many0(alt((var_decl, func_decl, expression_statement, comment)))(input)
}

fn eval<'a, 'b>(e: &'b Expression<'a>, ctx: &mut EvalContext<'a, 'b>) -> f64 {
    match e {
        Expression::NumLiteral(val) => *val,
        Expression::Variable(str) => *ctx
            .variables
            .get(str)
            .expect(&format!("Variable {} not found in scope", str)),
        Expression::VarAssign(str, rhs) => {
            let value = eval(rhs, ctx);
            if let None = ctx.variables.insert(str, value) {
                panic!("Variable was not declared!");
            }
            value
        }
        Expression::FnInvoke(str, args) => {
            let args = args.iter().map(|v| eval(v, ctx)).collect::<Vec<_>>();
            let mut subctx = ctx.clone();
            let func = ctx.functions.get(str).unwrap();
            for (k, v) in func.args.iter().zip(args) {
                subctx.variables.insert(k, v);
            }
            run(func.stmts, subctx).unwrap()
        }
        Expression::Add(lhs, rhs) => eval(lhs, ctx) + eval(rhs, ctx),
        Expression::Sub(lhs, rhs) => eval(lhs, ctx) - eval(rhs, ctx),
        Expression::Mult(lhs, rhs) => eval(lhs, ctx) * eval(rhs, ctx),
        Expression::Div(lhs, rhs) => eval(lhs, ctx) / eval(rhs, ctx),
        Expression::LT(lhs, rhs) => {
            if eval(lhs, ctx) < eval(rhs, ctx) {
                1.
            } else {
                0.
            }
        }
        Expression::GT(lhs, rhs) => {
            if eval(lhs, ctx) > eval(rhs, ctx) {
                1.
            } else {
                0.
            }
        }
        Expression::Conditional(cond, true_branch, false_branch) => {
            if eval(cond, ctx) != 0. {
                eval(true_branch, ctx)
            } else if let Some(ast) = false_branch {
                eval(ast, ctx)
            } else {
                0.
            }
        }
    }
}

#[derive(Clone, Debug)]
struct FuncDef<'src, 'ast> {
    args: &'ast Vec<&'src str>,
    stmts: &'ast Vec<Statement<'src>>,
}

/// A context stat for evaluating a script.
///
/// It has 2 lifetime arguments, one for the source code ('src) and the other for
/// the AST ('ast), because usually AST is created after the source.
#[derive(Clone, Debug)]
struct EvalContext<'src, 'ast> {
    variables: HashMap<&'src str, f64>,
    functions: HashMap<&'src str, FuncDef<'src, 'ast>>,
}

impl<'a, 'b> EvalContext<'a, 'b> {
    fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
        }
    }
}

fn run<'src, 'ast>(
    stmts: &'ast Vec<Statement<'src>>,
    mut ctx: EvalContext<'src, 'ast>,
) -> Result<f64, ()> {
    let mut res = 0.;
    for stmt in stmts {
        match stmt {
            Statement::VarDecl(var) => {
                ctx.variables.insert(*var, 0.);
            }
            Statement::FnDecl(var, args, stmts) => {
                ctx.functions.insert(var, FuncDef { args, stmts });
            }
            Statement::Expression(e) => {
                res = eval(&e, &mut ctx);
                println!("Expression evaluates to: {}", res);
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
        run(&result.1, EvalContext::new()).expect("Error in run()");
    } else {
        println!("failed");
    }
    Ok(())
}

mod test;
