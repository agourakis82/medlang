//! Parser extensions for Phase V1 features (effects, epistemic, refinements).
//!
//! This module extends the main parser with support for:
//! - Effect annotations: `with Pure`, `with Prob | IO`
//! - Epistemic types: `Knowledge<ConcMass>(0.8)`, `Knowledge<f64>`
//! - Refinement constraints: `where CL > 0.0`, `where age in 18.0..120.0`

use crate::ast::phase_v1::*;
use crate::effects::Effect;
use crate::lexer::Token;
use nom::{
    branch::alt,
    combinator::{map, opt},
    multi::separated_list1,
    sequence::delimited,
    Err, IResult,
};

/// Parser input is a slice of tokens
pub type TokenSlice<'a> = &'a [(Token, usize, usize)];

// ============================================================================
// Effect Annotations
// ============================================================================

/// Parse an effect annotation: `with Pure` or `with Prob | IO | GPU`
pub fn effect_annotation(input: TokenSlice) -> IResult<TokenSlice, EffectAnnotationAst> {
    let (input, _) = token(Token::With)(input)?;
    let (input, effects) = separated_list1(token(Token::Pipe), effect_name)(input)?;
    Ok((input, EffectAnnotationAst { effects }))
}

/// Parse a single effect name: Pure | Prob | IO | GPU
fn effect_name(input: TokenSlice) -> IResult<TokenSlice, Effect> {
    alt((
        map(ident_literal("Pure"), |_| Effect::Pure),
        map(ident_literal("Prob"), |_| Effect::Prob),
        map(ident_literal("IO"), |_| Effect::IO),
        map(ident_literal("GPU"), |_| Effect::GPU),
    ))(input)
}

// ============================================================================
// Epistemic Types
// ============================================================================

/// Parse an epistemic type: `Knowledge<ConcMass>` or `Knowledge<ConcMass>(0.8)`
pub fn epistemic_type(input: TokenSlice) -> IResult<TokenSlice, EpistemicTypeAst> {
    let (input, _) = token(Token::Knowledge)(input)?;
    let (input, _) = token(Token::Lt)(input)?;
    let (input, inner_type) = type_name(input)?;
    let (input, _) = token(Token::Gt)(input)?;
    let (input, min_confidence) = opt(delimited(
        token(Token::LParen),
        float_literal,
        token(Token::RParen),
    ))(input)?;

    Ok((
        input,
        EpistemicTypeAst {
            inner_type,
            min_confidence,
        },
    ))
}

// ============================================================================
// Refinement Constraints
// ============================================================================

/// Parse a refinement constraint: `where CL > 0.0` or `where age in 18.0..120.0`
pub fn refinement_constraint(input: TokenSlice) -> IResult<TokenSlice, RefinementConstraintAst> {
    let (input, _) = token(Token::Where)(input)?;
    let (input, constraint) = constraint_expr(input)?;
    Ok((input, RefinementConstraintAst { constraint }))
}

/// Parse a constraint expression
fn constraint_expr(input: TokenSlice) -> IResult<TokenSlice, ConstraintExpr> {
    alt((binary_constraint, range_constraint, comparison_constraint))(input)
}

/// Parse range constraint: `age in 18.0..120.0`
fn range_constraint(input: TokenSlice) -> IResult<TokenSlice, ConstraintExpr> {
    let (input, var) = identifier(input)?;
    let (input, _) = token(Token::In)(input)?;
    let (input, lower) = constraint_literal(input)?;
    let (input, _) = token(Token::Dot)(input)?;
    let (input, _) = token(Token::Dot)(input)?;
    let (input, upper) = constraint_literal(input)?;

    Ok((input, ConstraintExpr::Range { var, lower, upper }))
}

/// Parse comparison constraint: `CL > 0.0`
fn comparison_constraint(input: TokenSlice) -> IResult<TokenSlice, ConstraintExpr> {
    let (input, var) = identifier(input)?;
    let (input, op) = comparison_op(input)?;
    let (input, value) = constraint_literal(input)?;

    Ok((input, ConstraintExpr::Comparison { var, op, value }))
}

/// Parse binary constraint: `CL > 0.0 && V > 0.0` (two constraints combined)
fn binary_constraint(input: TokenSlice) -> IResult<TokenSlice, ConstraintExpr> {
    // Try to parse left comparison
    let (input, left) = comparison_constraint(input)?;
    let (input, op) = logical_op(input)?;
    let (input, right) = comparison_constraint(input)?;

    Ok((
        input,
        ConstraintExpr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        },
    ))
}

/// Parse comparison operator: `>`, `<`, `>=`, `<=`, `==`, `!=`
fn comparison_op(input: TokenSlice) -> IResult<TokenSlice, ComparisonOp> {
    alt((
        map(token(Token::Gt), |_| ComparisonOp::Gt),
        map(token(Token::Lt), |_| ComparisonOp::Lt),
        map(token(Token::Gte), |_| ComparisonOp::Ge),
        map(token(Token::Lte), |_| ComparisonOp::Le),
        map(token(Token::EqEq), |_| ComparisonOp::Eq),
        map(token(Token::NotEq), |_| ComparisonOp::Ne),
    ))(input)
}

/// Parse logical operator: `&&`, `||`
fn logical_op(input: TokenSlice) -> IResult<TokenSlice, LogicalOp> {
    alt((
        map(token(Token::And), |_| LogicalOp::And),
        // Note: Token::Or doesn't exist yet, would need "||" token
        // For now, just support And
    ))(input)
}

// ============================================================================
// Helper Parsers
// ============================================================================

/// Parse a specific token
fn token(tok: Token) -> impl Fn(TokenSlice) -> IResult<TokenSlice, &Token> {
    move |input: TokenSlice| {
        if let Some(first) = input.first() {
            if first.0 == tok {
                Ok((&input[1..], &first.0))
            } else {
                Err(Err::Error(nom::error::Error::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )))
            }
        } else {
            Err(Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Eof,
            )))
        }
    }
}

/// Parse an identifier (Token::Ident)
fn identifier(input: TokenSlice) -> IResult<TokenSlice, String> {
    if let Some(first) = input.first() {
        if let Token::Ident(name) = &first.0 {
            Ok((&input[1..], name.clone()))
        } else {
            Err(Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Tag,
            )))
        }
    } else {
        Err(Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Eof,
        )))
    }
}

/// Parse a float literal (Token::Float) and return raw f64
fn float_literal(input: TokenSlice) -> IResult<TokenSlice, f64> {
    if let Some(first) = input.first() {
        if let Token::Float(val) = first.0 {
            Ok((&input[1..], val))
        } else {
            Err(Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Tag,
            )))
        }
    } else {
        Err(Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Eof,
        )))
    }
}

/// Parse a constraint literal (float, int, or unit value)
fn constraint_literal(input: TokenSlice) -> IResult<TokenSlice, ConstraintLiteral> {
    if let Some(first) = input.first() {
        match &first.0 {
            Token::Float(val) => Ok((&input[1..], ConstraintLiteral::Float(*val))),
            Token::UnitLiteral(ul) => Ok((
                &input[1..],
                ConstraintLiteral::UnitValue(ul.value, ul.unit.clone()),
            )),
            _ => Err(Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Tag,
            ))),
        }
    } else {
        Err(Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Eof,
        )))
    }
}

/// Parse a specific identifier literal (e.g., "Pure", "Prob")
fn ident_literal(expected: &'static str) -> impl Fn(TokenSlice) -> IResult<TokenSlice, String> {
    move |input: TokenSlice| {
        if let Some(first) = input.first() {
            if let Token::Ident(name) = &first.0 {
                if name == expected {
                    Ok((&input[1..], name.clone()))
                } else {
                    Err(Err::Error(nom::error::Error::new(
                        input,
                        nom::error::ErrorKind::Tag,
                    )))
                }
            } else {
                Err(Err::Error(nom::error::Error::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )))
            }
        } else {
            Err(Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Eof,
            )))
        }
    }
}

/// Parse a type name (identifier for now, could be extended for complex types)
fn type_name(input: TokenSlice) -> IResult<TokenSlice, String> {
    // For now, support simple type names like ConcMass, Volume, f64
    // Could be extended to parse Token::Mass, Token::Volume, etc.
    alt((
        identifier,
        map(token(Token::Mass), |_| "Mass".to_string()),
        map(token(Token::Volume), |_| "Volume".to_string()),
        map(token(Token::Time), |_| "Time".to_string()),
        map(token(Token::ConcMass), |_| "ConcMass".to_string()),
        map(token(Token::Clearance), |_| "Clearance".to_string()),
        map(token(Token::F64), |_| "f64".to_string()),
    ))(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    #[test]
    fn test_parse_effect_pure() {
        let source = "with Pure";
        let tokens = tokenize(source).unwrap();
        let result = effect_annotation(&tokens);

        assert!(result.is_ok());
        let (_, ann) = result.unwrap();
        assert_eq!(ann.effects.len(), 1);
        assert_eq!(ann.effects[0], Effect::Pure);
    }

    #[test]
    fn test_parse_epistemic_type_simple() {
        let source = "Knowledge<ConcMass>";
        let tokens = tokenize(source).unwrap();
        let result = epistemic_type(&tokens);

        assert!(result.is_ok());
        let (_, etype) = result.unwrap();
        assert_eq!(etype.inner_type, "ConcMass");
        assert_eq!(etype.min_confidence, None);
    }

    #[test]
    fn test_parse_epistemic_type_with_confidence() {
        let source = "Knowledge<ConcMass>(0.8)";
        let tokens = tokenize(source).unwrap();
        let result = epistemic_type(&tokens);

        assert!(result.is_ok());
        let (_, etype) = result.unwrap();
        assert_eq!(etype.inner_type, "ConcMass");
        assert_eq!(etype.min_confidence, Some(0.8));
    }

    #[test]
    fn test_parse_refinement_comparison() {
        let source = "where CL > 0.0";
        let tokens = tokenize(source).unwrap();
        let result = refinement_constraint(&tokens);

        assert!(result.is_ok());
        let (_, constraint) = result.unwrap();
        match constraint.constraint {
            ConstraintExpr::Comparison { var, op, value } => {
                assert_eq!(var, "CL");
                assert_eq!(op, ComparisonOp::Gt);
                assert!(matches!(value, ConstraintLiteral::Float(v) if v == 0.0));
            }
            _ => panic!("Expected comparison constraint"),
        }
    }

    #[test]
    fn test_parse_refinement_range() {
        let source = "where age in 18.0..120.0";
        let tokens = tokenize(source).unwrap();
        let result = refinement_constraint(&tokens);

        assert!(result.is_ok());
        let (_, constraint) = result.unwrap();
        match constraint.constraint {
            ConstraintExpr::Range { var, lower, upper } => {
                assert_eq!(var, "age");
                assert!(matches!(lower, ConstraintLiteral::Float(v) if v == 18.0));
                assert!(matches!(upper, ConstraintLiteral::Float(v) if v == 120.0));
            }
            _ => panic!("Expected range constraint"),
        }
    }
}
