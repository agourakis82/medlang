//! Lexer (tokenizer) for MedLang using logos.
//!
//! Converts source text into a stream of tokens with position information.

use logos::Logos;
use std::fmt;

/// Unit literal value (e.g., 100.0_mg)
#[derive(Debug, Clone, PartialEq)]
pub struct UnitLiteralValue {
    pub value: f64,
    pub unit: String,
}

/// Token types for MedLang
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n]+")] // Skip whitespace
#[logos(skip r"//[^\n]*")] // Skip single-line comments
#[logos(skip r"/\*([^*]|\*[^/])*\*/")] // Skip multi-line comments
pub enum Token {
    // Keywords
    #[token("model")]
    Model,

    #[token("population")]
    Population,

    #[token("measure")]
    Measure,

    #[token("timeline")]
    Timeline,

    #[token("cohort")]
    Cohort,

    // Week 8: L₂ Protocol DSL tokens
    #[token("protocol")]
    Protocol,

    #[token("arms")]
    Arms,

    #[token("visits")]
    Visits,

    #[token("inclusion")]
    Inclusion,

    #[token("endpoints")]
    Endpoints,

    #[token("between")]
    Between,

    // Week 10: Additional protocol tokens
    #[token("label")]
    Label,

    #[token("type")]
    Type,

    #[token("observable")]
    Observable,

    #[token("shrink_frac")]
    ShrinkFrac,

    #[token("progression_frac")]
    ProgressionFrac,

    #[token("ref_baseline")]
    RefBaseline,

    #[token("window")]
    Window,

    #[token("age")]
    Age,

    #[token("ECOG")]
    ECOG,

    #[token("in")]
    In,

    #[token("and")]
    And,

    #[token("baseline_tumour_volume")]
    BaselineTumourVolume,

    // Week 14: Decision layer tokens
    #[token("decisions")]
    Decisions,

    #[token("endpoint")]
    Endpoint,

    #[token("compare")]
    Compare,

    #[token("margin")]
    Margin,

    #[token("prob_threshold")]
    ProbThreshold,

    #[token("true")]
    True,

    #[token("false")]
    False,

    #[token("state")]
    State,

    #[token("param")]
    Param,

    #[token("obs")]
    Obs,

    #[token("rand")]
    Rand,

    #[token("input")]
    Input,

    #[token("submodel")]
    Submodel,

    #[token("connect")]
    Connect,

    #[token("pred")]
    Pred,

    #[token("at")]
    At,

    #[token("dose")]
    Dose,

    #[token("observe")]
    Observe,

    #[token("to")]
    To,

    #[token("use_measure")]
    UseMeasure,

    #[token("bind_params")]
    BindParams,

    #[token("for")]
    For,

    #[token("let")]
    Let,

    #[token("amount")]
    Amount,

    #[token("data_file")]
    DataFile,

    #[token("log_likelihood")]
    LogLikelihood,

    // Unit type keywords
    #[token("Mass")]
    Mass,

    #[token("Volume")]
    Volume,

    #[token("Time")]
    Time,

    #[token("DoseMass")]
    DoseMass,

    #[token("ConcMass")]
    ConcMass,

    #[token("Clearance")]
    Clearance,

    #[token("RateConst")]
    RateConst,

    #[token("TumourVolume")]
    TumourVolume,

    #[token("Quantity")]
    Quantity,

    // Type keywords
    #[token("f64")]
    F64,

    // Distribution names
    #[token("Normal")]
    Normal,

    #[token("LogNormal")]
    LogNormal,

    #[token("Uniform")]
    Uniform,

    // Operators and punctuation
    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    #[token("^")]
    Caret,

    #[token("=")]
    Eq,

    #[token("==")]
    EqEq,

    #[token("!=")]
    NotEq,

    #[token("<")]
    Lt,

    #[token(">")]
    Gt,

    #[token(">=")]
    Gte,

    #[token("<=")]
    Lte,

    #[token("~")]
    Tilde,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    #[token(",")]
    Comma,

    #[token(";")]
    Semicolon,

    #[token(":")]
    Colon,

    #[token(".")]
    Dot,

    // Special ODE syntax: d<name>/dt
    #[regex(r"d[A-Za-z_][A-Za-z0-9_]*/dt", |lex| {
        let s = lex.slice();
        // Extract the variable name between 'd' and '/dt'
        s[1..s.len()-3].to_string()
    })]
    ODEDeriv(String),

    // Identifiers
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Numeric literals
    #[regex(r"[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),

    // Unit literals: number_unit (e.g., 100.0_mg, 1.5_h)
    #[regex(r"[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?_[A-Za-z][A-Za-z0-9/]*", |lex| {
        let s = lex.slice();
        let parts: Vec<&str> = s.split('_').collect();
        if parts.len() == 2 {
            if let Ok(value) = parts[0].parse::<f64>() {
                return Some(UnitLiteralValue { value, unit: parts[1].to_string() });
            }
        }
        None
    })]
    UnitLiteral(UnitLiteralValue),

    // String literals
    #[regex(r#""([^"\\]|\\["\\bnfrt])*""#, |lex| {
        let s = lex.slice();
        // Remove surrounding quotes
        s[1..s.len()-1].to_string()
    })]
    String(String),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Model => write!(f, "model"),
            Token::Population => write!(f, "population"),
            Token::Measure => write!(f, "measure"),
            Token::Timeline => write!(f, "timeline"),
            Token::Cohort => write!(f, "cohort"),
            Token::Protocol => write!(f, "protocol"),
            Token::Arms => write!(f, "arms"),
            Token::Visits => write!(f, "visits"),
            Token::Inclusion => write!(f, "inclusion"),
            Token::Endpoints => write!(f, "endpoints"),
            Token::Between => write!(f, "between"),
            Token::Label => write!(f, "label"),
            Token::Type => write!(f, "type"),
            Token::Observable => write!(f, "observable"),
            Token::ShrinkFrac => write!(f, "shrink_frac"),
            Token::ProgressionFrac => write!(f, "progression_frac"),
            Token::RefBaseline => write!(f, "ref_baseline"),
            Token::Window => write!(f, "window"),
            Token::Age => write!(f, "age"),
            Token::ECOG => write!(f, "ECOG"),
            Token::In => write!(f, "in"),
            Token::And => write!(f, "and"),
            Token::BaselineTumourVolume => write!(f, "baseline_tumour_volume"),
            Token::Decisions => write!(f, "decisions"),
            Token::Endpoint => write!(f, "endpoint"),
            Token::Compare => write!(f, "compare"),
            Token::Margin => write!(f, "margin"),
            Token::ProbThreshold => write!(f, "prob_threshold"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::State => write!(f, "state"),
            Token::Param => write!(f, "param"),
            Token::Obs => write!(f, "obs"),
            Token::Rand => write!(f, "rand"),
            Token::Input => write!(f, "input"),
            Token::Submodel => write!(f, "submodel"),
            Token::Connect => write!(f, "connect"),
            Token::Pred => write!(f, "pred"),
            Token::At => write!(f, "at"),
            Token::Dose => write!(f, "dose"),
            Token::Observe => write!(f, "observe"),
            Token::To => write!(f, "to"),
            Token::UseMeasure => write!(f, "use_measure"),
            Token::BindParams => write!(f, "bind_params"),
            Token::For => write!(f, "for"),
            Token::Let => write!(f, "let"),
            Token::Amount => write!(f, "amount"),
            Token::DataFile => write!(f, "data_file"),
            Token::LogLikelihood => write!(f, "log_likelihood"),
            Token::Mass => write!(f, "Mass"),
            Token::Volume => write!(f, "Volume"),
            Token::Time => write!(f, "Time"),
            Token::DoseMass => write!(f, "DoseMass"),
            Token::ConcMass => write!(f, "ConcMass"),
            Token::Clearance => write!(f, "Clearance"),
            Token::RateConst => write!(f, "RateConst"),
            Token::TumourVolume => write!(f, "TumourVolume"),
            Token::Quantity => write!(f, "Quantity"),
            Token::F64 => write!(f, "f64"),
            Token::Normal => write!(f, "Normal"),
            Token::LogNormal => write!(f, "LogNormal"),
            Token::Uniform => write!(f, "Uniform"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Caret => write!(f, "^"),
            Token::Eq => write!(f, "="),
            Token::EqEq => write!(f, "=="),
            Token::NotEq => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::Gte => write!(f, ">="),
            Token::Lte => write!(f, "<="),
            Token::Tilde => write!(f, "~"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Comma => write!(f, ","),
            Token::Semicolon => write!(f, ";"),
            Token::Colon => write!(f, ":"),
            Token::Dot => write!(f, "."),
            Token::ODEDeriv(name) => write!(f, "d{}/dt", name),
            Token::Ident(s) => write!(f, "{}", s),
            Token::Float(n) => write!(f, "{}", n),
            Token::UnitLiteral(ul) => write!(f, "{}_{}", ul.value, ul.unit),
            Token::String(s) => write!(f, "\"{}\"", s),
        }
    }
}

/// Tokenize source code
pub fn tokenize(source: &str) -> Result<Vec<(Token, usize, usize)>, LexError> {
    let mut tokens = Vec::new();
    let mut lexer = Token::lexer(source);

    while let Some(result) = lexer.next() {
        match result {
            Ok(token) => {
                let span = lexer.span();
                tokens.push((token, span.start, span.end));
            }
            Err(_) => {
                let span = lexer.span();
                return Err(LexError {
                    position: span.start,
                    snippet: source[span.clone()].to_string(),
                });
            }
        }
    }

    Ok(tokens)
}

/// Lexical error
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub position: usize,
    pub snippet: String,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unexpected token at position {}: '{}'",
            self.position, self.snippet
        )
    }
}

impl std::error::Error for LexError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let source = "model population measure timeline cohort";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].0, Token::Model);
        assert_eq!(tokens[1].0, Token::Population);
        assert_eq!(tokens[2].0, Token::Measure);
        assert_eq!(tokens[3].0, Token::Timeline);
        assert_eq!(tokens[4].0, Token::Cohort);
    }

    #[test]
    fn test_identifiers() {
        let source = "CL_pop V_pop Ka_pop patient_WT";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 4);
        assert!(matches!(tokens[0].0, Token::Ident(ref s) if s == "CL_pop"));
        assert!(matches!(tokens[1].0, Token::Ident(ref s) if s == "V_pop"));
    }

    #[test]
    fn test_float_literals() {
        let source = "100.0 1.5 0.75 3.14e-2";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 4);
        assert!(matches!(tokens[0].0, Token::Float(v) if (v - 100.0).abs() < 1e-10));
        assert!(matches!(tokens[1].0, Token::Float(v) if (v - 1.5).abs() < 1e-10));
        assert!(matches!(tokens[2].0, Token::Float(v) if (v - 0.75).abs() < 1e-10));
        assert!(matches!(tokens[3].0, Token::Float(v) if (v - 0.0314).abs() < 1e-10));
    }

    #[test]
    fn test_unit_literals() {
        let source = "100.0_mg 70.0_kg 1.5_h 10.0_L/h";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 4);
        assert!(matches!(tokens[0].0, Token::UnitLiteral(ref ul)
                         if (ul.value - 100.0).abs() < 1e-10 && ul.unit == "mg"));
        assert!(matches!(tokens[1].0, Token::UnitLiteral(ref ul)
                         if (ul.value - 70.0).abs() < 1e-10 && ul.unit == "kg"));
        assert!(matches!(tokens[2].0, Token::UnitLiteral(ref ul)
                         if (ul.value - 1.5).abs() < 1e-10 && ul.unit == "h"));
        assert!(matches!(tokens[3].0, Token::UnitLiteral(ref ul)
                         if (ul.value - 10.0).abs() < 1e-10 && ul.unit == "L/h"));
    }

    #[test]
    fn test_ode_derivative() {
        let source = "dA_gut/dt dA_central/dt";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, Token::ODEDeriv(ref s) if s == "A_gut"));
        assert!(matches!(tokens[1].0, Token::ODEDeriv(ref s) if s == "A_central"));
    }

    #[test]
    fn test_operators() {
        let source = "+ - * / ^ = == != < > ~";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 11);
        assert_eq!(tokens[0].0, Token::Plus);
        assert_eq!(tokens[1].0, Token::Minus);
        assert_eq!(tokens[2].0, Token::Star);
        assert_eq!(tokens[3].0, Token::Slash);
        assert_eq!(tokens[4].0, Token::Caret);
        assert_eq!(tokens[5].0, Token::Eq);
        assert_eq!(tokens[6].0, Token::EqEq);
        assert_eq!(tokens[7].0, Token::NotEq);
        assert_eq!(tokens[8].0, Token::Lt);
        assert_eq!(tokens[9].0, Token::Gt);
        assert_eq!(tokens[10].0, Token::Tilde);
    }

    #[test]
    fn test_punctuation() {
        let source = "( ) { } [ ] , ; : .";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0].0, Token::LParen);
        assert_eq!(tokens[1].0, Token::RParen);
        assert_eq!(tokens[2].0, Token::LBrace);
        assert_eq!(tokens[3].0, Token::RBrace);
    }

    #[test]
    fn test_string_literals() {
        let source = r#""data/onecomp_synth.csv" "hello world""#;
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, Token::String(ref s) if s == "data/onecomp_synth.csv"));
        assert!(matches!(tokens[1].0, Token::String(ref s) if s == "hello world"));
    }

    #[test]
    fn test_comments_ignored() {
        let source = "model // this is a comment\npopulation /* multi\nline */ measure";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].0, Token::Model);
        assert_eq!(tokens[1].0, Token::Population);
        assert_eq!(tokens[2].0, Token::Measure);
    }

    #[test]
    fn test_simple_model() {
        let source = "model Test { state A : Mass }";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[0].0, Token::Model);
        assert!(matches!(tokens[1].0, Token::Ident(ref s) if s == "Test"));
        assert_eq!(tokens[2].0, Token::LBrace);
        assert_eq!(tokens[3].0, Token::State);
        assert!(matches!(tokens[4].0, Token::Ident(ref s) if s == "A"));
        assert_eq!(tokens[5].0, Token::Colon);
        assert_eq!(tokens[6].0, Token::Mass);
        assert_eq!(tokens[7].0, Token::RBrace);
    }

    #[test]
    fn test_ode_equation() {
        let source = "dA_gut/dt = -Ka * A_gut";
        let tokens = tokenize(source).unwrap();

        assert_eq!(tokens.len(), 6);
        assert!(matches!(tokens[0].0, Token::ODEDeriv(ref s) if s == "A_gut"));
        assert_eq!(tokens[1].0, Token::Eq);
        assert_eq!(tokens[2].0, Token::Minus);
        assert!(matches!(tokens[3].0, Token::Ident(ref s) if s == "Ka"));
        assert_eq!(tokens[4].0, Token::Star);
        assert!(matches!(tokens[5].0, Token::Ident(ref s) if s == "A_gut"));
    }

    #[test]
    fn test_dose_event() {
        let source = r#"at 0.0_h: dose { amount = 100.0_mg; to = Model.A_gut }"#;
        let tokens = tokenize(source).unwrap();

        // at 0.0_h : dose { amount = 100.0_mg ; to = Model . A_gut }
        assert!(tokens.len() >= 10);
        assert_eq!(tokens[0].0, Token::At);
        assert!(
            matches!(tokens[1].0, Token::UnitLiteral(ref ul) if (ul.value - 0.0).abs() < 1e-10)
        );
        assert_eq!(tokens[2].0, Token::Colon);
        assert_eq!(tokens[3].0, Token::Dose);
    }
}
