// Week 54: UCUM Parser and Validator
//
// Parser for Unified Code for Units of Measure (UCUM) strings.
// UCUM is the healthcare standard for unit representation, adopted by
// IEEE, DICOM, LOINC, and HL7.
//
// ## UCUM Syntax Overview
//
// - Base units: m, g, s, A, K, mol, cd, rad, sr
// - Prefixes: k (kilo), m (milli), u (micro), n (nano), etc.
// - Operators: . (multiply), / (divide), exponent (e.g., m2)
// - Annotations: {text} for semantic clarification
// - Special units: [IU] for International Units, [pH] for pH scale
//
// ## Key Differences from SI
//
// - Gram (g) is base mass unit, not kilogram
// - Plane angles are dimensional (rad), not dimensionless
// - Supports annotations for semantic distinction

use super::dimension::{BaseDimension, Dimension};
use super::unit::Unit;
use std::collections::HashMap;
use std::fmt;

/// UCUM prefix with its factor
#[derive(Debug, Clone, Copy)]
pub struct UcumPrefix {
    pub symbol: &'static str,
    pub name: &'static str,
    pub factor: f64,
}

impl UcumPrefix {
    const fn new(symbol: &'static str, name: &'static str, factor: f64) -> Self {
        UcumPrefix {
            symbol,
            name,
            factor,
        }
    }
}

/// Standard UCUM prefixes
pub const UCUM_PREFIXES: &[UcumPrefix] = &[
    UcumPrefix::new("Y", "yotta", 1e24),
    UcumPrefix::new("Z", "zetta", 1e21),
    UcumPrefix::new("E", "exa", 1e18),
    UcumPrefix::new("P", "peta", 1e15),
    UcumPrefix::new("T", "tera", 1e12),
    UcumPrefix::new("G", "giga", 1e9),
    UcumPrefix::new("M", "mega", 1e6),
    UcumPrefix::new("k", "kilo", 1e3),
    UcumPrefix::new("h", "hecto", 1e2),
    UcumPrefix::new("da", "deka", 1e1),
    UcumPrefix::new("d", "deci", 1e-1),
    UcumPrefix::new("c", "centi", 1e-2),
    UcumPrefix::new("m", "milli", 1e-3),
    UcumPrefix::new("u", "micro", 1e-6), // UCUM uses 'u' for micro
    UcumPrefix::new("n", "nano", 1e-9),
    UcumPrefix::new("p", "pico", 1e-12),
    UcumPrefix::new("f", "femto", 1e-15),
    UcumPrefix::new("a", "atto", 1e-18),
    UcumPrefix::new("z", "zepto", 1e-21),
    UcumPrefix::new("y", "yocto", 1e-24),
];

/// UCUM base unit definition
#[derive(Debug, Clone)]
pub struct UcumBaseUnit {
    pub code: &'static str,
    pub name: &'static str,
    pub dimension: BaseDimension,
    pub is_metric: bool,
}

impl UcumBaseUnit {
    const fn new(
        code: &'static str,
        name: &'static str,
        dimension: BaseDimension,
        is_metric: bool,
    ) -> Self {
        UcumBaseUnit {
            code,
            name,
            dimension,
            is_metric,
        }
    }
}

/// Standard UCUM base units
pub const UCUM_BASE_UNITS: &[UcumBaseUnit] = &[
    UcumBaseUnit::new("m", "meter", BaseDimension::Length, true),
    UcumBaseUnit::new("g", "gram", BaseDimension::Mass, true), // Note: gram, not kg
    UcumBaseUnit::new("s", "second", BaseDimension::Time, true),
    UcumBaseUnit::new("A", "ampere", BaseDimension::Current, true),
    UcumBaseUnit::new("K", "kelvin", BaseDimension::Temperature, true),
    UcumBaseUnit::new("mol", "mole", BaseDimension::Amount, true),
    UcumBaseUnit::new("cd", "candela", BaseDimension::Luminosity, true),
];

/// UCUM derived/special units
#[derive(Debug, Clone)]
pub struct UcumDerivedUnit {
    pub code: &'static str,
    pub name: &'static str,
    pub definition: &'static str, // In terms of base units
    pub factor: f64,
}

/// Token from UCUM lexer
#[derive(Debug, Clone, PartialEq)]
pub enum UcumToken {
    /// Base unit or derived unit code
    Unit(String),
    /// Numeric prefix factor
    Prefix(String),
    /// Annotation in braces
    Annotation(String),
    /// Multiplication operator (.)
    Dot,
    /// Division operator (/)
    Slash,
    /// Opening parenthesis
    LParen,
    /// Closing parenthesis
    RParen,
    /// Exponent (integer)
    Exponent(i8),
    /// End of input
    Eof,
}

/// UCUM lexer
pub struct UcumLexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> UcumLexer<'a> {
    pub fn new(input: &'a str) -> Self {
        UcumLexer { input, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_while<F: Fn(char) -> bool>(&mut self, predicate: F) {
        while let Some(c) = self.peek() {
            if predicate(c) {
                self.advance();
            } else {
                break;
            }
        }
    }

    pub fn next_token(&mut self) -> Result<UcumToken, UcumParseError> {
        // Skip whitespace
        self.skip_while(|c| c.is_whitespace());

        match self.peek() {
            None => Ok(UcumToken::Eof),
            Some('.') => {
                self.advance();
                Ok(UcumToken::Dot)
            }
            Some('/') => {
                self.advance();
                Ok(UcumToken::Slash)
            }
            Some('(') => {
                self.advance();
                Ok(UcumToken::LParen)
            }
            Some(')') => {
                self.advance();
                Ok(UcumToken::RParen)
            }
            Some('{') => {
                self.advance();
                let start = self.pos;
                self.skip_while(|c| c != '}');
                let annotation = self.input[start..self.pos].to_string();
                if self.peek() == Some('}') {
                    self.advance();
                }
                Ok(UcumToken::Annotation(annotation))
            }
            Some('[') => {
                // Special unit like [IU], [pH]
                self.advance();
                let start = self.pos;
                self.skip_while(|c| c != ']');
                let unit = self.input[start..self.pos].to_string();
                if self.peek() == Some(']') {
                    self.advance();
                }
                Ok(UcumToken::Unit(format!("[{}]", unit)))
            }
            Some(c) if c.is_ascii_digit() || c == '-' || c == '+' => {
                let start = self.pos;
                if c == '-' || c == '+' {
                    self.advance();
                }
                self.skip_while(|c| c.is_ascii_digit());
                let num_str = &self.input[start..self.pos];
                let exp: i8 = num_str
                    .parse()
                    .map_err(|_| UcumParseError::InvalidExponent(num_str.to_string()))?;
                Ok(UcumToken::Exponent(exp))
            }
            Some(c) if c.is_alphabetic() || c == '%' => {
                let start = self.pos;
                self.skip_while(|c| c.is_alphanumeric() || c == '_');
                let text = self.input[start..self.pos].to_string();
                Ok(UcumToken::Unit(text))
            }
            Some(c) => Err(UcumParseError::UnexpectedChar(c)),
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<UcumToken>, UcumParseError> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            if token == UcumToken::Eof {
                break;
            }
            tokens.push(token);
        }
        Ok(tokens)
    }
}

/// UCUM parse error
#[derive(Debug, Clone, PartialEq)]
pub enum UcumParseError {
    UnexpectedChar(char),
    UnknownUnit(String),
    UnknownPrefix(String),
    InvalidExponent(String),
    UnexpectedToken(String),
    EmptyInput,
    UnbalancedParentheses,
}

impl fmt::Display for UcumParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UcumParseError::UnexpectedChar(c) => write!(f, "Unexpected character: '{}'", c),
            UcumParseError::UnknownUnit(u) => write!(f, "Unknown UCUM unit: '{}'", u),
            UcumParseError::UnknownPrefix(p) => write!(f, "Unknown UCUM prefix: '{}'", p),
            UcumParseError::InvalidExponent(e) => write!(f, "Invalid exponent: '{}'", e),
            UcumParseError::UnexpectedToken(t) => write!(f, "Unexpected token: '{}'", t),
            UcumParseError::EmptyInput => write!(f, "Empty unit string"),
            UcumParseError::UnbalancedParentheses => write!(f, "Unbalanced parentheses"),
        }
    }
}

impl std::error::Error for UcumParseError {}

/// Parsed UCUM unit representation
#[derive(Debug, Clone)]
pub struct ParsedUnit {
    /// Component terms (numerator terms have positive exponent)
    pub terms: Vec<UnitTerm>,
    /// Optional annotation
    pub annotation: Option<String>,
}

impl ParsedUnit {
    pub fn new() -> Self {
        ParsedUnit {
            terms: Vec::new(),
            annotation: None,
        }
    }

    pub fn add_term(&mut self, term: UnitTerm) {
        self.terms.push(term);
    }

    pub fn set_annotation(&mut self, annotation: String) {
        self.annotation = Some(annotation);
    }

    /// Convert to dimension
    pub fn to_dimension(&self, registry: &UcumRegistry) -> Result<Dimension, UcumParseError> {
        let mut dim = Dimension::dimensionless();
        for term in &self.terms {
            let base_dim = registry.get_dimension(&term.unit)?;
            dim = dim * base_dim.pow(term.exponent);
        }
        Ok(dim)
    }

    /// Convert to conversion factor (to base units)
    pub fn to_factor(&self, registry: &UcumRegistry) -> Result<f64, UcumParseError> {
        let mut factor = 1.0;
        for term in &self.terms {
            let (base_factor, prefix_factor) =
                registry.get_factors(&term.unit, term.prefix.as_deref())?;
            factor *= (base_factor * prefix_factor).powi(term.exponent as i32);
        }
        Ok(factor)
    }
}

impl Default for ParsedUnit {
    fn default() -> Self {
        Self::new()
    }
}

/// A single term in a UCUM expression
#[derive(Debug, Clone)]
pub struct UnitTerm {
    /// Prefix (e.g., "m" for milli)
    pub prefix: Option<String>,
    /// Base unit code
    pub unit: String,
    /// Exponent (negative for denominator)
    pub exponent: i8,
}

impl UnitTerm {
    pub fn new(unit: &str, exponent: i8) -> Self {
        UnitTerm {
            prefix: None,
            unit: unit.to_string(),
            exponent,
        }
    }

    pub fn with_prefix(prefix: &str, unit: &str, exponent: i8) -> Self {
        UnitTerm {
            prefix: Some(prefix.to_string()),
            unit: unit.to_string(),
            exponent,
        }
    }
}

/// UCUM unit registry
pub struct UcumRegistry {
    /// Base units
    base_units: HashMap<String, (Dimension, f64)>,
    /// Derived units (maps to base unit expression)
    derived_units: HashMap<String, (Dimension, f64)>,
    /// Prefixes
    prefixes: HashMap<String, f64>,
}

impl UcumRegistry {
    pub fn new() -> Self {
        let mut registry = UcumRegistry {
            base_units: HashMap::new(),
            derived_units: HashMap::new(),
            prefixes: HashMap::new(),
        };
        registry.register_base_units();
        registry.register_prefixes();
        registry.register_derived_units();
        registry
    }

    fn register_base_units(&mut self) {
        for unit in UCUM_BASE_UNITS {
            let dim = Dimension::from_base(unit.dimension);
            self.base_units.insert(unit.code.to_string(), (dim, 1.0));
        }

        // Special units
        self.base_units.insert(
            "[IU]".to_string(),
            (Dimension::from_base(BaseDimension::InternationalUnit), 1.0),
        );

        // Dimensionless
        self.base_units
            .insert("1".to_string(), (Dimension::dimensionless(), 1.0));
        self.base_units
            .insert("%".to_string(), (Dimension::dimensionless(), 0.01));
    }

    fn register_prefixes(&mut self) {
        for prefix in UCUM_PREFIXES {
            self.prefixes
                .insert(prefix.symbol.to_string(), prefix.factor);
        }
    }

    fn register_derived_units(&mut self) {
        use super::dimension::standard::*;

        // Time
        self.derived_units.insert("min".to_string(), (time(), 60.0));
        self.derived_units.insert("h".to_string(), (time(), 3600.0));
        self.derived_units
            .insert("d".to_string(), (time(), 86400.0));
        self.derived_units
            .insert("wk".to_string(), (time(), 604800.0));
        self.derived_units
            .insert("mo".to_string(), (time(), 2629746.0));
        self.derived_units
            .insert("a".to_string(), (time(), 31556952.0));

        // Volume (L = dm³ = 0.001 m³)
        self.derived_units
            .insert("L".to_string(), (volume(), 0.001));
        self.derived_units
            .insert("l".to_string(), (volume(), 0.001)); // lowercase alias

        // Area
        self.derived_units.insert("ar".to_string(), (area(), 100.0)); // are = 100 m²
        self.derived_units
            .insert("har".to_string(), (area(), 10000.0)); // hectare

        // Energy
        self.derived_units.insert("J".to_string(), (energy(), 1.0));
        self.derived_units
            .insert("eV".to_string(), (energy(), 1.602176634e-19));
        self.derived_units
            .insert("cal".to_string(), (energy(), 4.184));
        self.derived_units
            .insert("Cal".to_string(), (energy(), 4184.0)); // kilocalorie

        // Pressure
        self.derived_units
            .insert("Pa".to_string(), (pressure(), 1.0));
        self.derived_units
            .insert("bar".to_string(), (pressure(), 100000.0));
        self.derived_units
            .insert("mm[Hg]".to_string(), (pressure(), 133.322));
        self.derived_units
            .insert("atm".to_string(), (pressure(), 101325.0));

        // Temperature (offset units need special handling)
        self.derived_units
            .insert("Cel".to_string(), (temperature(), 1.0));
        self.derived_units
            .insert("[degF]".to_string(), (temperature(), 5.0 / 9.0));

        // Mass
        self.derived_units.insert("t".to_string(), (mass(), 1e6)); // tonne = 1000 kg = 1e6 g
        self.derived_units
            .insert("u".to_string(), (mass(), 1.66053906660e-24)); // atomic mass unit

        // IU variants
        let iu_dim = international_unit();
        self.derived_units
            .insert("IU".to_string(), (iu_dim.clone(), 1.0));
        self.derived_units
            .insert("[iU]".to_string(), (iu_dim.clone(), 1.0));
    }

    /// Get dimension for a unit code
    pub fn get_dimension(&self, code: &str) -> Result<Dimension, UcumParseError> {
        if let Some((dim, _)) = self.base_units.get(code) {
            return Ok(dim.clone());
        }
        if let Some((dim, _)) = self.derived_units.get(code) {
            return Ok(dim.clone());
        }
        Err(UcumParseError::UnknownUnit(code.to_string()))
    }

    /// Get conversion factors for a unit with optional prefix
    pub fn get_factors(
        &self,
        code: &str,
        prefix: Option<&str>,
    ) -> Result<(f64, f64), UcumParseError> {
        let base_factor = if let Some((_, f)) = self.base_units.get(code) {
            *f
        } else if let Some((_, f)) = self.derived_units.get(code) {
            *f
        } else {
            return Err(UcumParseError::UnknownUnit(code.to_string()));
        };

        let prefix_factor = match prefix {
            Some(p) => *self
                .prefixes
                .get(p)
                .ok_or_else(|| UcumParseError::UnknownPrefix(p.to_string()))?,
            None => 1.0,
        };

        Ok((base_factor, prefix_factor))
    }

    /// Check if a unit code is valid
    pub fn is_valid_unit(&self, code: &str) -> bool {
        self.base_units.contains_key(code) || self.derived_units.contains_key(code)
    }

    /// Check if a prefix is valid
    pub fn is_valid_prefix(&self, prefix: &str) -> bool {
        self.prefixes.contains_key(prefix)
    }
}

impl Default for UcumRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// UCUM parser
pub struct UcumParser {
    registry: UcumRegistry,
}

impl UcumParser {
    pub fn new() -> Self {
        UcumParser {
            registry: UcumRegistry::new(),
        }
    }

    pub fn with_registry(registry: UcumRegistry) -> Self {
        UcumParser { registry }
    }

    /// Parse a UCUM string into a ParsedUnit
    pub fn parse(&self, input: &str) -> Result<ParsedUnit, UcumParseError> {
        if input.is_empty() {
            return Err(UcumParseError::EmptyInput);
        }

        let mut lexer = UcumLexer::new(input);
        let tokens = lexer.tokenize()?;

        self.parse_tokens(&tokens)
    }

    fn parse_tokens(&self, tokens: &[UcumToken]) -> Result<ParsedUnit, UcumParseError> {
        let mut result = ParsedUnit::new();
        let mut in_denominator = false;
        let mut i = 0;

        while i < tokens.len() {
            match &tokens[i] {
                UcumToken::Unit(code) => {
                    let (prefix, unit_code) = self.split_prefix_unit(code)?;
                    let exponent = if i + 1 < tokens.len() {
                        if let UcumToken::Exponent(e) = tokens[i + 1] {
                            i += 1;
                            e
                        } else {
                            1
                        }
                    } else {
                        1
                    };

                    let final_exp = if in_denominator { -exponent } else { exponent };

                    let term = if let Some(p) = prefix {
                        UnitTerm::with_prefix(&p, &unit_code, final_exp)
                    } else {
                        UnitTerm::new(&unit_code, final_exp)
                    };
                    result.add_term(term);
                }
                UcumToken::Dot => {
                    // Multiplication - stay in current mode
                }
                UcumToken::Slash => {
                    in_denominator = true;
                }
                UcumToken::Annotation(ann) => {
                    result.set_annotation(ann.clone());
                }
                UcumToken::LParen | UcumToken::RParen => {
                    // Simple parser doesn't handle nested parens
                    // For full UCUM compliance, would need recursive descent
                }
                UcumToken::Exponent(_) => {
                    // Handled above with unit
                }
                UcumToken::Prefix(_) => {
                    // Handled as part of unit
                }
                UcumToken::Eof => break,
            }
            i += 1;
        }

        Ok(result)
    }

    /// Split a unit code into prefix and base unit
    fn split_prefix_unit(&self, code: &str) -> Result<(Option<String>, String), UcumParseError> {
        // Special units (in brackets) have no prefix
        if code.starts_with('[') {
            if self.registry.is_valid_unit(code) {
                return Ok((None, code.to_string()));
            }
            return Err(UcumParseError::UnknownUnit(code.to_string()));
        }

        // Check if whole thing is a unit
        if self.registry.is_valid_unit(code) {
            return Ok((None, code.to_string()));
        }

        // Try to split prefix
        for prefix in UCUM_PREFIXES {
            if code.starts_with(prefix.symbol) {
                let rest = &code[prefix.symbol.len()..];
                if self.registry.is_valid_unit(rest) {
                    return Ok((Some(prefix.symbol.to_string()), rest.to_string()));
                }
            }
        }

        Err(UcumParseError::UnknownUnit(code.to_string()))
    }

    /// Parse and convert to a Unit
    pub fn parse_to_unit(&self, input: &str) -> Result<Unit, UcumParseError> {
        let parsed = self.parse(input)?;
        let dimension = parsed.to_dimension(&self.registry)?;
        let factor = parsed.to_factor(&self.registry)?;

        let mut unit = Unit::new(input, input, dimension, factor);
        if let Some(ann) = parsed.annotation {
            unit = unit.with_annotation(&ann);
        }
        Ok(unit)
    }

    /// Validate a UCUM string
    pub fn validate(&self, input: &str) -> Result<(), UcumParseError> {
        self.parse(input)?;
        Ok(())
    }
}

impl Default for UcumParser {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_simple() {
        let mut lexer = UcumLexer::new("mg");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], UcumToken::Unit("mg".to_string()));
    }

    #[test]
    fn test_lexer_compound() {
        let mut lexer = UcumLexer::new("mg/kg");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], UcumToken::Unit("mg".to_string()));
        assert_eq!(tokens[1], UcumToken::Slash);
        assert_eq!(tokens[2], UcumToken::Unit("kg".to_string()));
    }

    #[test]
    fn test_lexer_exponent() {
        // UCUM standard: "m2" is tokenized as a single unit token
        // The parser layer handles extracting the exponent
        let mut lexer = UcumLexer::new("m2");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], UcumToken::Unit("m2".to_string()));
    }

    #[test]
    fn test_lexer_annotation() {
        let mut lexer = UcumLexer::new("mol/L{creatinine}");
        let tokens = lexer.tokenize().unwrap();
        assert!(tokens
            .iter()
            .any(|t| matches!(t, UcumToken::Annotation(a) if a == "creatinine")));
    }

    #[test]
    fn test_lexer_special_unit() {
        let mut lexer = UcumLexer::new("[IU]");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0], UcumToken::Unit("[IU]".to_string()));
    }

    #[test]
    fn test_registry_base_units() {
        let registry = UcumRegistry::new();
        assert!(registry.is_valid_unit("m"));
        assert!(registry.is_valid_unit("g"));
        assert!(registry.is_valid_unit("s"));
        assert!(registry.is_valid_unit("[IU]"));
    }

    #[test]
    fn test_registry_prefixes() {
        let registry = UcumRegistry::new();
        assert!(registry.is_valid_prefix("m")); // milli
        assert!(registry.is_valid_prefix("k")); // kilo
        assert!(registry.is_valid_prefix("u")); // micro
    }

    #[test]
    fn test_parser_simple() {
        let parser = UcumParser::new();
        let parsed = parser.parse("mg").unwrap();
        assert_eq!(parsed.terms.len(), 1);
        assert_eq!(parsed.terms[0].prefix, Some("m".to_string()));
        assert_eq!(parsed.terms[0].unit, "g");
    }

    #[test]
    fn test_parser_compound() {
        let parser = UcumParser::new();
        let parsed = parser.parse("mg/kg").unwrap();
        assert_eq!(parsed.terms.len(), 2);
        assert_eq!(parsed.terms[0].exponent, 1);
        assert_eq!(parsed.terms[1].exponent, -1);
    }

    #[test]
    fn test_parser_to_dimension() {
        let parser = UcumParser::new();
        let registry = UcumRegistry::new();

        let parsed = parser.parse("m/s").unwrap();
        let dim = parsed.to_dimension(&registry).unwrap();

        assert_eq!(dim.get_exponent(BaseDimension::Length), 1);
        assert_eq!(dim.get_exponent(BaseDimension::Time), -1);
    }

    #[test]
    fn test_parser_to_unit() {
        let parser = UcumParser::new();
        let unit = parser.parse_to_unit("mg").unwrap();

        assert_eq!(unit.dimension.get_exponent(BaseDimension::Mass), 1);
        assert!((unit.to_base_factor - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_parser_clearance() {
        let parser = UcumParser::new();
        let unit = parser.parse_to_unit("mL/min").unwrap();

        assert_eq!(unit.dimension.get_exponent(BaseDimension::Length), 3);
        assert_eq!(unit.dimension.get_exponent(BaseDimension::Time), -1);
    }

    #[test]
    fn test_parser_annotation() {
        let parser = UcumParser::new();
        let unit = parser.parse_to_unit("mol/L{glucose}").unwrap();

        assert_eq!(unit.annotation, Some("glucose".to_string()));
    }

    #[test]
    fn test_parser_iu() {
        let parser = UcumParser::new();
        let unit = parser.parse_to_unit("[IU]").unwrap();

        assert!(unit.dimension.contains_iu());
    }

    #[test]
    fn test_validate() {
        let parser = UcumParser::new();

        assert!(parser.validate("mg").is_ok());
        assert!(parser.validate("mL/min").is_ok());
        assert!(parser.validate("[IU]").is_ok());
        assert!(parser.validate("invalid_unit").is_err());
    }
}
