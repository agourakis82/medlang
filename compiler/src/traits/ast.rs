// Week 53: Trait AST Definitions
//
// This module defines the AST nodes for trait declarations and implementations.

use std::fmt;

// =============================================================================
// Trait Declaration
// =============================================================================

/// A trait declaration
///
/// Example:
/// ```medlang
/// trait Numeric<T> {
///     fn zero() -> T;
///     fn one() -> T;
///     fn add(x: T, y: T) -> T;
/// }
/// ```
#[derive(Clone, Debug)]
pub struct TraitDecl {
    /// Trait name
    pub name: String,

    /// Type parameters for the trait itself
    /// For `trait Numeric<T>`, this is `["T"]`
    pub type_params: Vec<TraitTypeParam>,

    /// Super-traits (trait inheritance)
    /// For `trait Real<T>: Numeric<T> + Ord<T>`, this is `[Numeric<T>, Ord<T>]`
    pub super_traits: Vec<TraitRef>,

    /// Associated types (future)
    pub associated_types: Vec<AssociatedType>,

    /// Required methods
    pub methods: Vec<TraitMethod>,

    /// Default method implementations (optional)
    pub default_impls: Vec<DefaultImpl>,

    /// Visibility
    pub visibility: Visibility,
}

impl TraitDecl {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_params: Vec::new(),
            super_traits: Vec::new(),
            associated_types: Vec::new(),
            methods: Vec::new(),
            default_impls: Vec::new(),
            visibility: Visibility::Public,
        }
    }

    pub fn with_type_param(mut self, name: &str) -> Self {
        self.type_params.push(TraitTypeParam::simple(name));
        self
    }

    pub fn with_type_param_bounded(mut self, name: &str, bounds: Vec<TraitRef>) -> Self {
        self.type_params.push(TraitTypeParam::bounded(name, bounds));
        self
    }

    pub fn with_super_trait(mut self, tr: TraitRef) -> Self {
        self.super_traits.push(tr);
        self
    }

    pub fn with_method(mut self, method: TraitMethod) -> Self {
        self.methods.push(method);
        self
    }

    pub fn with_default_impl(mut self, impl_: DefaultImpl) -> Self {
        self.default_impls.push(impl_);
        self
    }

    pub fn with_associated_type(mut self, assoc: AssociatedType) -> Self {
        self.associated_types.push(assoc);
        self
    }

    /// Check if trait has a method with given name
    pub fn has_method(&self, name: &str) -> bool {
        self.methods.iter().any(|m| m.name == name)
    }

    /// Get method by name
    pub fn get_method(&self, name: &str) -> Option<&TraitMethod> {
        self.methods.iter().find(|m| m.name == name)
    }
}

impl fmt::Display for TraitDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trait {}", self.name)?;
        if !self.type_params.is_empty() {
            write!(f, "<")?;
            for (i, tp) in self.type_params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", tp)?;
            }
            write!(f, ">")?;
        }
        if !self.super_traits.is_empty() {
            write!(f, ": ")?;
            for (i, st) in self.super_traits.iter().enumerate() {
                if i > 0 {
                    write!(f, " + ")?;
                }
                write!(f, "{}", st)?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// Type Parameter with Bounds
// =============================================================================

/// Type parameter in a trait or impl
#[derive(Clone, Debug)]
pub struct TraitTypeParam {
    /// Parameter name: T, U, etc.
    pub name: String,

    /// Trait bounds: T: Numeric + Ord
    pub bounds: Vec<TraitRef>,

    /// Default type (optional): T = Float
    pub default: Option<String>,
}

impl TraitTypeParam {
    pub fn simple(name: &str) -> Self {
        Self {
            name: name.to_string(),
            bounds: Vec::new(),
            default: None,
        }
    }

    pub fn bounded(name: &str, bounds: Vec<TraitRef>) -> Self {
        Self {
            name: name.to_string(),
            bounds,
            default: None,
        }
    }

    pub fn with_default(name: &str, default: &str) -> Self {
        Self {
            name: name.to_string(),
            bounds: Vec::new(),
            default: Some(default.to_string()),
        }
    }
}

impl fmt::Display for TraitTypeParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.bounds.is_empty() {
            write!(f, ": ")?;
            for (i, b) in self.bounds.iter().enumerate() {
                if i > 0 {
                    write!(f, " + ")?;
                }
                write!(f, "{}", b)?;
            }
        }
        if let Some(default) = &self.default {
            write!(f, " = {}", default)?;
        }
        Ok(())
    }
}

// =============================================================================
// Trait Reference
// =============================================================================

/// Reference to a trait with type arguments
/// Example: `Numeric<Float>`, `Ord<T>`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitRef {
    pub trait_name: String,
    pub type_args: Vec<String>,
}

impl TraitRef {
    pub fn simple(name: &str) -> Self {
        Self {
            trait_name: name.to_string(),
            type_args: Vec::new(),
        }
    }

    pub fn with_type_args(name: &str, args: Vec<String>) -> Self {
        Self {
            trait_name: name.to_string(),
            type_args: args,
        }
    }

    pub fn with_single_arg(name: &str, arg: &str) -> Self {
        Self {
            trait_name: name.to_string(),
            type_args: vec![arg.to_string()],
        }
    }
}

impl fmt::Display for TraitRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.trait_name)?;
        if !self.type_args.is_empty() {
            write!(f, "<")?;
            for (i, arg) in self.type_args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", arg)?;
            }
            write!(f, ">")?;
        }
        Ok(())
    }
}

// =============================================================================
// Trait Method
// =============================================================================

/// A method signature in a trait
#[derive(Clone, Debug)]
pub struct TraitMethod {
    /// Method name
    pub name: String,

    /// Method's own type parameters (rare, but possible)
    pub type_params: Vec<TraitTypeParam>,

    /// Parameters with names and types
    pub params: Vec<MethodParam>,

    /// Return type
    pub ret_type: String,

    /// Whether this method has a default implementation
    pub has_default: bool,
}

impl TraitMethod {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_params: Vec::new(),
            params: Vec::new(),
            ret_type: "Unit".to_string(),
            has_default: false,
        }
    }

    pub fn with_type_param(mut self, name: &str) -> Self {
        self.type_params.push(TraitTypeParam::simple(name));
        self
    }

    pub fn with_param(mut self, name: &str, ty: &str) -> Self {
        self.params.push(MethodParam {
            name: name.to_string(),
            ty: ty.to_string(),
        });
        self
    }

    pub fn with_ret_type(mut self, ty: &str) -> Self {
        self.ret_type = ty.to_string();
        self
    }

    pub fn with_default(mut self) -> Self {
        self.has_default = true;
        self
    }
}

impl fmt::Display for TraitMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn {}", self.name)?;
        if !self.type_params.is_empty() {
            write!(f, "<")?;
            for (i, tp) in self.type_params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", tp)?;
            }
            write!(f, ">")?;
        }
        write!(f, "(")?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", p.name, p.ty)?;
        }
        write!(f, ") -> {}", self.ret_type)
    }
}

/// Method parameter
#[derive(Clone, Debug)]
pub struct MethodParam {
    pub name: String,
    pub ty: String,
}

// =============================================================================
// Default Implementation
// =============================================================================

/// Default implementation for a trait method
#[derive(Clone, Debug)]
pub struct DefaultImpl {
    pub method_name: String,
    pub body: String, // Simplified: just store as string for now
}

// =============================================================================
// Associated Type
// =============================================================================

/// Associated type in a trait (future feature)
#[derive(Clone, Debug)]
pub struct AssociatedType {
    pub name: String,
    pub bounds: Vec<TraitRef>,
    pub default: Option<String>,
}

impl AssociatedType {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            bounds: Vec::new(),
            default: None,
        }
    }

    pub fn with_bound(mut self, bound: TraitRef) -> Self {
        self.bounds.push(bound);
        self
    }

    pub fn with_default(mut self, default: &str) -> Self {
        self.default = Some(default.to_string());
        self
    }
}

// =============================================================================
// Trait Implementation
// =============================================================================

/// A trait implementation
///
/// Example:
/// ```medlang
/// impl Numeric<Float> {
///     fn zero() -> Float { 0.0 }
///     fn add(x: Float, y: Float) -> Float { x + y }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct TraitImpl {
    /// The trait being implemented
    pub trait_ref: TraitRef,

    /// Type parameters for generic impls
    /// For `impl<T: Ord> Sortable<Vector<T>>`, this is `[T: Ord]`
    pub type_params: Vec<TraitTypeParam>,

    /// "Where" clauses for additional constraints
    pub where_clauses: Vec<WhereClause>,

    /// Method implementations
    pub methods: Vec<ImplMethod>,

    /// Associated type bindings (future)
    pub type_bindings: Vec<TypeBinding>,
}

impl TraitImpl {
    pub fn new(trait_ref: TraitRef) -> Self {
        Self {
            trait_ref,
            type_params: Vec::new(),
            where_clauses: Vec::new(),
            methods: Vec::new(),
            type_bindings: Vec::new(),
        }
    }

    pub fn with_type_param(mut self, name: &str) -> Self {
        self.type_params.push(TraitTypeParam::simple(name));
        self
    }

    pub fn with_type_param_bounded(mut self, name: &str, bounds: Vec<TraitRef>) -> Self {
        self.type_params.push(TraitTypeParam::bounded(name, bounds));
        self
    }

    pub fn with_where_clause(mut self, clause: WhereClause) -> Self {
        self.where_clauses.push(clause);
        self
    }

    pub fn with_method(mut self, method: ImplMethod) -> Self {
        self.methods.push(method);
        self
    }

    pub fn with_method_body(
        mut self,
        name: &str,
        params: Vec<(&str, &str)>,
        ret_type: &str,
        body: &str,
    ) -> Self {
        self.methods.push(ImplMethod {
            name: name.to_string(),
            type_params: Vec::new(),
            params: params
                .into_iter()
                .map(|(n, t)| MethodParam {
                    name: n.to_string(),
                    ty: t.to_string(),
                })
                .collect(),
            ret_type: ret_type.to_string(),
            body: body.to_string(),
        });
        self
    }

    pub fn with_type_binding(mut self, binding: TypeBinding) -> Self {
        self.type_bindings.push(binding);
        self
    }

    /// Check if impl provides method
    pub fn has_method(&self, name: &str) -> bool {
        self.methods.iter().any(|m| m.name == name)
    }

    /// Get method by name
    pub fn get_method(&self, name: &str) -> Option<&ImplMethod> {
        self.methods.iter().find(|m| m.name == name)
    }
}

impl fmt::Display for TraitImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "impl")?;
        if !self.type_params.is_empty() {
            write!(f, "<")?;
            for (i, tp) in self.type_params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", tp)?;
            }
            write!(f, ">")?;
        }
        write!(f, " {}", self.trait_ref)?;
        if !self.where_clauses.is_empty() {
            write!(f, " where ")?;
            for (i, wc) in self.where_clauses.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", wc)?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// Implementation Method
// =============================================================================

/// A method implementation in an impl block
#[derive(Clone, Debug)]
pub struct ImplMethod {
    pub name: String,
    pub type_params: Vec<TraitTypeParam>,
    pub params: Vec<MethodParam>,
    pub ret_type: String,
    pub body: String, // Simplified: store as string
}

impl ImplMethod {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_params: Vec::new(),
            params: Vec::new(),
            ret_type: "Unit".to_string(),
            body: String::new(),
        }
    }

    pub fn with_param(mut self, name: &str, ty: &str) -> Self {
        self.params.push(MethodParam {
            name: name.to_string(),
            ty: ty.to_string(),
        });
        self
    }

    pub fn with_ret_type(mut self, ty: &str) -> Self {
        self.ret_type = ty.to_string();
        self
    }

    pub fn with_body(mut self, body: &str) -> Self {
        self.body = body.to_string();
        self
    }
}

// =============================================================================
// Where Clause
// =============================================================================

/// A where clause constraint
#[derive(Clone, Debug)]
pub struct WhereClause {
    pub type_param: String,
    pub bounds: Vec<TraitRef>,
}

impl WhereClause {
    pub fn new(type_param: &str, bounds: Vec<TraitRef>) -> Self {
        Self {
            type_param: type_param.to_string(),
            bounds,
        }
    }
}

impl fmt::Display for WhereClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.type_param)?;
        for (i, b) in self.bounds.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}", b)?;
        }
        Ok(())
    }
}

// =============================================================================
// Type Binding
// =============================================================================

/// Associated type binding in an impl
#[derive(Clone, Debug)]
pub struct TypeBinding {
    pub name: String,
    pub ty: String,
}

impl TypeBinding {
    pub fn new(name: &str, ty: &str) -> Self {
        Self {
            name: name.to_string(),
            ty: ty.to_string(),
        }
    }
}

// =============================================================================
// Visibility
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
    Crate,
}

impl Default for Visibility {
    fn default() -> Self {
        Visibility::Private
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_decl_builder() {
        let trait_decl = TraitDecl::new("Numeric")
            .with_type_param("T")
            .with_method(TraitMethod::new("zero").with_ret_type("T"))
            .with_method(
                TraitMethod::new("add")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            );

        assert_eq!(trait_decl.name, "Numeric");
        assert_eq!(trait_decl.type_params.len(), 1);
        assert_eq!(trait_decl.type_params[0].name, "T");
        assert_eq!(trait_decl.methods.len(), 2);
        assert!(trait_decl.has_method("zero"));
        assert!(trait_decl.has_method("add"));
        assert!(!trait_decl.has_method("mul"));
    }

    #[test]
    fn test_trait_ref() {
        let simple = TraitRef::simple("Eq");
        assert_eq!(simple.trait_name, "Eq");
        assert!(simple.type_args.is_empty());

        let with_args = TraitRef::with_type_args("Ord", vec!["Float".to_string()]);
        assert_eq!(with_args.trait_name, "Ord");
        assert_eq!(with_args.type_args, vec!["Float".to_string()]);
    }

    #[test]
    fn test_trait_impl_builder() {
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body(
                "add",
                vec![("x", "Float"), ("y", "Float")],
                "Float",
                "x + y",
            );

        assert_eq!(impl_.trait_ref.trait_name, "Numeric");
        assert_eq!(impl_.methods.len(), 2);
        assert!(impl_.has_method("zero"));
        assert!(impl_.has_method("add"));
    }

    #[test]
    fn test_trait_method_display() {
        let method = TraitMethod::new("compare")
            .with_param("a", "T")
            .with_param("b", "T")
            .with_ret_type("Int");

        assert_eq!(method.to_string(), "fn compare(a: T, b: T) -> Int");
    }

    #[test]
    fn test_where_clause() {
        let clause = WhereClause::new("T", vec![TraitRef::simple("Eq"), TraitRef::simple("Hash")]);

        assert_eq!(clause.to_string(), "T: Eq + Hash");
    }

    #[test]
    fn test_trait_decl_display() {
        let trait_decl = TraitDecl::new("Sortable")
            .with_type_param("T")
            .with_super_trait(TraitRef::with_single_arg("Ord", "T"));

        assert_eq!(trait_decl.to_string(), "trait Sortable<T>: Ord<T>");
    }

    #[test]
    fn test_bounded_type_param() {
        let param = TraitTypeParam::bounded(
            "T",
            vec![TraitRef::simple("Numeric"), TraitRef::simple("Ord")],
        );

        assert_eq!(param.to_string(), "T: Numeric + Ord");
    }
}
