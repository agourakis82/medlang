// Week 53: Trait Type System
//
// Type-level representations for traits and their implementations.

use std::collections::HashMap;
use std::fmt;

// =============================================================================
// Trait Declaration Type
// =============================================================================

/// Internal representation of a trait declaration
#[derive(Clone, Debug)]
pub struct TraitDeclTy {
    /// Trait name
    pub name: String,

    /// Type parameters
    pub type_params: Vec<String>,

    /// Super-traits
    pub super_traits: Vec<TraitRefTy>,

    /// Method signatures: name -> signature
    pub methods: HashMap<String, MethodSig>,

    /// Associated types
    pub associated_types: HashMap<String, AssocTypeTy>,
}

impl TraitDeclTy {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_params: Vec::new(),
            super_traits: Vec::new(),
            methods: HashMap::new(),
            associated_types: HashMap::new(),
        }
    }

    pub fn add_type_param(&mut self, name: &str) {
        self.type_params.push(name.to_string());
    }

    pub fn add_super_trait(&mut self, tr: TraitRefTy) {
        self.super_traits.push(tr);
    }

    pub fn add_method(&mut self, name: &str, sig: MethodSig) {
        self.methods.insert(name.to_string(), sig);
    }

    pub fn add_associated_type(&mut self, name: &str, assoc: AssocTypeTy) {
        self.associated_types.insert(name.to_string(), assoc);
    }

    /// Get method signature by name
    pub fn get_method(&self, name: &str) -> Option<&MethodSig> {
        self.methods.get(name)
    }

    /// Check if this trait has a super-trait
    pub fn has_super_trait(&self, trait_name: &str) -> bool {
        self.super_traits
            .iter()
            .any(|st| st.trait_name == trait_name)
    }
}

// =============================================================================
// Trait Reference Type
// =============================================================================

/// Internal representation of a trait reference
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitRefTy {
    pub trait_name: String,
    pub type_args: Vec<String>,
}

impl TraitRefTy {
    pub fn new(name: &str, args: Vec<String>) -> Self {
        Self {
            trait_name: name.to_string(),
            type_args: args,
        }
    }

    pub fn simple(name: &str) -> Self {
        Self {
            trait_name: name.to_string(),
            type_args: Vec::new(),
        }
    }

    /// Substitute type parameters with concrete types
    pub fn substitute(&self, subst: &HashMap<String, String>) -> Self {
        let new_args = self
            .type_args
            .iter()
            .map(|arg| subst.get(arg).cloned().unwrap_or_else(|| arg.clone()))
            .collect();

        Self {
            trait_name: self.trait_name.clone(),
            type_args: new_args,
        }
    }
}

impl fmt::Display for TraitRefTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.trait_name)?;
        if !self.type_args.is_empty() {
            write!(f, "<{}>", self.type_args.join(", "))?;
        }
        Ok(())
    }
}

// =============================================================================
// Method Signature
// =============================================================================

/// Method signature (type-level)
#[derive(Clone, Debug)]
pub struct MethodSig {
    /// Method's own type parameters
    pub type_params: Vec<String>,

    /// Parameter names
    pub param_names: Vec<String>,

    /// Parameter types (as strings, may contain type variables)
    pub param_types: Vec<String>,

    /// Return type
    pub ret_type: String,
}

impl MethodSig {
    pub fn new(params: Vec<(&str, &str)>, ret: &str) -> Self {
        Self {
            type_params: Vec::new(),
            param_names: params.iter().map(|(n, _)| n.to_string()).collect(),
            param_types: params.iter().map(|(_, t)| t.to_string()).collect(),
            ret_type: ret.to_string(),
        }
    }

    pub fn nullary(ret: &str) -> Self {
        Self {
            type_params: Vec::new(),
            param_names: Vec::new(),
            param_types: Vec::new(),
            ret_type: ret.to_string(),
        }
    }

    pub fn unary(param_name: &str, param_type: &str, ret: &str) -> Self {
        Self {
            type_params: Vec::new(),
            param_names: vec![param_name.to_string()],
            param_types: vec![param_type.to_string()],
            ret_type: ret.to_string(),
        }
    }

    pub fn binary(p1: (&str, &str), p2: (&str, &str), ret: &str) -> Self {
        Self {
            type_params: Vec::new(),
            param_names: vec![p1.0.to_string(), p2.0.to_string()],
            param_types: vec![p1.1.to_string(), p2.1.to_string()],
            ret_type: ret.to_string(),
        }
    }

    /// Substitute type parameters with concrete types
    pub fn substitute(&self, subst: &HashMap<String, String>) -> Self {
        let substitute_type = |ty: &str| subst.get(ty).cloned().unwrap_or_else(|| ty.to_string());

        Self {
            type_params: self.type_params.clone(),
            param_names: self.param_names.clone(),
            param_types: self
                .param_types
                .iter()
                .map(|t| substitute_type(t))
                .collect(),
            ret_type: substitute_type(&self.ret_type),
        }
    }

    /// Get arity (number of parameters)
    pub fn arity(&self) -> usize {
        self.param_types.len()
    }
}

impl fmt::Display for MethodSig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, (name, ty)) in self.param_names.iter().zip(&self.param_types).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", name, ty)?;
        }
        write!(f, ") -> {}", self.ret_type)
    }
}

// =============================================================================
// Associated Type
// =============================================================================

/// Associated type info
#[derive(Clone, Debug)]
pub struct AssocTypeTy {
    pub bounds: Vec<TraitRefTy>,
    pub default: Option<String>,
}

impl AssocTypeTy {
    pub fn new() -> Self {
        Self {
            bounds: Vec::new(),
            default: None,
        }
    }

    pub fn with_bound(mut self, bound: TraitRefTy) -> Self {
        self.bounds.push(bound);
        self
    }

    pub fn with_default(mut self, default: &str) -> Self {
        self.default = Some(default.to_string());
        self
    }
}

impl Default for AssocTypeTy {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Trait Implementation Type
// =============================================================================

/// Internal representation of a trait implementation
#[derive(Clone, Debug)]
pub struct TraitImplTy {
    /// The trait being implemented
    pub trait_ref: TraitRefTy,

    /// Type parameters for this impl (for generic impls)
    pub type_params: Vec<TypeParamTy>,

    /// Where clauses
    pub where_clauses: Vec<WhereClauseTy>,

    /// Method implementations: name -> (mangled_symbol, signature)
    pub methods: HashMap<String, (String, MethodSig)>,

    /// Associated type bindings
    pub type_bindings: HashMap<String, String>,
}

impl TraitImplTy {
    pub fn new(trait_ref: TraitRefTy) -> Self {
        Self {
            trait_ref,
            type_params: Vec::new(),
            where_clauses: Vec::new(),
            methods: HashMap::new(),
            type_bindings: HashMap::new(),
        }
    }

    pub fn add_type_param(&mut self, param: TypeParamTy) {
        self.type_params.push(param);
    }

    pub fn add_method(&mut self, name: &str, symbol: &str, sig: MethodSig) {
        self.methods
            .insert(name.to_string(), (symbol.to_string(), sig));
    }

    pub fn add_type_binding(&mut self, name: &str, ty: &str) {
        self.type_bindings.insert(name.to_string(), ty.to_string());
    }

    /// Get the mangled symbol for a method
    pub fn get_method_symbol(&self, name: &str) -> Option<&str> {
        self.methods.get(name).map(|(s, _)| s.as_str())
    }

    /// Get method signature
    pub fn get_method_sig(&self, name: &str) -> Option<&MethodSig> {
        self.methods.get(name).map(|(_, sig)| sig)
    }

    /// Check if this is a generic impl (has type params)
    pub fn is_generic(&self) -> bool {
        !self.type_params.is_empty()
    }

    /// Check if impl provides a method
    pub fn has_method(&self, name: &str) -> bool {
        self.methods.contains_key(name)
    }
}

// =============================================================================
// Type Parameter (in impl)
// =============================================================================

/// Type parameter with bounds
#[derive(Clone, Debug)]
pub struct TypeParamTy {
    pub name: String,
    pub bounds: Vec<TraitRefTy>,
}

impl TypeParamTy {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            bounds: Vec::new(),
        }
    }

    pub fn with_bound(mut self, bound: TraitRefTy) -> Self {
        self.bounds.push(bound);
        self
    }
}

// =============================================================================
// Where Clause (type level)
// =============================================================================

/// Where clause
#[derive(Clone, Debug)]
pub struct WhereClauseTy {
    pub type_param: String,
    pub bounds: Vec<TraitRefTy>,
}

impl WhereClauseTy {
    pub fn new(type_param: &str, bounds: Vec<TraitRefTy>) -> Self {
        Self {
            type_param: type_param.to_string(),
            bounds,
        }
    }
}

// =============================================================================
// Trait Implementation Index
// =============================================================================

/// Index structure for efficient trait impl lookup
#[derive(Clone, Debug, Default)]
pub struct TraitImplIndex {
    /// All implementations
    pub impls: Vec<TraitImplTy>,

    /// Index by trait name: trait_name -> [impl indices]
    pub by_trait: HashMap<String, Vec<usize>>,

    /// Index by first type argument: type -> [impl indices]
    pub by_type: HashMap<String, Vec<usize>>,
}

impl TraitImplIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an implementation
    pub fn add(&mut self, impl_ty: TraitImplTy) {
        let idx = self.impls.len();

        // Index by trait name
        self.by_trait
            .entry(impl_ty.trait_ref.trait_name.clone())
            .or_default()
            .push(idx);

        // Index by first type argument (common case: impl Trait<T>)
        if let Some(first_type) = impl_ty.trait_ref.type_args.first() {
            self.by_type
                .entry(first_type.clone())
                .or_default()
                .push(idx);
        }

        self.impls.push(impl_ty);
    }

    /// Find implementations for a trait
    pub fn find_for_trait(&self, trait_name: &str) -> Vec<&TraitImplTy> {
        self.by_trait
            .get(trait_name)
            .map(|indices| indices.iter().map(|&i| &self.impls[i]).collect())
            .unwrap_or_default()
    }

    /// Find implementations for a specific type
    pub fn find_for_type(&self, ty: &str) -> Vec<&TraitImplTy> {
        self.by_type
            .get(ty)
            .map(|indices| indices.iter().map(|&i| &self.impls[i]).collect())
            .unwrap_or_default()
    }

    /// Find a specific implementation: impl Trait<Type>
    pub fn find_impl(&self, trait_name: &str, type_args: &[String]) -> Option<&TraitImplTy> {
        self.by_trait
            .get(trait_name)?
            .iter()
            .map(|&i| &self.impls[i])
            .find(|impl_ty| {
                // For now, exact match. Could add unification for generic impls later.
                impl_ty.trait_ref.type_args == type_args
            })
    }

    /// Find impl with potential generic matching
    pub fn find_matching_impl(
        &self,
        trait_name: &str,
        type_args: &[String],
    ) -> Option<(&TraitImplTy, HashMap<String, String>)> {
        // First try exact match
        if let Some(impl_) = self.find_impl(trait_name, type_args) {
            return Some((impl_, HashMap::new()));
        }

        // Try generic impls
        for impl_ in self.find_for_trait(trait_name) {
            if impl_.is_generic() {
                if let Some(subst) = self.try_match_impl(impl_, type_args) {
                    return Some((impl_, subst));
                }
            }
        }

        None
    }

    /// Try to match a generic impl against concrete type args
    fn try_match_impl(
        &self,
        impl_: &TraitImplTy,
        type_args: &[String],
    ) -> Option<HashMap<String, String>> {
        if impl_.trait_ref.type_args.len() != type_args.len() {
            return None;
        }

        let mut subst = HashMap::new();

        for (impl_arg, concrete_arg) in impl_.trait_ref.type_args.iter().zip(type_args) {
            // Check if impl_arg is a type parameter
            if impl_.type_params.iter().any(|tp| tp.name == *impl_arg) {
                // It's a type param, bind it
                if let Some(existing) = subst.get(impl_arg) {
                    if existing != concrete_arg {
                        return None; // Conflicting bindings
                    }
                } else {
                    subst.insert(impl_arg.clone(), concrete_arg.clone());
                }
            } else if impl_arg != concrete_arg {
                // Not a type param and doesn't match
                return None;
            }
        }

        Some(subst)
    }

    /// Get all impls
    pub fn all_impls(&self) -> &[TraitImplTy] {
        &self.impls
    }

    /// Count of impls
    pub fn len(&self) -> usize {
        self.impls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.impls.is_empty()
    }
}

// =============================================================================
// Trait Resolver
// =============================================================================

/// Trait method resolver
pub struct TraitResolver<'a> {
    traits: &'a HashMap<String, TraitDeclTy>,
    impls: &'a TraitImplIndex,
}

impl<'a> TraitResolver<'a> {
    pub fn new(traits: &'a HashMap<String, TraitDeclTy>, impls: &'a TraitImplIndex) -> Self {
        Self { traits, impls }
    }

    /// Resolve a trait method call to a concrete function symbol
    pub fn resolve_method(
        &self,
        trait_name: &str,
        method_name: &str,
        type_args: &[String],
    ) -> Result<ResolvedMethod, ResolveError> {
        // Find the trait declaration
        let trait_decl = self
            .traits
            .get(trait_name)
            .ok_or_else(|| ResolveError::UnknownTrait {
                name: trait_name.to_string(),
            })?;

        // Check method exists in trait
        let method_sig =
            trait_decl
                .get_method(method_name)
                .ok_or_else(|| ResolveError::UnknownMethod {
                    trait_name: trait_name.to_string(),
                    method_name: method_name.to_string(),
                })?;

        // Find the implementation
        let (impl_ty, subst) = self
            .impls
            .find_matching_impl(trait_name, type_args)
            .ok_or_else(|| ResolveError::NoImpl {
                trait_name: trait_name.to_string(),
                type_args: type_args.to_vec(),
            })?;

        // Get the method's generated symbol
        let symbol = impl_ty.get_method_symbol(method_name).ok_or_else(|| {
            ResolveError::MethodNotProvided {
                trait_name: trait_name.to_string(),
                method_name: method_name.to_string(),
            }
        })?;

        // Apply substitution if this was a generic impl
        let concrete_sig = if !subst.is_empty() {
            method_sig.substitute(&subst)
        } else {
            method_sig.clone()
        };

        Ok(ResolvedMethod {
            symbol: symbol.to_string(),
            signature: concrete_sig,
            trait_ref: impl_ty.trait_ref.clone(),
        })
    }

    /// Check if a type satisfies a trait bound
    pub fn satisfies_bound(&self, ty: &str, bound: &TraitRefTy) -> bool {
        // Build type args with ty substituted for trait's first type param
        let type_args = if bound.type_args.is_empty() {
            vec![ty.to_string()]
        } else {
            bound.type_args.clone()
        };

        self.impls
            .find_impl(&bound.trait_name, &type_args)
            .is_some()
    }

    /// Check all bounds for a type
    pub fn check_bounds(&self, ty: &str, bounds: &[TraitRefTy]) -> Result<(), ResolveError> {
        for bound in bounds {
            if !self.satisfies_bound(ty, bound) {
                return Err(ResolveError::BoundNotSatisfied {
                    ty: ty.to_string(),
                    bound: bound.to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Result of resolving a trait method
#[derive(Clone, Debug)]
pub struct ResolvedMethod {
    /// The generated function symbol
    pub symbol: String,
    /// The concrete method signature
    pub signature: MethodSig,
    /// The trait reference that provided this method
    pub trait_ref: TraitRefTy,
}

/// Errors during trait resolution
#[derive(Debug, Clone)]
pub enum ResolveError {
    UnknownTrait {
        name: String,
    },
    UnknownMethod {
        trait_name: String,
        method_name: String,
    },
    NoImpl {
        trait_name: String,
        type_args: Vec<String>,
    },
    MethodNotProvided {
        trait_name: String,
        method_name: String,
    },
    BoundNotSatisfied {
        ty: String,
        bound: String,
    },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolveError::UnknownTrait { name } => write!(f, "unknown trait: {}", name),
            ResolveError::UnknownMethod {
                trait_name,
                method_name,
            } => write!(f, "trait {} has no method {}", trait_name, method_name),
            ResolveError::NoImpl {
                trait_name,
                type_args,
            } => {
                write!(
                    f,
                    "no implementation of {} for [{}]",
                    trait_name,
                    type_args.join(", ")
                )
            }
            ResolveError::MethodNotProvided {
                trait_name,
                method_name,
            } => write!(
                f,
                "impl {} does not provide method {}",
                trait_name, method_name
            ),
            ResolveError::BoundNotSatisfied { ty, bound } => {
                write!(f, "type {} does not satisfy bound {}", ty, bound)
            }
        }
    }
}

impl std::error::Error for ResolveError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_decl_ty() {
        let mut decl = TraitDeclTy::new("Numeric");
        decl.add_type_param("T");
        decl.add_method("zero", MethodSig::nullary("T"));
        decl.add_method("add", MethodSig::binary(("x", "T"), ("y", "T"), "T"));

        assert_eq!(decl.name, "Numeric");
        assert_eq!(decl.type_params, vec!["T"]);
        assert!(decl.get_method("zero").is_some());
        assert!(decl.get_method("add").is_some());
        assert!(decl.get_method("mul").is_none());
    }

    #[test]
    fn test_trait_ref_ty() {
        let simple = TraitRefTy::simple("Eq");
        assert_eq!(simple.to_string(), "Eq");

        let with_args = TraitRefTy::new("Numeric", vec!["Float".to_string()]);
        assert_eq!(with_args.to_string(), "Numeric<Float>");
    }

    #[test]
    fn test_method_sig_substitute() {
        let sig = MethodSig::binary(("x", "T"), ("y", "T"), "T");

        let mut subst = HashMap::new();
        subst.insert("T".to_string(), "Float".to_string());

        let concrete = sig.substitute(&subst);
        assert_eq!(concrete.param_types, vec!["Float", "Float"]);
        assert_eq!(concrete.ret_type, "Float");
    }

    #[test]
    fn test_trait_impl_index() {
        let mut index = TraitImplIndex::new();

        let impl1 = TraitImplTy::new(TraitRefTy::new("Numeric", vec!["Float".to_string()]));
        let impl2 = TraitImplTy::new(TraitRefTy::new("Numeric", vec!["Int".to_string()]));
        let impl3 = TraitImplTy::new(TraitRefTy::new("Ord", vec!["Float".to_string()]));

        index.add(impl1);
        index.add(impl2);
        index.add(impl3);

        assert_eq!(index.len(), 3);
        assert_eq!(index.find_for_trait("Numeric").len(), 2);
        assert_eq!(index.find_for_trait("Ord").len(), 1);
        assert_eq!(index.find_for_type("Float").len(), 2);
        assert_eq!(index.find_for_type("Int").len(), 1);
    }

    #[test]
    fn test_trait_resolver() {
        let mut traits = HashMap::new();
        let mut decl = TraitDeclTy::new("Numeric");
        decl.add_type_param("T");
        decl.add_method("add", MethodSig::binary(("x", "T"), ("y", "T"), "T"));
        traits.insert("Numeric".to_string(), decl);

        let mut index = TraitImplIndex::new();
        let mut impl_ = TraitImplTy::new(TraitRefTy::new("Numeric", vec!["Float".to_string()]));
        impl_.add_method(
            "add",
            "Numeric_Float_add",
            MethodSig::binary(("x", "Float"), ("y", "Float"), "Float"),
        );
        index.add(impl_);

        let resolver = TraitResolver::new(&traits, &index);

        let result = resolver.resolve_method("Numeric", "add", &["Float".to_string()]);
        assert!(result.is_ok());
        let resolved = result.unwrap();
        assert_eq!(resolved.symbol, "Numeric_Float_add");
    }

    #[test]
    fn test_satisfies_bound() {
        let mut traits = HashMap::new();
        let decl = TraitDeclTy::new("Eq");
        traits.insert("Eq".to_string(), decl);

        let mut index = TraitImplIndex::new();
        index.add(TraitImplTy::new(TraitRefTy::new(
            "Eq",
            vec!["Float".to_string()],
        )));

        let resolver = TraitResolver::new(&traits, &index);

        assert!(resolver.satisfies_bound("Float", &TraitRefTy::simple("Eq")));
        assert!(!resolver.satisfies_bound("String", &TraitRefTy::simple("Eq")));
    }
}
