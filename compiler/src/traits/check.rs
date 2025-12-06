// Week 53: Trait Type Checker
//
// Validates trait declarations and implementations.

use super::ast::{ImplMethod, TraitDecl, TraitImpl, TraitMethod, TraitRef};
use super::types::{
    MethodSig, TraitDeclTy, TraitImplIndex, TraitImplTy, TraitRefTy, TraitResolver, TypeParamTy,
    WhereClauseTy,
};
use std::collections::{HashMap, HashSet};
use std::fmt;

// =============================================================================
// Trait Checker
// =============================================================================

/// Trait declaration and implementation checker
#[derive(Debug)]
pub struct TraitChecker {
    /// Registered trait declarations
    pub traits: HashMap<String, TraitDeclTy>,

    /// Registered trait implementations
    pub impls: TraitImplIndex,

    /// Errors accumulated during checking
    pub errors: Vec<TraitCheckError>,
}

impl TraitChecker {
    pub fn new() -> Self {
        Self {
            traits: HashMap::new(),
            impls: TraitImplIndex::new(),
            errors: Vec::new(),
        }
    }

    /// Create a trait resolver for method resolution
    pub fn resolver(&self) -> TraitResolver<'_> {
        TraitResolver::new(&self.traits, &self.impls)
    }

    // =========================================================================
    // Trait Declaration Checking
    // =========================================================================

    /// Check and register a trait declaration
    pub fn check_trait_decl(&mut self, decl: &TraitDecl) -> Result<(), TraitCheckError> {
        // Check for duplicate trait
        if self.traits.contains_key(&decl.name) {
            return Err(TraitCheckError::DuplicateTrait {
                name: decl.name.clone(),
            });
        }

        // Check super-traits exist
        for super_trait in &decl.super_traits {
            self.check_trait_ref_exists(super_trait)?;
        }

        // Check for duplicate methods
        let mut method_names = HashSet::new();
        for method in &decl.methods {
            if !method_names.insert(&method.name) {
                return Err(TraitCheckError::DuplicateMethod {
                    trait_name: decl.name.clone(),
                    method_name: method.name.clone(),
                });
            }
        }

        // Convert to type-level representation
        let trait_ty = self.lower_trait_decl(decl);

        // Register
        self.traits.insert(decl.name.clone(), trait_ty);

        Ok(())
    }

    /// Convert AST trait decl to type-level representation
    fn lower_trait_decl(&self, decl: &TraitDecl) -> TraitDeclTy {
        let mut trait_ty = TraitDeclTy::new(&decl.name);

        // Type parameters
        for tp in &decl.type_params {
            trait_ty.add_type_param(&tp.name);
        }

        // Super-traits
        for st in &decl.super_traits {
            trait_ty.add_super_trait(self.lower_trait_ref(st));
        }

        // Methods
        for method in &decl.methods {
            let sig = self.lower_method_sig(method);
            trait_ty.add_method(&method.name, sig);
        }

        trait_ty
    }

    /// Lower a TraitRef to TraitRefTy
    fn lower_trait_ref(&self, tr: &TraitRef) -> TraitRefTy {
        TraitRefTy::new(&tr.trait_name, tr.type_args.clone())
    }

    /// Lower a TraitMethod to MethodSig
    fn lower_method_sig(&self, method: &TraitMethod) -> MethodSig {
        MethodSig {
            type_params: method
                .type_params
                .iter()
                .map(|tp| tp.name.clone())
                .collect(),
            param_names: method.params.iter().map(|p| p.name.clone()).collect(),
            param_types: method.params.iter().map(|p| p.ty.clone()).collect(),
            ret_type: method.ret_type.clone(),
        }
    }

    /// Check that a trait reference refers to an existing trait
    fn check_trait_ref_exists(&self, tr: &TraitRef) -> Result<(), TraitCheckError> {
        if !self.traits.contains_key(&tr.trait_name) {
            return Err(TraitCheckError::UnknownTrait {
                name: tr.trait_name.clone(),
            });
        }
        Ok(())
    }

    // =========================================================================
    // Trait Implementation Checking
    // =========================================================================

    /// Check and register a trait implementation
    pub fn check_trait_impl(&mut self, impl_: &TraitImpl) -> Result<(), TraitCheckError> {
        // Get the trait declaration
        let trait_decl = self
            .traits
            .get(&impl_.trait_ref.trait_name)
            .cloned()
            .ok_or_else(|| TraitCheckError::UnknownTrait {
                name: impl_.trait_ref.trait_name.clone(),
            })?;

        // Check for duplicate impl
        if self
            .impls
            .find_impl(&impl_.trait_ref.trait_name, &impl_.trait_ref.type_args)
            .is_some()
        {
            return Err(TraitCheckError::DuplicateImpl {
                trait_ref: impl_.trait_ref.to_string(),
            });
        }

        // Check all required methods are implemented
        self.check_required_methods(impl_, &trait_decl)?;

        // Check no extra methods are provided
        self.check_no_extra_methods(impl_, &trait_decl)?;

        // Check method signatures match
        for method in &impl_.methods {
            self.check_method_signature(impl_, method, &trait_decl)?;
        }

        // Convert to type-level and register
        let impl_ty = self.lower_trait_impl(impl_);
        self.impls.add(impl_ty);

        Ok(())
    }

    /// Check that all required methods are provided
    fn check_required_methods(
        &self,
        impl_: &TraitImpl,
        trait_decl: &TraitDeclTy,
    ) -> Result<(), TraitCheckError> {
        let provided: HashSet<_> = impl_.methods.iter().map(|m| m.name.as_str()).collect();

        for (method_name, _) in &trait_decl.methods {
            if !provided.contains(method_name.as_str()) {
                return Err(TraitCheckError::MissingMethod {
                    trait_name: impl_.trait_ref.trait_name.clone(),
                    method_name: method_name.clone(),
                });
            }
        }

        Ok(())
    }

    /// Check that no extra methods are provided
    fn check_no_extra_methods(
        &self,
        impl_: &TraitImpl,
        trait_decl: &TraitDeclTy,
    ) -> Result<(), TraitCheckError> {
        for method in &impl_.methods {
            if !trait_decl.methods.contains_key(&method.name) {
                return Err(TraitCheckError::ExtraMethod {
                    trait_name: impl_.trait_ref.trait_name.clone(),
                    method_name: method.name.clone(),
                });
            }
        }

        Ok(())
    }

    /// Check that a method signature matches the trait's expected signature
    fn check_method_signature(
        &self,
        impl_: &TraitImpl,
        method: &ImplMethod,
        trait_decl: &TraitDeclTy,
    ) -> Result<(), TraitCheckError> {
        let expected_sig = trait_decl.methods.get(&method.name).unwrap();

        // Build substitution from trait type params to impl type args
        let subst = self.build_type_substitution(trait_decl, impl_);

        // Apply substitution to expected signature
        let concrete_sig = expected_sig.substitute(&subst);

        // Check arity
        if method.params.len() != concrete_sig.param_types.len() {
            return Err(TraitCheckError::SignatureMismatch {
                trait_name: impl_.trait_ref.trait_name.clone(),
                method_name: method.name.clone(),
                expected: format!("{} parameters", concrete_sig.param_types.len()),
                found: format!("{} parameters", method.params.len()),
            });
        }

        // Check parameter types
        for (i, (param, expected_ty)) in method
            .params
            .iter()
            .zip(&concrete_sig.param_types)
            .enumerate()
        {
            if param.ty != *expected_ty {
                return Err(TraitCheckError::SignatureMismatch {
                    trait_name: impl_.trait_ref.trait_name.clone(),
                    method_name: method.name.clone(),
                    expected: format!("parameter {} type {}", i, expected_ty),
                    found: format!("parameter {} type {}", i, param.ty),
                });
            }
        }

        // Check return type
        if method.ret_type != concrete_sig.ret_type {
            return Err(TraitCheckError::SignatureMismatch {
                trait_name: impl_.trait_ref.trait_name.clone(),
                method_name: method.name.clone(),
                expected: format!("return type {}", concrete_sig.ret_type),
                found: format!("return type {}", method.ret_type),
            });
        }

        Ok(())
    }

    /// Build type substitution from trait params to impl args
    fn build_type_substitution(
        &self,
        trait_decl: &TraitDeclTy,
        impl_: &TraitImpl,
    ) -> HashMap<String, String> {
        let mut subst = HashMap::new();

        for (param, arg) in trait_decl
            .type_params
            .iter()
            .zip(&impl_.trait_ref.type_args)
        {
            subst.insert(param.clone(), arg.clone());
        }

        subst
    }

    /// Convert AST impl to type-level representation
    fn lower_trait_impl(&self, impl_: &TraitImpl) -> TraitImplTy {
        let mut impl_ty = TraitImplTy::new(self.lower_trait_ref(&impl_.trait_ref));

        // Type parameters
        for tp in &impl_.type_params {
            let mut param = TypeParamTy::new(&tp.name);
            for bound in &tp.bounds {
                param = param.with_bound(self.lower_trait_ref(bound));
            }
            impl_ty.add_type_param(param);
        }

        // Where clauses
        for wc in &impl_.where_clauses {
            impl_ty.where_clauses.push(WhereClauseTy::new(
                &wc.type_param,
                wc.bounds.iter().map(|b| self.lower_trait_ref(b)).collect(),
            ));
        }

        // Methods with mangled symbols
        for method in &impl_.methods {
            let symbol = super::lower::mangle_trait_method_symbol(
                &impl_.trait_ref.trait_name,
                &impl_.trait_ref.type_args,
                &method.name,
            );

            let sig = MethodSig {
                type_params: method
                    .type_params
                    .iter()
                    .map(|tp| tp.name.clone())
                    .collect(),
                param_names: method.params.iter().map(|p| p.name.clone()).collect(),
                param_types: method.params.iter().map(|p| p.ty.clone()).collect(),
                ret_type: method.ret_type.clone(),
            };

            impl_ty.add_method(&method.name, &symbol, sig);
        }

        // Type bindings
        for binding in &impl_.type_bindings {
            impl_ty.add_type_binding(&binding.name, &binding.ty);
        }

        impl_ty
    }

    // =========================================================================
    // Bounded Generics Checking
    // =========================================================================

    /// Check that a type satisfies all bounds
    pub fn check_bounds(&self, ty: &str, bounds: &[TraitRef]) -> Result<(), TraitCheckError> {
        let resolver = self.resolver();

        for bound in bounds {
            let bound_ty = self.lower_trait_ref(bound);
            if !resolver.satisfies_bound(ty, &bound_ty) {
                return Err(TraitCheckError::BoundNotSatisfied {
                    ty: ty.to_string(),
                    bound: bound.to_string(),
                });
            }
        }

        Ok(())
    }

    /// Get all methods available for a type via its trait bounds
    pub fn methods_for_bounded_type(
        &self,
        bounds: &[TraitRef],
    ) -> HashMap<String, (String, MethodSig)> {
        let mut methods = HashMap::new();

        for bound in bounds {
            if let Some(trait_decl) = self.traits.get(&bound.trait_name) {
                for (name, sig) in &trait_decl.methods {
                    // Build substitution
                    let mut subst = HashMap::new();
                    for (param, arg) in trait_decl.type_params.iter().zip(&bound.type_args) {
                        subst.insert(param.clone(), arg.clone());
                    }

                    // Store with trait name for disambiguation
                    let key = format!("{}::{}", bound.trait_name, name);
                    methods.insert(key, (bound.trait_name.clone(), sig.substitute(&subst)));
                }
            }
        }

        methods
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    /// Get a trait by name
    pub fn get_trait(&self, name: &str) -> Option<&TraitDeclTy> {
        self.traits.get(name)
    }

    /// Check if a trait exists
    pub fn has_trait(&self, name: &str) -> bool {
        self.traits.contains_key(name)
    }

    /// Get all registered traits
    pub fn all_traits(&self) -> &HashMap<String, TraitDeclTy> {
        &self.traits
    }

    /// Get all registered impls
    pub fn all_impls(&self) -> &TraitImplIndex {
        &self.impls
    }

    /// Clear all errors
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Check if there are errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

impl Default for TraitChecker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, Clone)]
pub enum TraitCheckError {
    DuplicateTrait {
        name: String,
    },
    UnknownTrait {
        name: String,
    },
    DuplicateImpl {
        trait_ref: String,
    },
    DuplicateMethod {
        trait_name: String,
        method_name: String,
    },
    MissingMethod {
        trait_name: String,
        method_name: String,
    },
    ExtraMethod {
        trait_name: String,
        method_name: String,
    },
    SignatureMismatch {
        trait_name: String,
        method_name: String,
        expected: String,
        found: String,
    },
    BoundNotSatisfied {
        ty: String,
        bound: String,
    },
    CyclicSuperTrait {
        trait_name: String,
    },
}

impl fmt::Display for TraitCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraitCheckError::DuplicateTrait { name } => {
                write!(f, "duplicate trait declaration: {}", name)
            }
            TraitCheckError::UnknownTrait { name } => {
                write!(f, "unknown trait: {}", name)
            }
            TraitCheckError::DuplicateImpl { trait_ref } => {
                write!(f, "duplicate implementation for {}", trait_ref)
            }
            TraitCheckError::DuplicateMethod {
                trait_name,
                method_name,
            } => {
                write!(
                    f,
                    "duplicate method {} in trait {}",
                    method_name, trait_name
                )
            }
            TraitCheckError::MissingMethod {
                trait_name,
                method_name,
            } => {
                write!(
                    f,
                    "missing method '{}' in impl of {}",
                    method_name, trait_name
                )
            }
            TraitCheckError::ExtraMethod {
                trait_name,
                method_name,
            } => {
                write!(
                    f,
                    "method '{}' is not defined in trait {}",
                    method_name, trait_name
                )
            }
            TraitCheckError::SignatureMismatch {
                trait_name,
                method_name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "signature mismatch for {}::{}: expected {}, found {}",
                    trait_name, method_name, expected, found
                )
            }
            TraitCheckError::BoundNotSatisfied { ty, bound } => {
                write!(f, "type {} does not satisfy bound {}", ty, bound)
            }
            TraitCheckError::CyclicSuperTrait { trait_name } => {
                write!(f, "cyclic super-trait dependency in {}", trait_name)
            }
        }
    }
}

impl std::error::Error for TraitCheckError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ast::*;

    #[test]
    fn test_check_simple_trait_decl() {
        let mut checker = TraitChecker::new();

        let decl = TraitDecl::new("Eq").with_type_param("T").with_method(
            TraitMethod::new("eq")
                .with_param("x", "T")
                .with_param("y", "T")
                .with_ret_type("Bool"),
        );

        let result = checker.check_trait_decl(&decl);
        assert!(result.is_ok());
        assert!(checker.has_trait("Eq"));
    }

    #[test]
    fn test_duplicate_trait_error() {
        let mut checker = TraitChecker::new();

        let decl1 = TraitDecl::new("Eq").with_type_param("T");
        let decl2 = TraitDecl::new("Eq").with_type_param("T");

        assert!(checker.check_trait_decl(&decl1).is_ok());
        assert!(matches!(
            checker.check_trait_decl(&decl2),
            Err(TraitCheckError::DuplicateTrait { .. })
        ));
    }

    #[test]
    fn test_check_simple_impl() {
        let mut checker = TraitChecker::new();

        // Register trait
        let trait_decl = TraitDecl::new("Eq").with_type_param("T").with_method(
            TraitMethod::new("eq")
                .with_param("x", "T")
                .with_param("y", "T")
                .with_ret_type("Bool"),
        );
        checker.check_trait_decl(&trait_decl).unwrap();

        // Register impl
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Eq", "Float")).with_method_body(
            "eq",
            vec![("x", "Float"), ("y", "Float")],
            "Bool",
            "x == y",
        );

        let result = checker.check_trait_impl(&impl_);
        assert!(result.is_ok());
    }

    #[test]
    fn test_missing_method_error() {
        let mut checker = TraitChecker::new();

        // Register trait with two methods
        let trait_decl = TraitDecl::new("Numeric")
            .with_type_param("T")
            .with_method(TraitMethod::new("zero").with_ret_type("T"))
            .with_method(
                TraitMethod::new("add")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            );
        checker.check_trait_decl(&trait_decl).unwrap();

        // Impl with missing method
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float")).with_method_body(
            "zero",
            vec![],
            "Float",
            "0.0",
        );

        let result = checker.check_trait_impl(&impl_);
        assert!(matches!(result, Err(TraitCheckError::MissingMethod { .. })));
    }

    #[test]
    fn test_signature_mismatch_error() {
        let mut checker = TraitChecker::new();

        // Register trait
        let trait_decl = TraitDecl::new("Numeric").with_type_param("T").with_method(
            TraitMethod::new("add")
                .with_param("x", "T")
                .with_param("y", "T")
                .with_ret_type("T"),
        );
        checker.check_trait_decl(&trait_decl).unwrap();

        // Impl with wrong return type
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float")).with_method_body(
            "add",
            vec![("x", "Float"), ("y", "Float")],
            "Int",
            "0",
        );

        let result = checker.check_trait_impl(&impl_);
        assert!(matches!(
            result,
            Err(TraitCheckError::SignatureMismatch { .. })
        ));
    }

    #[test]
    fn test_trait_with_super_trait() {
        let mut checker = TraitChecker::new();

        // Register Eq trait first
        let eq_decl = TraitDecl::new("Eq").with_type_param("T").with_method(
            TraitMethod::new("eq")
                .with_param("x", "T")
                .with_param("y", "T")
                .with_ret_type("Bool"),
        );
        checker.check_trait_decl(&eq_decl).unwrap();

        // Register Ord trait with Eq as super-trait
        let ord_decl = TraitDecl::new("Ord")
            .with_type_param("T")
            .with_super_trait(TraitRef::with_single_arg("Eq", "T"))
            .with_method(
                TraitMethod::new("lt")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("Bool"),
            );

        let result = checker.check_trait_decl(&ord_decl);
        assert!(result.is_ok());

        let ord_ty = checker.get_trait("Ord").unwrap();
        assert!(ord_ty.has_super_trait("Eq"));
    }

    #[test]
    fn test_bound_checking() {
        let mut checker = TraitChecker::new();

        // Register trait
        let decl = TraitDecl::new("Numeric")
            .with_type_param("T")
            .with_method(TraitMethod::new("zero").with_ret_type("T"));
        checker.check_trait_decl(&decl).unwrap();

        // Register impl for Float
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float")).with_method_body(
            "zero",
            vec![],
            "Float",
            "0.0",
        );
        checker.check_trait_impl(&impl_).unwrap();

        // Float satisfies Numeric, Int does not
        assert!(checker
            .check_bounds("Float", &[TraitRef::simple("Numeric")])
            .is_ok());
        assert!(checker
            .check_bounds("Int", &[TraitRef::simple("Numeric")])
            .is_err());
    }
}
