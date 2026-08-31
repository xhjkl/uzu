use std::collections::{BTreeMap, BTreeSet};

#[cfg(all(feature = "metal", target_os = "macos"))]
use rhai::EvalAltResult;
use rhai::{AST, Dynamic, Engine, Module, Scope};

pub fn unqualify_variant(value: &str) -> &str {
    value.rsplit("::").next().unwrap_or(value)
}

struct Constraint {
    source: Box<str>,
    ast: AST,
}

pub struct Constraints {
    engine: Engine,
    constraints: Box<[Constraint]>,
}

impl Constraints {
    pub fn new(
        variant_values: impl IntoIterator<Item = impl AsRef<str>>,
        constraints: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Self {
        let mut engine = Engine::new();
        let mut namespaces: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for value in variant_values {
            if let Some((namespace, name)) = value.as_ref().rsplit_once("::") {
                namespaces.entry(namespace.to_owned()).or_default().insert(name.to_owned());
            }
        }
        for (namespace, names) in namespaces {
            let mut module = Module::new();
            for name in names {
                module.set_var(name.clone(), name);
            }
            engine.register_static_module(namespace, module.into());
        }
        let constraints = constraints
            .into_iter()
            .map(|constraint| {
                let source: Box<str> = constraint.as_ref().into();
                let ast = engine
                    .compile_expression(&source)
                    .unwrap_or_else(|error| panic!("constraint `{source}` failed to compile: {error}"));
                Constraint {
                    source,
                    ast,
                }
            })
            .collect();
        Self {
            engine,
            constraints,
        }
    }

    fn scope<N, V>(
        &self,
        bindings: impl IntoIterator<Item = (N, V)>,
    ) -> Scope<'static>
    where
        N: AsRef<str>,
        V: AsRef<str>,
    {
        let bindings = bindings.into_iter();
        let mut scope = Scope::with_capacity(bindings.size_hint().0);
        for (name, val) in bindings {
            let name = name.as_ref();
            let val = unqualify_variant(val.as_ref());
            scope.push(
                name.to_owned(),
                self.engine.eval_expression::<Dynamic>(val).unwrap_or_else(|_| val.to_owned().into()),
            );
        }
        scope
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn could_satisfy<N, V>(
        &self,
        bindings: impl IntoIterator<Item = (N, V)>,
    ) -> bool
    where
        N: AsRef<str>,
        V: AsRef<str>,
    {
        let mut scope = self.scope(bindings);
        self.constraints.iter().all(|constraint| {
            match self.engine.eval_ast_with_scope::<bool>(&mut scope, &constraint.ast) {
                Ok(satisfied) => satisfied,
                Err(error) if matches!(error.as_ref(), EvalAltResult::ErrorVariableNotFound(..)) => true,
                Err(error) => panic!("constraint `{}` failed to evaluate: {error}", constraint.source),
            }
        })
    }

    pub fn satisfied<N, V>(
        &self,
        bindings: impl IntoIterator<Item = (N, V)>,
    ) -> bool
    where
        N: AsRef<str>,
        V: AsRef<str>,
    {
        let mut scope = self.scope(bindings);
        self.constraints.iter().all(|constraint| {
            self.engine
                .eval_ast_with_scope::<bool>(&mut scope, &constraint.ast)
                .unwrap_or_else(|error| panic!("constraint `{}` failed to evaluate: {error}", constraint.source))
        })
    }
}
