use std::collections::{BTreeMap, BTreeSet};

#[cfg(all(feature = "metal", target_os = "macos"))]
use rhai::EvalAltResult;
use rhai::{AST, Dynamic, Engine, Module, Scope};

fn unqualify_variant(value: &str) -> &str {
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
    pub fn new<'a>(
        variant_values: impl IntoIterator<Item = &'a str>,
        constraints: &[impl AsRef<str>],
    ) -> Self {
        let mut engine = Engine::new();
        let mut namespaces: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for value in variant_values {
            if let Some((namespace, name)) = value.rsplit_once("::") {
                namespaces.entry(namespace).or_default().insert(name);
            }
        }
        for (namespace, names) in namespaces {
            let mut module = Module::new();
            for name in names {
                module.set_var(name, name.to_string());
            }
            engine.register_static_module(namespace, module.into());
        }
        let constraints = constraints
            .iter()
            .map(|constraint| {
                let source = constraint.as_ref();
                let ast = engine
                    .compile_expression(source)
                    .unwrap_or_else(|error| panic!("constraint `{source}` failed to compile: {error}"));
                Constraint {
                    source: source.into(),
                    ast,
                }
            })
            .collect();
        Self {
            engine,
            constraints,
        }
    }

    fn scope<N: AsRef<str>, V: AsRef<str>>(
        &self,
        bindings: &[(N, V)],
    ) -> Scope<'static> {
        let mut scope = Scope::with_capacity(bindings.len());
        for (name, val) in bindings {
            let val = unqualify_variant(val.as_ref());
            scope.push(
                name.as_ref().to_owned(),
                self.engine.eval_expression::<Dynamic>(val).unwrap_or_else(|_| val.to_owned().into()),
            );
        }
        scope
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn could_satisfy<N: AsRef<str>, V: AsRef<str>>(
        &self,
        bindings: &[(N, V)],
    ) -> bool {
        let mut scope = self.scope(bindings);
        self.constraints.iter().all(|constraint| {
            match self.engine.eval_ast_with_scope::<bool>(&mut scope, &constraint.ast) {
                Ok(satisfied) => satisfied,
                Err(error) if matches!(error.as_ref(), EvalAltResult::ErrorVariableNotFound(..)) => true,
                Err(error) => panic!("constraint `{}` failed to evaluate: {error}", constraint.source),
            }
        })
    }

    pub fn satisfied<N: AsRef<str>, V: AsRef<str>>(
        &self,
        bindings: &[(N, V)],
    ) -> bool {
        let mut scope = self.scope(bindings);
        self.constraints.iter().all(|constraint| {
            self.engine
                .eval_ast_with_scope::<bool>(&mut scope, &constraint.ast)
                .unwrap_or_else(|error| panic!("constraint `{}` failed to evaluate: {error}", constraint.source))
        })
    }
}
