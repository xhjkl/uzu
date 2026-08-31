use quote::ToTokens;
use syn::{
    Attribute, Expr, Ident, Token, Visibility, braced,
    parse::{Parse, ParseStream},
};

#[derive(Debug)]
pub struct GpuTypeOptionSet {
    pub name: String,
    pub underlying_type: String,
    pub variants: Vec<(String, String)>,
}

impl Parse for GpuTypeOptionSet {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let _attrs = input.call(Attribute::parse_outer)?;
        let _vis: Visibility = input.parse()?;
        let _struct_kw: Token![struct] = input.parse()?;
        let name: Ident = input.parse()?;
        let _colon: Token![:] = input.parse()?;
        let underlying: Ident = input.parse()?;

        let body;
        braced!(body in input);

        let mut variants = Vec::new();
        while !body.is_empty() {
            let _const_kw: Token![const] = body.parse()?;
            let name: Ident = body.parse()?;
            let _equals: Token![=] = body.parse()?;
            let expression: Expr = body.parse()?;
            let _semi: Token![;] = body.parse()?;
            variants.push((name.to_string(), expression.into_token_stream().to_string()));
        }

        Ok(Self {
            name: name.to_string(),
            underlying_type: underlying.to_string(),
            variants,
        })
    }
}
