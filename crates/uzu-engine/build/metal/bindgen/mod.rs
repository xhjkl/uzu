mod arguments;
mod dispatch;
mod host_expression_rewriter;
mod specialize;
mod variants;

use anyhow::Result;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use self::host_expression_rewriter::HostExpressionRewriter;
use super::{ast::MetalKernelInfo, wrapper::SpecializeBaseIndices};
use crate::common::{enum_paths::EnumPaths, kernel::Kernel, mangling::dynamic_mangle};

pub fn bindgen(
    kernel: &MetalKernelInfo,
    specialize_indices: &SpecializeBaseIndices,
    enum_paths: &EnumPaths,
    library_const: &proc_macro2::Ident,
    num_shards: usize,
    library_compressed: bool,
) -> Result<TokenStream> {
    let kernel_name = kernel.name.as_ref();
    let trait_name = format_ident!("{}Kernel", kernel_name);
    let struct_name = format_ident!("{}MetalKernel", kernel_name);

    let variant_binds = variants::parse(kernel)?;
    let specialize_emission =
        specialize::parse(kernel, specialize_indices.get(&kernel.name).copied(), kernel_name, enum_paths)?;
    let mut host_expression_rewriter =
        HostExpressionRewriter::new(&variant_binds, enum_paths, &specialize_emission, kernel_name);
    let argument_emissions = arguments::parse(kernel, enum_paths, &mut host_expression_rewriter)?;

    let (dispatch_code, empty_dispatch_guards) = dispatch::parse(kernel, &mut host_expression_rewriter)?;
    let referenced_parameter_names = host_expression_rewriter.finish();

    let (conditional_buffer_fields, conditional_buffer_initializers): (Vec<TokenStream>, Vec<TokenStream>) =
        argument_emissions.iter().filter_map(|argument| argument.struct_parts()).unzip();
    let mut encode_argument_definitions: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.encode_argument_definition()).collect();
    let mut encode_lifetimes: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.encode_lifetime()).collect();
    let encode_deconstructs: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.encode_deconstruct()).collect();
    let encode_set_calls: Vec<TokenStream> = argument_emissions.iter().map(|argument| argument.encode_set()).collect();
    let encode_accesses_call = arguments::encode_accesses_call(&argument_emissions);

    let (variant_struct_fields, variant_struct_initializers): (Vec<TokenStream>, Vec<TokenStream>) =
        variant_binds.iter().filter_map(|variant| variant.struct_parts(&referenced_parameter_names)).unzip();
    let variant_constructor_arguments: Vec<TokenStream> =
        variant_binds.iter().map(|variant| variant.constructor_argument()).collect();
    let variant_kernel_format: Vec<TokenStream> = variant_binds.iter().map(|variant| variant.kernel_format()).collect();
    let entry_name = dynamic_mangle(kernel_name, variant_kernel_format);

    let specialize_arguments = specialize_emission.constructor_arguments();
    let (retained_specialization_fields, retained_specialization_initializers) =
        specialize_emission.retain_referenced(&referenced_parameter_names);
    let function_constants_initialization = specialize_emission.function_constants_initialization();
    let function_constants_argument = specialize_emission.function_constants_argument();
    let cache_key = specialize_emission.cache_key();

    let library_data = if num_shards == 1 {
        quote! { #library_const[0] }
    } else {
        let num_shards = num_shards as u64;
        quote! { #library_const[(xxhash_rust::xxh3::xxh3_64(entry_name.as_bytes()) % #num_shards) as usize] }
    };

    let (trait_implementation_for, associate_backend, method_visibility) = if kernel.public {
        (
            quote! { crate::backends::common::kernel::#trait_name for },
            quote! { type Backend = crate::backends::metal::Metal; },
            quote! {},
        )
    } else {
        (quote! {}, quote! {}, quote! { pub(crate) })
    };

    encode_lifetimes.push(quote! { 'encoder });
    encode_argument_definitions.push(quote! {
        encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::metal::Metal>
    });

    let kernel_tokens = quote! {
        pub struct #struct_name {
            pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
            #(#conditional_buffer_fields,)*
            #(#variant_struct_fields,)*
            #(#retained_specialization_fields,)*
        }

        #[allow(clippy::style, clippy::complexity, clippy::perf)]
        impl #trait_implementation_for #struct_name {
            #associate_backend

            #method_visibility fn new(
                context: &MetalContext
                #(, #variant_constructor_arguments)*
                #(, #specialize_arguments)*
            ) -> Result<Self, MetalError> {
                let entry_name = #entry_name;
                #function_constants_initialization
                let pipeline = context.compute_pipeline_state(#library_data, #library_compressed, #cache_key, &entry_name, #function_constants_argument)?;
                Ok(Self {
                    pipeline
                    #(, #conditional_buffer_initializers)*
                    #(, #variant_struct_initializers)*
                    #(, #retained_specialization_initializers)*
                })
            }

            #method_visibility fn encode<#(#encode_lifetimes),*>(
                &self,
                #(#encode_argument_definitions),*
            ) {
                #empty_dispatch_guards
                encoder.push_debug_group(#kernel_name);
                #(#encode_deconstructs)*
                #encode_accesses_call
                let compute_encoder = encoder.as_command_buffer_mut().ensure_compute();
                compute_encoder.set_compute_pipeline_state(&self.pipeline);
                #(#encode_set_calls)*
                #dispatch_code
                encoder.pop_debug_group();
            }
        }
    };

    Ok(kernel_tokens)
}

pub fn bindgen_global(kernels: &[(impl AsRef<std::path::Path>, &[Kernel])]) -> Result<TokenStream> {
    let includes = kernels.iter().map(|(path, _kernels)| {
        let path = path.as_ref().to_str().expect("bindings path is not utf-8");

        quote! {
            include!(#path);
        }
    });

    let associated_types = kernels.iter().flat_map(|(_path, kernels)| kernels.iter()).map(|kernel| {
        let trait_name = format_ident!("{}Kernel", kernel.name.as_ref());
        let struct_name = format_ident!("{}MetalKernel", kernel.name.as_ref());

        quote! {
            type #trait_name = #struct_name;
        }
    });

    let tokens = quote! {
        use metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLFunctionConstantValues, MTLSize};
        use objc2::{rc::Retained, runtime::ProtocolObject};

        use crate::backends::common::BufferGpuAddressRangeExt;
        use crate::backends::metal::{
            context::MetalContext,
            error::MetalError,
            metal_extensions::{
                ComputeEncoderSetValue, FunctionConstantValuesSetValue, MetalDataTypeExt,
            },
        };

        #(#includes)*

        macro_rules! autogen_kernels {
            () => {
                #(#associated_types)*
            }
        }
    };

    Ok(tokens)
}
