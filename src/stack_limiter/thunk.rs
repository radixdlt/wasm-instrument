use super::Context;
use crate::parser::{
	translator::{ConstExprKind, DefaultTranslator, Translator},
	ModuleInfo,
};
#[cfg(not(features = "std"))]
use alloc::collections::BTreeMap as Map;
use alloc::vec::Vec;
#[cfg(features = "std")]
use std::collections::HashMap as Map;
use wasm_encoder::{
	CodeSection, ElementMode, ElementSection, Elements, ExportSection, FunctionSection, SectionId,
};
use wasmparser::{
	CodeSectionReader, ElementItem, ElementItems, ElementKind, ElementSectionReader,
	ExportSectionReader, ExternalKind, FuncType, FunctionSectionReader, Result as WasmParserResult,
	Type,
};

struct Thunk {
	signature: FuncType,
	// Index in function space of this thunk.
	idx: Option<u32>,
	callee_stack_cost: u32,
}

pub fn generate_thunks(ctx: &mut Context, module: &mut ModuleInfo) -> Result<(), &'static str> {
	// First, we need to collect all function indices that should be replaced by thunks
	let exports = match module.raw_sections.get(&SectionId::Export.into()) {
		Some(raw_sec) => ExportSectionReader::new(&raw_sec.data, 0)
			.map_err(|err| stringify!(err))?
			.into_iter()
			.collect::<WasmParserResult<Vec<wasmparser::Export>>>()
			.map_err(|err| stringify!(err))?,
		None => vec![],
	};

	//element maybe null
	let elem_segments = match module.raw_sections.get(&SectionId::Element.into()) {
		Some(v) => ElementSectionReader::new(&v.data, 0)
			.map_err(|err| stringify!(err))?
			.into_iter()
			.collect::<WasmParserResult<Vec<wasmparser::Element>>>()
			.map_err(|err| stringify!(err))?,
		None => vec![],
	};

	let mut replacement_map: Map<u32, Thunk> = {
		let exported_func_indices = exports.iter().filter_map(|entry| match entry.kind {
			ExternalKind::Func => Some(entry.index),
			_ => None,
		});

		let mut table_func_indices = vec![];
		for segment in elem_segments.clone() {
			match segment.items {
				ElementItems::Functions(func_indexes) => {
					let segment_func_indices = &func_indexes
						.into_iter()
						.map(|item| match item {
							Ok(idx) => Ok(idx),
							Err(err) => Err(stringify!(err)),
						})
						.collect::<Result<Vec<u32>, &'static str>>()?;

					table_func_indices.extend_from_slice(segment_func_indices);
				},
				ElementItems::Expressions(_) => return Err("never exec here"),
			}
		}

		// Replacement map is at least export section size.
		let mut replacement_map: Map<u32, Thunk> = Map::new();

		for func_idx in exported_func_indices
			.chain(table_func_indices)
			.chain(module.start_function.into_iter())
		{
			let callee_stack_cost = ctx
				.stack_cost(func_idx)
				.ok_or_else(|| stringify!("function index isn't found"))?;

			// Don't generate a thunk if stack_cost of a callee is zero.
			if callee_stack_cost != 0 {
				replacement_map.insert(
					func_idx,
					Thunk {
						signature: match module.get_functype_idx(func_idx)?.clone() {
							Type::Func(ft) => ft,
						},
						idx: None,
						callee_stack_cost,
					},
				);
			}
		}

		replacement_map
	};

	// Then, we generate a thunk for each original function.

	// Save current func_idx
	let mut func_body_sec_builder = CodeSection::new();
	let func_body_sec_data = &module
		.raw_sections
		.get(&SectionId::Code.into())
		.ok_or_else(|| stringify!("no function body"))?
		.data;

	let code_sec_reader =
		CodeSectionReader::new(func_body_sec_data, 0).map_err(|err| stringify!(err))?;
	for func_body in code_sec_reader {
		let func_body = func_body.map_err(|err| stringify!(err))?;
		DefaultTranslator
			.translate_code(func_body, &mut func_body_sec_builder)
			.map_err(|err| stringify!(err))?;
	}

	let mut func_sec_builder = FunctionSection::new();
	let func_sec_data = &module
		.raw_sections
		.get(&SectionId::Function.into())
		.ok_or_else(|| anyhow!("no function section"))? //todo allow empty function file?
		.data;
	for func_body in FunctionSectionReader::new(func_sec_data, 0)? {
		func_sec_builder.function(func_body?);
	}

	let mut next_func_idx = module.function_map.len() as u32;
	for (func_idx, thunk) in replacement_map.iter_mut() {
		// Thunk body consist of:
		//  - argument pushing
		//  - instrumented call
		//  - end
		let mut thunk_body = wasm_encoder::Function::new(None);

		for (arg_idx, _) in thunk.signature.params().iter().enumerate() {
			thunk_body.instruction(&wasm_encoder::Instruction::LocalGet(arg_idx as u32));
		}

		instrument_call!(
			*func_idx,
			thunk.callee_stack_cost as i32,
			ctx.stack_height_global_idx(),
			ctx.stack_limit()
		)
		.iter()
		.for_each(|v| {
			thunk_body.instruction(v);
		});
		thunk_body.instruction(&wasm_encoder::Instruction::End);

		let func_type = module
			.resolve_type_idx(&Type::Func(thunk.signature.clone()))
			.ok_or_else(|| anyhow!("signature not exit"))?; //resolve thunk func type, this signature should exit
		func_sec_builder.function(func_type); //add thunk function
		func_body_sec_builder.function(&thunk_body); //add thunk body

		thunk.idx = Some(next_func_idx);
		next_func_idx += 1;
	}

	// And finally, fixup thunks in export and table sections.

	// Fixup original function index to a index of a thunk generated earlier.
	let fixup = |function_idx: &mut u32| {
		// Check whether this function is in replacement_map, since
		// we can skip thunk generation (e.g. if stack_cost of function is 0).
		if let Some(thunk) = replacement_map.get(function_idx) {
			*function_idx =
				thunk.idx.expect("At this point an index must be assigned to each thunk");
		}
	};

	for section in module.sections_mut() {
		match section {
			elements::Section::Export(export_section) => {
				for entry in export_section.entries_mut() {
					if let Internal::Function(function_idx) = entry.internal_mut() {
						fixup(function_idx)
					}
				}
			},
			elements::Section::Element(elem_section) => {
				for segment in elem_section.entries_mut() {
					for function_idx in segment.members_mut() {
						fixup(function_idx)
					}
				}
			},
			elements::Section::Start(start_idx) => fixup(start_idx),
			_ => {},
		}
	}

	Ok(module)
}
