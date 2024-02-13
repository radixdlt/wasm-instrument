use super::Context;
use crate::utils::{
	module_info::ModuleInfo,
	translator::{ConstExprKind, DefaultTranslator, Translator},
};
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as Map;
use alloc::{vec, vec::Vec};
use anyhow::{anyhow, Result};
#[cfg(feature = "std")]
use std::collections::BTreeMap as Map;
use wasm_encoder::{
	ElementMode, ElementSection, ElementSegment, Elements, ExportSection, SectionId,
};
use wasmparser::{ElementItems, ElementKind, ExternalKind, FuncType, Type};

struct Thunk {
	signature: FuncType,
	// Index in function space of this thunk.
	idx: Option<u32>,
	callee_stack_cost: u32,
}

pub fn generate_thunks(ctx: &mut Context, module_info: &mut ModuleInfo) -> Result<()> {
	// First, we need to collect all function indices that should be replaced by thunks

	let mut replacement_map: Map<u32, Thunk> = {
		let elements = module_info.element_section()?.unwrap_or(vec![]);
		let exports = module_info.export_section()?.unwrap_or(vec![]);
		let exported_func_indices = exports.iter().filter_map(|entry| match entry.kind {
			ExternalKind::Func => Some(entry.index),
			_ => None,
		});

		let mut table_func_indices = vec![];
		for elem in elements {
			match elem.items {
				ElementItems::Functions(func_indexes) => {
					let segment_func_indices: Vec<u32> = func_indexes
						.into_iter()
						.map(|item| match item {
							Ok(val) => Ok(val),
							Err(err) => Err(anyhow!(err)),
						})
						.collect::<anyhow::Result<Vec<u32>>>()?;

					table_func_indices.extend_from_slice(&segment_func_indices);
				},
				ElementItems::Expressions(_) => return Err(anyhow!("never exec here")),
			}
		}

		// Replacement map is at least export section size.
		let mut replacement_map: Map<u32, Thunk> = Map::new();

		for func_idx in exported_func_indices
			.chain(table_func_indices)
			.chain(module_info.start_function.into_iter())
		{
			let callee_stack_cost =
				ctx.stack_cost(func_idx).ok_or_else(|| anyhow!("function index isn't found"))?;

			// Don't generate a thunk if stack_cost of a callee is zero.
			if callee_stack_cost != 0 {
				replacement_map.insert(
					func_idx,
					Thunk {
						signature: match module_info.get_type_by_func_idx(func_idx)?.clone() {
							Type::Func(ft) => ft,
							// TODO: Type::Array(_) part of GC proposal
							Type::Array(_) => todo!("Array type not supported yet"),
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

	let mut next_func_idx = module_info.function_map.len() as u32;
	let mut functions = vec![];
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

		functions.push((Type::Func(thunk.signature.clone()), thunk_body));

		thunk.idx = Some(next_func_idx);
		next_func_idx += 1;
	}
	module_info.add_functions(&functions)?;

	// And finally, fixup thunks in export and table sections.

	// Fixup original function index to a index of a thunk generated earlier.
	let mut export_sec_builder = ExportSection::new();
	let exports = module_info.export_section()?.unwrap_or(vec![]);
	for export in exports {
		let mut function_idx = export.index;
		if let ExternalKind::Func = export.kind {
			if let Some(thunk) = replacement_map.get(&function_idx) {
				function_idx =
					thunk.idx.expect("at this point an index must be assigned to each thunk");
			}
		}
		export_sec_builder.export(
			export.name,
			DefaultTranslator.translate_export_kind(export.kind)?,
			function_idx,
		);
	}

	let mut ele_sec_builder = ElementSection::new();
	let elements = module_info.element_section()?.unwrap_or(vec![]);
	for elem in elements {
		let mut functions = vec![];
		match elem.items {
			ElementItems::Functions(func_indexes) => {
				for item in func_indexes.into_iter() {
					let func_idx = item.map_err(|err| anyhow!(err))?;
					if let Some(thunk) = replacement_map.get(&func_idx) {
						let new_func_idx = thunk.idx.ok_or_else(|| {
							anyhow!("at this point an index must be assigned to each thunk")
						})?; //resolve thunk func type, this signature should exit
						functions.push(new_func_idx);
					} else {
						functions.push(func_idx);
					}
				}
			},
			ElementItems::Expressions(_) => return Err(anyhow!("element must be func here")),
		}

		let offset;
		let mode = match elem.kind {
			ElementKind::Active { table_index, offset_expr } => {
				offset = DefaultTranslator.translate_const_expr(
					&offset_expr,
					&wasmparser::ValType::I32,
					ConstExprKind::ElementOffset,
				)?;

				ElementMode::Active { table: table_index, offset: &offset }
			},
			ElementKind::Passive => ElementMode::Passive,
			ElementKind::Declared => ElementMode::Declared,
		};

		let element_type = DefaultTranslator.translate_ref_ty(&elem.ty)?;
		let elements = Elements::Functions(&functions);

		ele_sec_builder.segment(ElementSegment {
			mode,
			// The element type.
			element_type,
			// The element functions.
			elements,
		});
	}

	module_info.replace_section(SectionId::Export.into(), &export_sec_builder)?;
	module_info.replace_section(SectionId::Element.into(), &ele_sec_builder)?;
	if let Some(start_idx) = module_info.start_function {
		let mut new_func_idx = start_idx;
		if let Some(thunk) = replacement_map.get(&start_idx) {
			new_func_idx =
				thunk.idx.expect("at this point an index must be assigned to each thunk");
		}

		module_info.replace_section(
			SectionId::Start.into(),
			&wasm_encoder::StartSection { function_index: new_func_idx },
		)?;
	}
	Ok(())
}
