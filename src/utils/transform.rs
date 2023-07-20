use crate::utils::module_info::ModuleInfo;
use anyhow::Result;
use wasm_encoder::SectionId;
use wasmparser::{IndirectNameMap, NameMap, NameSectionReader};

fn rebuild_name_map(name_map: NameMap) -> Result<wasm_encoder::NameMap> {
	let mut encoded_map = wasm_encoder::NameMap::new();
	for naming in name_map {
		let naming = naming?;
		encoded_map.append(naming.index, naming.name);
	}
	Ok(encoded_map)
}

fn rebuild_indirect_name_map(
	indirect_name_map: IndirectNameMap,
) -> Result<wasm_encoder::IndirectNameMap> {
	let mut encoded_map = wasm_encoder::IndirectNameMap::new();
	for indirect_naming in indirect_name_map {
		let indirect_naming = indirect_naming?;
		let new_map = rebuild_name_map(indirect_naming.names)?;

		encoded_map.append(indirect_naming.index, &new_map);
	}
	Ok(encoded_map)
}

/// - Update function indices in Custom Name section Increment indices greater or equal than given
///   one (incrementing by 1 because it is assumed one function has been added before the given
///   index)
/// - Rebuild remaining section items
pub fn process_custom_section(
	module_info: &mut ModuleInfo,
	update_func_idx: Option<u32>,
) -> Result<()> {
	if let Some(custom_section) = module_info.raw_sections.get_mut(&SectionId::Custom.into()) {
		let mut name_section_builder = wasm_encoder::NameSection::new();
		let name_sec_reader = NameSectionReader::new(&custom_section.data, 0);

		for item in name_sec_reader {
			match item? {
				wasmparser::Name::Function(name_map) => {
					let mut new_map = wasm_encoder::NameMap::new();
					for naming in name_map.into_iter() {
						let naming = naming?;
						let idx = if let Some(idx) = update_func_idx {
							if naming.index >= idx {
								naming.index + 1
							} else {
								naming.index
							}
						} else {
							naming.index
						};
						new_map.append(idx, naming.name);
					}
					name_section_builder.functions(&new_map);
				},
				wasmparser::Name::Local(local) => {
					let map = rebuild_indirect_name_map(local)?;
					name_section_builder.locals(&map);
				},
				wasmparser::Name::Label(label) => {
					let map = rebuild_indirect_name_map(label)?;
					name_section_builder.labels(&map);
				},
				wasmparser::Name::Type(types) => {
					let map = rebuild_name_map(types)?;
					name_section_builder.types(&map);
				},
				wasmparser::Name::Data(data) => {
					let map = rebuild_name_map(data)?;
					name_section_builder.data(&map);
				},
				wasmparser::Name::Table(table) => {
					let map = rebuild_name_map(table)?;
					name_section_builder.tables(&map);
				},
				wasmparser::Name::Memory(memory) => {
					let map = rebuild_name_map(memory)?;
					name_section_builder.memories(&map);
				},
				wasmparser::Name::Global(global) => {
					let map = rebuild_name_map(global)?;
					name_section_builder.globals(&map);
				},
				wasmparser::Name::Element(element) => {
					let map = rebuild_name_map(element)?;
					name_section_builder.elements(&map);
				},
				wasmparser::Name::Module { name, .. } => {
					todo!("Name Module section not supported - {:?}", name);
				},
				wasmparser::Name::Unknown { ty, .. } => {
					todo!("Name Unknown section not supported - {:?}", ty);
				},
			}
		}
		module_info.replace_section(SectionId::Custom.into(), &name_section_builder.as_custom())?;
	}

	Ok(())
}
