use alloc::string::String;
use alloc::{vec, vec::Vec};
pub mod translator;
use crate::parser::translator::{DefaultTranslator, Translator};
use alloc::collections::{BTreeMap, BTreeSet};
use anyhow::{anyhow, Result};
use core::ops::Range;
use paste::paste;
use wasm_encoder::{Encode, ExportKind, SectionId};
use wasmparser::{
	Chunk, CodeSectionReader, Export, ExportSectionReader, ExternalKind, FunctionBody,
	FunctionSectionReader, Global, GlobalSectionReader, GlobalType, Import, ImportSectionReader,
	IndirectNameMap, MemorySectionReader, MemoryType, NameMap, NameSectionReader, Parser, Payload,
	Table, TableSectionReader, TableType, Type,
};

#[derive(Clone, Debug)]
pub struct RawSection {
	/// The id for this section.
	pub id: u8,
	/// The raw data for this section.
	pub data: Vec<u8>,
}

impl RawSection {
	pub fn new(id: u8, data: Vec<u8>) -> Self {
		RawSection { id, data }
	}
}

impl Encode for RawSection {
	fn encode(&self, sink: &mut Vec<u8>) {
		self.data.encode(sink);
	}
}

impl wasm_encoder::Section for RawSection {
	fn id(&self) -> u8 {
		self.id
	}
}

impl wasm_encoder::ComponentSection for RawSection {
	fn id(&self) -> u8 {
		self.id
	}
}

/// Provides module information for future usage during mutation
/// an instance of ModuleInfo could be user to determine which mutation could be applied
#[derive(Default, Clone, Debug)]
#[warn(dead_code)]
pub struct ModuleInfo {
	// The following fields are offsets inside the `raw_sections` field.
	// The main idea is to maintain the order of the sections in the input Wasm.
	pub export_names: BTreeSet<String>,

	pub exports_count: u32,
	pub exports_global_count: u32,

	pub elements_count: u32,
	pub data_segments_count: u32,
	pub start_function: Option<u32>,
	pub memory_count: u32,
	pub table_count: u32,
	pub tag_count: u32,

	pub imported_functions_count: u32,
	pub imported_globals_count: u32,
	pub imported_memories_count: u32,
	pub imported_tables_count: u32,
	pub imported_tags_count: u32,

	// types for inner functions
	pub types_map: Vec<Type>,

	// function idx to type idx
	pub function_map: Vec<u32>,
	pub global_types: Vec<GlobalType>,
	pub table_elem_types: Vec<TableType>,
	pub memory_types: Vec<MemoryType>,

	// raw_sections
	pub raw_sections: BTreeMap<u8, RawSection>,
	pub component_raw_sections: BTreeMap<u8, RawSection>,
}

macro_rules! add_section_function {
	($name:ident, $ty:expr) => {
		paste! {
			#[allow(dead_code)]
			pub fn [< $name:lower _section>](&self) -> Vec<$ty> {

				if let Some(section) =  self.raw_sections.get(&SectionId::$name.into()) {
					if let Ok(reader) = [< $name SectionReader >]::new(&section.data, 0) {
						// Assuming the module has already been validated and no errors shall occur
						// when iterating the items
						return reader.into_iter().filter_map(|x| x.ok()).collect::<Vec<$ty>>();
					}
				}

				return vec![];
			}
		}
	};
}

impl ModuleInfo {
	/// Parse the given Wasm bytes and fill out a `ModuleInfo` AST for it.
	pub fn new(input_wasm: &[u8]) -> Result<ModuleInfo> {
		let mut parser = Parser::new(0);
		let mut info = ModuleInfo::default();
		let mut wasm = input_wasm;

		loop {
			let (payload, consumed) = match parser.parse(wasm, true)? {
				Chunk::NeedMoreData(hint) => {
					panic!("Invalid Wasm module {:?}", hint);
				},
				Chunk::Parsed { consumed, payload } => (payload, consumed),
			};

			match payload {
				Payload::CodeSectionStart { count: _, range, size: _ } => {
					info.section(SectionId::Code.into(), range.clone(), input_wasm);
					parser.skip_section();
					// update slice, bypass the section
					wasm = &input_wasm[range.end..];

					continue;
				},
				Payload::TypeSection(reader) => {
					info.section(SectionId::Type.into(), reader.range(), input_wasm);

					// Save function types
					for ty in reader.into_iter() {
						info.types_map.push(ty?);
					}
				},
				Payload::ImportSection(reader) => {
					info.section(SectionId::Import.into(), reader.range(), input_wasm);

					//for _ in 0..reader.get_count()
					for import in reader.into_iter() {
						let import = import?;
						match import.ty {
							wasmparser::TypeRef::Func(ty) => {
								// Save imported functions
								info.function_map.push(ty);
								info.imported_functions_count += 1;
							},
							wasmparser::TypeRef::Global(ty) => {
								info.global_types.push(ty);
								info.imported_globals_count += 1;
							},
							wasmparser::TypeRef::Memory(ty) => {
								info.memory_count += 1;
								info.imported_memories_count += 1;
								info.memory_types.push(ty);
							},
							wasmparser::TypeRef::Table(ty) => {
								info.table_count += 1;
								info.imported_tables_count += 1;
								info.table_elem_types.push(ty);
							},
							wasmparser::TypeRef::Tag(_ty) => {
								info.tag_count += 1;
								info.imported_tags_count += 1;
							},
						}
					}
				},
				Payload::FunctionSection(reader) => {
					info.section(SectionId::Function.into(), reader.range(), input_wasm);

					for func_idx in reader.into_iter() {
						info.function_map.push(func_idx?);
					}
				},
				Payload::TableSection(reader) => {
					info.table_count += reader.count();
					info.section(SectionId::Table.into(), reader.range(), input_wasm);

					for table in reader.into_iter() {
						let table = table?;
						info.table_elem_types.push(table.ty);
					}
				},
				Payload::MemorySection(reader) => {
					info.memory_count += reader.count();
					info.section(SectionId::Memory.into(), reader.range(), input_wasm);

					for ty in reader.into_iter() {
						info.memory_types.push(ty?);
					}
				},
				Payload::GlobalSection(reader) => {
					info.section(SectionId::Global.into(), reader.range(), input_wasm);

					for global in reader.into_iter() {
						let global = global?;
						info.global_types.push(global.ty);
					}
				},
				Payload::ExportSection(reader) => {
					info.exports_count = reader.count();

					for export in reader.clone().into_iter() {
						let export = export?;
						if let ExternalKind::Global = export.kind {
							info.exports_global_count += 1;
						}
						info.export_names.insert(export.name.into());
					}

					info.section(SectionId::Export.into(), reader.range(), input_wasm);
				},
				Payload::StartSection { func, range } => {
					info.start_function = Some(func);
					info.section(SectionId::Start.into(), range, input_wasm);
				},
				Payload::ElementSection(reader) => {
					info.elements_count = reader.count();
					info.section(SectionId::Element.into(), reader.range(), input_wasm);
				},
				Payload::DataSection(reader) => {
					info.data_segments_count = reader.count();
					info.section(SectionId::Data.into(), reader.range(), input_wasm);
				},
				Payload::CustomSection(c) => {
					// At the moment only name section supported
					if c.name() == "name" {
						info.section(
							SectionId::Custom.into(),
							Range { start: c.data_offset(), end: c.range().end },
							input_wasm,
						);
					}
				},
				Payload::UnknownSection { id, contents: _, range } => {
					info.section(id, range, input_wasm);
				},
				Payload::DataCountSection { count: _, range } => {
					info.section(SectionId::DataCount.into(), range, input_wasm);
				},
				Payload::Version { .. } => {},
				Payload::End(_) => {
					break;
				},
				_ => todo!("{:?} not implemented", payload),
			}
			wasm = &wasm[consumed..];
		}

		Ok(info)
	}

	/// Registers a new raw_section in the ModuleInfo
	pub fn section(&mut self, id: u8, range: Range<usize>, full_wasm: &[u8]) {
		self.raw_sections.insert(id, RawSection::new(id, full_wasm[range].to_vec()));
	}

	/// Returns the function type based on the index of the function type
	/// `types[functions[idx]]`
	pub fn get_functype_idx(&self, idx: u32) -> Result<&Type> {
		if idx >= self.function_map.len() as u32 {
			return Err(anyhow!("function {} not exit", idx));
		}
		let functpeindex = self.function_map[idx as usize] as usize;
		if functpeindex >= self.types_map.len() {
			return Err(anyhow!("type {} not exit", functpeindex));
		}
		Ok(&self.types_map[functpeindex])
	}

	pub fn resolve_type_idx(&self, t: &Type) -> Option<u32> {
		for (index, ty) in self.types_map.iter().enumerate() {
			let Type::Func(ot) = ty else { todo!() };
			let Type::Func(dt) = t else { todo!() };
			if ot.eq(dt) {
				return Some(index as u32);
			}
		}
		None
	}

	pub fn add_func_type(&mut self, func_type: &Type) -> Result<u32> {
		let func_type_index = match self.resolve_type_idx(func_type) {
			None => self.types_map.len() as u32,
			Some(index) => return Ok(index),
		};
		//add new type
		let mut type_builder = wasm_encoder::TypeSection::new();
		for t in &self.types_map {
			DefaultTranslator.translate_type_def(t.clone(), &mut type_builder)?;
		}
		self.types_map.push(func_type.clone());
		DefaultTranslator.translate_type_def(func_type.clone(), &mut type_builder)?;
		self.replace_section(SectionId::Type.into(), &type_builder)?;
		Ok(func_type_index)
	}

	/// Returns the number of globals used by the Wasm binary including imported
	/// glboals
	#[allow(dead_code)]
	pub fn get_global_count(&self) -> usize {
		self.global_types.len()
	}

	/// Replace the `i`th section in this module with the given new section.
	pub fn replace_section(
		&mut self,
		sec_type: u8,
		new_section: &impl wasm_encoder::Section,
	) -> Result<()> {
		self.raw_sections
			.insert(sec_type, RawSection::new(sec_type, truncate_len_from_encoder(new_section)?));
		Ok(())
	}

	pub fn get_count(&mut self, section: SectionId) -> Result<u32> {
		if let Some(section) = self.raw_sections.get(&section.into()) {
			let section_reader = wasmparser::ExportSectionReader::new(&section.data, 0)?;
			Ok(section_reader.count())
		} else {
			Err(anyhow!("section not found"))
		}
	}

	pub fn add_export(&mut self, name: &str, kind: ExportKind, index: u32) -> Result<()> {
		let mut section_builder = wasm_encoder::ExportSection::new();

		for export in self.export_section() {
			let export_kind = DefaultTranslator.translate_export_kind(export.kind).unwrap();
			section_builder.export(export.name, export_kind, export.index);
		}

		section_builder.export(name, kind, index);
		if let ExportKind::Global = kind {
			self.exports_global_count += 1;
		}
		self.export_names.insert(name.into());
		self.replace_section(SectionId::Export.into(), &section_builder)
	}

	pub fn add_global(
		&mut self,
		global_type: GlobalType,
		init_expr: &wasm_encoder::ConstExpr,
	) -> Result<()> {
		let mut section_builder = wasm_encoder::GlobalSection::new();

		if let Some(section) = self.raw_sections.get(&SectionId::Global.into()) {
			let section_reader = wasmparser::GlobalSectionReader::new(&section.data, 0)?;

			for section_item in section_reader {
				DefaultTranslator.translate_global(section_item?, &mut section_builder)?;
			}
		}

		section_builder.global(DefaultTranslator.translate_global_type(&global_type)?, init_expr);
		self.global_types.push(global_type);

		self.replace_section(SectionId::Global.into(), &section_builder)
	}

	pub fn add_func(&mut self, func_type: Type, func_body: &wasm_encoder::Function) -> Result<()> {
		let func_type_index = self.add_func_type(&func_type)?;

		// Recreate Function section
		let mut section_builder = wasm_encoder::FunctionSection::new();

		for function in self.function_section() {
			section_builder.function(function);
		}

		// Define a new function in Function section
		section_builder.function(func_type_index);
		self.function_map.push(func_type_index);
		self.replace_section(SectionId::Function.into(), &section_builder)?;

		// Recreate Code section
		let mut section_builder = wasm_encoder::CodeSection::new();

		if let Some(section) = self.raw_sections.get(&SectionId::Code.into()) {
			let section_reader = wasmparser::CodeSectionReader::new(&section.data, 0)?;

			for section_item in section_reader {
				DefaultTranslator.translate_code(section_item?, &mut section_builder)?
			}
		}

		// Write new function body in Code section
		section_builder.function(func_body);
		self.replace_section(SectionId::Code.into(), &section_builder)
	}

	pub fn add_import_func(
		&mut self,
		module: &str,
		func_name: &str,
		func_type: Type,
	) -> Result<()> {
		let func_type_idx = self.add_func_type(&func_type)?;

		// Recreate Import section
		let mut import_decoder = wasm_encoder::ImportSection::new();
		for import in self.import_section() {
			DefaultTranslator.translate_import(import, &mut import_decoder)?;
		}

		// Define new function import in the Import section.
		import_decoder.import(module, func_name, wasm_encoder::EntityType::Function(func_type_idx));
		self.function_map.push(func_type_idx);
		self.imported_functions_count += 1;
		self.replace_section(SectionId::Import.into(), &import_decoder)
	}

	pub fn add_import_global(
		&mut self,
		module: &str,
		global_name: &str,
		global_type: GlobalType,
	) -> Result<()> {
		let mut import_decoder = wasm_encoder::ImportSection::new();
		for import in self.import_section() {
			DefaultTranslator.translate_import(import, &mut import_decoder)?;
		}

		// Define new global import in the Import section.
		import_decoder.import(
			module,
			global_name,
			DefaultTranslator.translate_global_type(&global_type)?,
		);
		self.imported_globals_count += 1;
		self.replace_section(SectionId::Import.into(), &import_decoder)
	}

	pub fn bytes(&self) -> Vec<u8> {
		let mut module = wasm_encoder::Module::new();
		for s in self.raw_sections.values() {
			module.section(s);
		}
		module.finish()
	}

	#[allow(dead_code)]
	pub fn num_functions(&self) -> u32 {
		self.function_map.len() as u32
	}

	#[allow(dead_code)]
	pub fn num_local_functions(&self) -> u32 {
		self.num_functions() - self.num_imported_functions()
	}

	#[allow(dead_code)]
	pub fn num_imported_functions(&self) -> u32 {
		self.imported_functions_count
	}

	#[allow(dead_code)]
	pub fn num_tables(&self) -> u32 {
		self.table_count
	}

	#[allow(dead_code)]
	pub fn num_imported_tables(&self) -> u32 {
		self.imported_tables_count
	}

	#[allow(dead_code)]
	pub fn num_memories(&self) -> u32 {
		self.memory_count
	}

	#[allow(dead_code)]
	pub fn num_imported_memories(&self) -> u32 {
		self.imported_memories_count
	}

	#[allow(dead_code)]
	pub fn num_globals(&self) -> u32 {
		self.global_types.len() as u32
	}

	#[allow(dead_code)]
	pub fn num_imported_globals(&self) -> u32 {
		self.imported_globals_count
	}

	#[allow(dead_code)]
	pub fn num_local_globals(&self) -> u32 {
		self.global_types.len() as u32 - self.imported_globals_count
	}

	#[allow(dead_code)]
	pub fn num_tags(&self) -> u32 {
		self.tag_count
	}

	#[allow(dead_code)]
	pub fn num_imported_tags(&self) -> u32 {
		self.imported_tags_count
	}

	#[allow(dead_code)]
	pub fn num_data(&self) -> u32 {
		self.data_segments_count
	}

	#[allow(dead_code)]
	pub fn num_elements(&self) -> u32 {
		self.elements_count
	}

	#[allow(dead_code)]
	pub fn num_types(&self) -> u32 {
		self.types_map.len() as u32
	}

	#[allow(dead_code)]
	pub fn num_export_global(&self) -> u32 {
		self.exports_global_count
	}

	add_section_function!(Export, Export);
	add_section_function!(Global, Global);
	add_section_function!(Import, Import);
	add_section_function!(Function, u32);
	add_section_function!(Memory, MemoryType);
	add_section_function!(Table, Table);
	add_section_function!(Code, FunctionBody);
}

// Then insert metering calls into a sequence of instructions given the block locations and costs.
pub fn copy_locals(func_body: &FunctionBody) -> Result<Vec<(u32, wasm_encoder::ValType)>> {
	let mut local_reader = func_body.get_locals_reader()?;
	// Get current locals and map to encoder types
	let current_locals: Vec<(u32, wasm_encoder::ValType)> = (0..local_reader.get_count())
		.map(|_| {
			let (count, ty) = local_reader.read().unwrap();
			(count, DefaultTranslator.translate_ty(&ty).unwrap())
		})
		.collect::<Vec<(u32, wasm_encoder::ValType)>>();

	Ok(current_locals)
}

//todo unable to get function encoder body directly, remove this after option wasmparser
pub fn truncate_len_from_encoder(func_builder: &dyn wasm_encoder::Encode) -> Result<Vec<u8>> {
	let mut d = vec![];
	func_builder.encode(&mut d);
	let mut r = wasmparser::BinaryReader::new(&d);
	let size = r.read_var_u32()?;
	Ok(r.read_bytes(size as usize)?.to_vec())
}

pub fn rebuild_name_map(name_map: NameMap) -> Result<wasm_encoder::NameMap> {
	let mut encoded_map = wasm_encoder::NameMap::new();
	for naming in name_map {
		let naming = naming?;
		encoded_map.append(naming.index, naming.name);
	}
	Ok(encoded_map)
}

pub fn rebuild_indirect_name_map(
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
