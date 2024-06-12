use crate::utils::{
	errors::ModuleInfoError,
	translator::{DefaultTranslator, Translator},
};
use alloc::{
	collections::{BTreeMap, BTreeSet},
	format,
	string::String,
	vec,
	vec::Vec,
};
use core::ops::Range;
use paste::paste;
use wasm_encoder::{Encode, ExportKind, SectionId};
use wasmparser::{
	Chunk, CodeSectionReader, Element, ElementSectionReader, Export, ExportSectionReader,
	ExternalKind, FunctionBody, FunctionSectionReader, Global, GlobalSectionReader, GlobalType,
	Import, ImportSectionReader, MemorySectionReader, MemoryType, Parser, Payload,
	Result as WasmParserResult, Table, TableSectionReader, TableType, Type, Validator,
	WasmFeatures,
};

#[derive(Clone, Debug)]
pub struct RawSection {
	/// The id for this section.
	pub id: u8,
	/// The offset of this section in the original WASM file.
	/// If the section is added manually, then set to None
	pub offset: Option<usize>,
	/// The raw data for this section.
	pub data: Vec<u8>,
}

impl RawSection {
	pub fn new(id: u8, offset: Option<usize>, data: Vec<u8>) -> Self {
		RawSection { id, offset, data }
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

type Result<T> = core::result::Result<T, ModuleInfoError>;

/// Provides module information for future usage during mutation
/// an instance of ModuleInfo could be user to determine which mutation could be applied
#[derive(Default, Clone, Debug)]
#[warn(dead_code)]
pub struct ModuleInfo {
	// The following fields are offsets inside the `raw_sections` field.
	// The main idea is to maintain the order of the sections in the input Wasm.
	pub export_names: BTreeSet<String>,

	pub code_section_entry_count: u32,
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
}

macro_rules! add_section_function {
	($name:ident, $ty:expr) => {
		paste! {
			// TODO Consider reworking it to return iterator
			#[allow(dead_code)]
			pub fn [< $name:lower _section>](&self) -> Result<Option<Vec<$ty>>> {

				if let Some(section) =  self.raw_sections.get(&SectionId::$name.into()) {
					let reader = [< $name SectionReader >]::new(&section.data, 0)?;
					let vec = reader.into_iter().collect::<WasmParserResult<Vec<$ty>>>()?;
					Ok(Some(vec))
				}
				else {
					Ok(None)
				}
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
				Payload::CodeSectionStart { count, range, size: _ } => {
					info.code_section_entry_count = count;
					info.section(SectionId::Code.into(), range.clone(), input_wasm)?;
					parser.skip_section();
					// update slice, bypass the section
					wasm = &input_wasm[range.end..];

					continue;
				},
				Payload::TypeSection(reader) => {
					info.section(SectionId::Type.into(), reader.range(), input_wasm)?;

					// Save function types
					for ty in reader.into_iter() {
						info.types_map.push(ty?);
					}
				},
				Payload::ImportSection(reader) => {
					info.section(SectionId::Import.into(), reader.range(), input_wasm)?;

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
					info.section(SectionId::Function.into(), reader.range(), input_wasm)?;

					for func_idx in reader.into_iter() {
						info.function_map.push(func_idx?);
					}
				},
				Payload::TableSection(reader) => {
					info.table_count = reader.count();
					info.section(SectionId::Table.into(), reader.range(), input_wasm)?;

					for table in reader.into_iter() {
						let table = table?;
						info.table_elem_types.push(table.ty);
					}
				},
				Payload::MemorySection(reader) => {
					info.memory_count = reader.count();
					info.section(SectionId::Memory.into(), reader.range(), input_wasm)?;

					for ty in reader.into_iter() {
						info.memory_types.push(ty?);
					}
				},
				Payload::GlobalSection(reader) => {
					info.section(SectionId::Global.into(), reader.range(), input_wasm)?;

					for global in reader.into_iter() {
						let global = global?;
						info.global_types.push(global.ty);
					}
				},
				Payload::ExportSection(reader) => {
					for export in reader.clone().into_iter() {
						let export = export?;
						if !info.export_names.contains(export.name) {
							if let ExternalKind::Global = export.kind {
								info.exports_global_count += 1;
							}
							info.export_names.insert(export.name.into());
							info.exports_count += 1;
						} else {
							return Err(ModuleInfoError::ExportAlreadyExists(export.name.into()));
						}
					}
					info.section(SectionId::Export.into(), reader.range(), input_wasm)?;
				},
				Payload::StartSection { func, range } => {
					info.start_function = Some(func);
					info.section(SectionId::Start.into(), range, input_wasm)?;
				},
				Payload::ElementSection(reader) => {
					info.elements_count = reader.count();
					info.section(SectionId::Element.into(), reader.range(), input_wasm)?;
				},
				Payload::DataSection(reader) => {
					info.data_segments_count = reader.count();
					info.section(SectionId::Data.into(), reader.range(), input_wasm)?;
				},
				#[allow(unused_variables)]
				Payload::CustomSection(c) => {
					#[cfg(not(feature = "ignore_custom_section"))]
					// At the moment only name section supported
					if c.name() == "name" {
						info.section(SectionId::Custom.into(), c.range(), input_wasm)?;
					}
				},

				Payload::DataCountSection { count: _, range } => {
					info.section(SectionId::DataCount.into(), range, input_wasm)?;
				},
				Payload::Version { .. } => {},
				Payload::End(_) => break,
				p => return Err(ModuleInfoError::SectionNotSupported(format!("{:?}", p))),
			}
			wasm = &wasm[consumed..];
		}

		Ok(info)
	}

	#[cfg(test)]
	pub fn assert_stats(&self) {
		// Global section
		assert_eq!(
			self.global_section().unwrap().unwrap_or_default().len(),
			self.num_local_globals() as usize
		);
		// Imported globals
		assert_eq!(
			self.global_types.len() - self.num_local_globals() as usize,
			self.num_imported_globals() as usize
		);

		// Export section
		assert_eq!(self.export_names.len(), self.exports_count as usize);
		assert_eq!(
			self.export_section().unwrap().unwrap_or_default().len(),
			self.exports_count as usize
		);
		// Element section
		assert_eq!(
			self.element_section().unwrap().unwrap_or_default().len(),
			self.elements_count as usize,
		);
	}

	/// Validates the WASM binary
	pub fn validate(&self, features: WasmFeatures) -> Result<()> {
		if self.code_section_entry_count != self.num_local_functions() {
			return Err(ModuleInfoError::CodeAndFuncSectionCntMismatch(
				self.code_section_entry_count,
				self.num_local_functions(),
			));
		}

		// TODO validate_all() creates internal parser and parses the binary again. Rework to
		// validate while parsing
		Validator::new_with_features(features).validate_all(&self.bytes())?;

		Ok(())
	}

	/// Registers a new raw_section in the ModuleInfo
	pub fn section(&mut self, id: u8, range: Range<usize>, full_wasm: &[u8]) -> Result<()> {
		if self.raw_sections.get(&id).is_none() {
			if range.start > full_wasm.len() || range.end > full_wasm.len() {
				Err(ModuleInfoError::SectionRangeExceedsWasmLength {
					range,
					wasm_len: full_wasm.len(),
				})
			} else {
				self.raw_sections
					.insert(id, RawSection::new(id, Some(range.start), full_wasm[range].to_vec()));
				Ok(())
			}
		} else {
			Err(ModuleInfoError::SectionAlreadyExists(id))
		}
	}

	/// Returns the function type based on the index of the type
	/// `types[idx]`
	pub fn get_type_by_idx(&self, type_idx: u32) -> Result<&Type> {
		if type_idx >= self.types_map.len() as u32 {
			return Err(ModuleInfoError::TypeDoesNotExist(type_idx));
		}
		Ok(&self.types_map[type_idx as usize])
	}

	/// Returns the function type based on the index of the function
	/// `types[functions[idx]]`
	pub fn get_type_by_func_idx(&self, func_idx: u32) -> Result<&Type> {
		if func_idx >= self.function_map.len() as u32 {
			return Err(ModuleInfoError::FunctionDoesNotExist(func_idx));
		}
		let type_idx = self.function_map[func_idx as usize];
		self.get_type_by_idx(type_idx)
	}

	pub fn resolve_type_idx(&self, t: &Type) -> Option<u32> {
		// TODO: Type::Array(_) part of GC proposal
		let Type::Func(dt) = t else { todo!("Array type not supported yet") };
		for (index, ty) in self.types_map.iter().enumerate() {
			let Type::Func(ot) = ty else { todo!("Array type not supported yet") };
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
		// Add new type
		let mut type_builder = wasm_encoder::TypeSection::new();
		for t in &self.types_map {
			DefaultTranslator.translate_type_def(t.clone(), &mut type_builder)?;
		}
		self.types_map.push(func_type.clone());
		DefaultTranslator.translate_type_def(func_type.clone(), &mut type_builder)?;
		self.replace_section(SectionId::Type.into(), &type_builder)?;
		Ok(func_type_index)
	}

	/// Replace the `i`th section in this module with the given new section.
	pub fn replace_section(
		&mut self,
		sec_type: u8,
		new_section: &impl wasm_encoder::Section,
	) -> Result<()> {
		// If section was not present before then offset is None
		let offset = self.raw_sections.get(&sec_type).and_then(|sec| sec.offset);

		self.raw_sections.insert(
			sec_type,
			RawSection::new(sec_type, offset, truncate_len_from_encoder(new_section)?),
		);
		Ok(())
	}

	pub fn add_exports(&mut self, exports: &[(String, ExportKind, u32)]) -> Result<()> {
		let mut section_builder = wasm_encoder::ExportSection::new();

		for export in self.export_section()?.unwrap_or(vec![]) {
			let export_kind = DefaultTranslator.translate_export_kind(export.kind)?;
			section_builder.export(export.name, export_kind, export.index);
		}

		for (name, kind, index) in exports {
			if !self.export_names.contains(name) {
				section_builder.export(name, *kind, *index);
				if let ExportKind::Global = kind {
					self.exports_global_count += 1;
				}
				self.export_names.insert(String::from(name));
				self.exports_count += 1;
			} else {
				return Err(ModuleInfoError::ExportAlreadyExists(String::from(name)));
			}
		}
		self.replace_section(SectionId::Export.into(), &section_builder)
	}

	pub fn remove_export(&mut self, name: &str) -> Result<()> {
		let mut section_builder = wasm_encoder::ExportSection::new();
		let export_section = self.export_section()?.unwrap_or(vec![]);

		// let mut idx = 0;
		let mut export_names: BTreeSet<String> = BTreeSet::new();
		let mut exports_count: u32 = 0;
		let mut exports_global_count: u32 = 0;

		for export in export_section {
			let export_kind = DefaultTranslator.translate_export_kind(export.kind)?;
			// println!("remove_export export = {:?} export_kind = {:?}", export, export_kind);
			if !export.name.ends_with(name) {
				section_builder.export(export.name, export_kind, export.index);
				exports_count += 1;
				export_names.insert(String::from(export.name));
				if let ExportKind::Global = export_kind {
					exports_global_count += 1;
				}
			}
		}
		self.export_names = export_names;
		self.exports_count = exports_count;
		self.exports_global_count = exports_global_count;

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

	/// Add functions specified as a vector of tuples: signature and body
	pub fn add_functions(&mut self, funcs: &[(Type, wasm_encoder::Function)]) -> Result<()> {
		// Recreate Function section
		let mut function_builder = wasm_encoder::FunctionSection::new();
		for function in self.function_section()?.unwrap_or(vec![]) {
			function_builder.function(function);
		}

		// Recreate Code section
		let mut code_builder = wasm_encoder::CodeSection::new();
		for funcion_body in self.code_section()?.unwrap_or(vec![]) {
			DefaultTranslator.translate_code(funcion_body, &mut code_builder)?
		}

		for (func_type, func_body) in funcs {
			let func_type_index = self.add_func_type(func_type)?;

			// Define a new function in Function section
			function_builder.function(func_type_index);
			self.function_map.push(func_type_index);

			// Write new function body in Code section
			code_builder.function(func_body);
			self.code_section_entry_count += 1;
		}
		self.replace_section(SectionId::Function.into(), &function_builder)?;
		self.replace_section(SectionId::Code.into(), &code_builder)
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
		for import in self.import_section()?.unwrap_or(vec![]) {
			DefaultTranslator.translate_import(import, &mut import_decoder)?;
		}

		// Define new function import in the Import section.
		import_decoder.import(module, func_name, wasm_encoder::EntityType::Function(func_type_idx));
		// function_map consist of:
		// - imported function first
		// - local functions then
		// This is important when getting function Type by its index get_type_by_func_idx()
		// When adding an import we make sure it is added as the last imported function
		// but before local functions
		self.function_map.insert(self.imported_functions_count as usize, func_type_idx);

		self.imported_functions_count += 1;
		self.replace_section(SectionId::Import.into(), &import_decoder)
	}

	pub fn add_import_global(
		&mut self,
		module: &str,
		global_name: &str,
		global_type: GlobalType,
	) -> Result<()> {
		// Recreate Import section
		let mut import_decoder = wasm_encoder::ImportSection::new();
		for import in self.import_section()?.unwrap_or(vec![]) {
			DefaultTranslator.translate_import(import, &mut import_decoder)?;
		}

		// Define new global import in the Import section.
		import_decoder.import(
			module,
			global_name,
			DefaultTranslator.translate_global_type(&global_type)?,
		);
		self.global_types.push(global_type);
		self.imported_globals_count += 1;
		self.replace_section(SectionId::Import.into(), &import_decoder)
	}

	pub fn modify_memory_type(&mut self, mem_index: u32, mem_type: MemoryType) -> Result<()> {
		let mut memory_builder = wasm_encoder::MemorySection::new();
		let mut memory_types = vec![];
		for (index, memory) in self
			.memory_section()?
			.ok_or_else(|| ModuleInfoError::NoMemorySection)?
			.iter()
			.enumerate()
		{
			let encoded_mem_type = if index as u32 != mem_index {
				memory_types.push(*memory);
				DefaultTranslator.translate_memory_type(memory)?
			} else {
				memory_types.push(mem_type);
				DefaultTranslator.translate_memory_type(&mem_type)?
			};
			memory_builder.memory(encoded_mem_type);
		}
		self.memory_types = memory_types;
		self.replace_section(SectionId::Memory.into(), &memory_builder)
	}

	pub fn bytes(&self) -> Vec<u8> {
		let mut module = wasm_encoder::Module::new();

		let section_order = [
			SectionId::Type,
			SectionId::Import,
			SectionId::Function,
			SectionId::Table,
			SectionId::Memory,
			SectionId::Global,
			SectionId::Export,
			SectionId::Start,
			SectionId::Element,
			SectionId::DataCount, /* datacount goes before code, see: https://webassembly.github.io/spec/core/binary/modules.html#data-count-section */
			SectionId::Code,
			SectionId::Data,
			SectionId::Custom, /* custom Name section after data section, see: https://webassembly.github.io/spec/core/appendix/custom.html#name-section */
			SectionId::Tag,
		];

		for s in section_order {
			if let Some(sec) = self.raw_sections.get(&s.into()) {
				module.section(sec);
			}
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

	/// Returns the number of globals: local and imported
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
	add_section_function!(Element, Element);
}

// Then insert metering calls into a sequence of instructions given the block locations and costs.
pub fn copy_locals(func_body: &FunctionBody) -> Result<Vec<(u32, wasm_encoder::ValType)>> {
	let local_reader = func_body.get_locals_reader()?;
	let mut current_locals = vec![];
	for local in local_reader.into_iter() {
		let (count, ty) = local?;
		current_locals.push((count, DefaultTranslator.translate_ty(&ty)?));
	}
	Ok(current_locals)
}

// TODO unable to get function encoder body directly, remove this after option wasmparser
pub fn truncate_len_from_encoder(func_builder: &dyn wasm_encoder::Encode) -> Result<Vec<u8>> {
	let mut d = vec![];
	func_builder.encode(&mut d);
	let mut r = wasmparser::BinaryReader::new(&d);
	let size = r.read_var_u32()?;
	Ok(r.read_bytes(size as usize)?.to_vec())
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{gas_metering, gas_metering::ConstantCostRules, stack_limiter};
	use alloc::string::ToString;
	use wasm_encoder::ExportKind;
	use wasmparser::{FuncType, ValType};

	fn wasm_to_wat(bytes: &[u8]) -> String {
		String::from_utf8(
			wabt::Wasm2Wat::new()
				.read_debug_names(true)
				.convert(bytes)
				.unwrap()
				.as_ref()
				.to_vec(),
		)
		.unwrap()
	}

	fn wat_to_wasm(code: &str) -> Vec<u8> {
		#[cfg(not(feature = "ignore_custom_section"))]
		let write_debug_names = true;
		#[cfg(feature = "ignore_custom_section")]
		let write_debug_names = false;

		wabt::Wat2Wasm::new()
			.write_debug_names(write_debug_names)
			.convert(code)
			.unwrap()
			.as_ref()
			.to_vec()
	}

	const WAT: &str = r#"
		(module

			(func $Test_f (param $0 i64) (result i64)
			  ;; Grow memory
			  (drop
				(memory.grow (i32.const 1000000))
			  )

			  ;; Encode () in SBOR at address 0x0
			  (i32.const 0)
			  (i32.const 92)  ;; prefix
			  (i32.store8)
			  (i32.const 1)
			  (i32.const 33)  ;; tuple value kind
			  (i32.store8)
			  (i32.const 2)
			  (i32.const 0)  ;; tuple length
			  (i32.store8)

			  ;; Return slice (ptr = 0, len = 3)
			  (i64.const 3)
			)

			(memory $0 1)
			(export "memory" (memory $0))
			(export "Test_f" (func $Test_f))
		)
		"#;

	#[test]
	fn test_check_wasm_wat_conversion() {
		let bytes = wat_to_wasm(WAT);
		let expected_wat = wasm_to_wat(&bytes);

		let module = ModuleInfo::new(&bytes).unwrap();
		let bytes = module.bytes();
		let wat = wasm_to_wat(&bytes);

		assert_eq!(expected_wat, wat)
	}

	#[test]
	fn test_module_info_stats() {
		let bytes = wat_to_wasm(WAT);
		let mut module = ModuleInfo::new(&bytes).unwrap();

		module.assert_stats();

		module
			.add_global(
				GlobalType { content_type: ValType::I64, mutable: true },
				&wasm_encoder::ConstExpr::i64_const(0),
			)
			.unwrap();

		module
			.add_import_global(
				"env",
				"some_global",
				GlobalType { content_type: ValType::I64, mutable: true },
			)
			.unwrap();

		let func_type = Type::Func(FuncType::new(vec![ValType::I64], vec![]));
		module.add_func_type(&func_type).unwrap();

		// Add import of function of type that already exists
		module.add_import_func("env", "some_func", func_type).unwrap();

		let func_type =
			Type::Func(FuncType::new(vec![ValType::I64, ValType::I64], vec![ValType::I32]));

		// Add import with function type that does not exist yet
		module.add_import_func("env", "some_func_2", func_type).unwrap();

		module
			.add_exports(&[
				("export_global".to_string(), ExportKind::Global, 0),
				("export_func".to_string(), ExportKind::Func, 0),
				("export_memory".to_string(), ExportKind::Memory, 0),
				("export_table".to_string(), ExportKind::Table, 0),
			])
			.unwrap();

		module.assert_stats();
	}

	#[test]
	fn test_instrument_vs_stats() {
		let bytes = include_bytes!("../../benches/fixtures/wasm/scrypto.wasm");
		let mut module = ModuleInfo::new(bytes).unwrap();

		module.assert_stats();

		let backend = gas_metering::host_function::Injector::new("env", "gas");
		let _injected_wasm =
			gas_metering::inject(&mut module, backend, &ConstantCostRules::default()).unwrap();
		module.assert_stats();

		let _stack_limited_wasm = stack_limiter::inject(&mut module, 1024).unwrap();
		module.assert_stats();

		let mut module = ModuleInfo::new(bytes).unwrap();

		module.assert_stats();

		let backend = gas_metering::mutable_global::Injector::new("env", "gas_left");
		let _injected_wasm =
			gas_metering::inject(&mut module, backend, &ConstantCostRules::default()).unwrap();
		module.assert_stats();

		let _stack_limited_wasm = stack_limiter::inject(&mut module, 1024).unwrap();
		module.assert_stats();
	}

	macro_rules! test_module_info_stats {
	    ($name:expr) => {
			paste! {
				#[test]
				fn [< test_module_info_stats_ $name:lower >]() {
					let bytes = include_bytes!(concat!("../../benches/fixtures/wasm/", stringify!($name), ".wasm"));
					let mut module = ModuleInfo::new(bytes).unwrap();

					module.assert_stats();

					let backend = gas_metering::host_function::Injector::new("env", "gas");
					let _injected_wasm =
						gas_metering::inject(&mut module, backend, &ConstantCostRules::default()).unwrap();
					module.assert_stats();

					let _stack_limited_wasm = stack_limiter::inject(&mut module, 1024).unwrap();
					module.assert_stats();

					let mut module = ModuleInfo::new(bytes).unwrap();

					module.assert_stats();

					let backend = gas_metering::mutable_global::Injector::new("env", "gas_left");
					let _injected_wasm =
						gas_metering::inject(&mut module, backend, &ConstantCostRules::default()).unwrap();
					module.assert_stats();

					let _stack_limited_wasm = stack_limiter::inject(&mut module, 1024).unwrap();
					module.assert_stats();
				}
			}
	    };
	}
	test_module_info_stats!(contract_terminate);
	test_module_info_stats!(contract_transfer);
	test_module_info_stats!(coremark_minimal);
	test_module_info_stats!(dns);
	test_module_info_stats!(erc1155);
	test_module_info_stats!(erc20);
	test_module_info_stats!(erc721);
	test_module_info_stats!(many_blocks);
	test_module_info_stats!(multisig);
	test_module_info_stats!(proxy);
	test_module_info_stats!(rand_extension);
	test_module_info_stats!(scrypto);
	test_module_info_stats!(trait_erc20);
	test_module_info_stats!(wasm_kernel);

	// #[test]
	// fn test_remove_export() {
	// 	let bytes = include_bytes!("../../_ex/everything_schema.wasm");
	// 	let mut module = ModuleInfo::new(bytes).unwrap();

	// 	module.remove_export("_schema").unwrap();

	// 	std::fs::write("/tmp/everything_schema_remove_schema.wasm", module.bytes()).unwrap();
	// }
}
