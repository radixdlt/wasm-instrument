use crate::utils::errors::TranslatorError;
use alloc::vec::Vec;
use wasm_encoder::*;
use wasmparser::{
	DataKind, ElementKind, ExternalKind, FunctionBody, Global, Import, Operator, Type,
};

type Result<T> = core::result::Result<T, TranslatorError>;

#[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub enum Item {
	Function,
	Table,
	Memory,
	Tag,
	Global,
	Type,
	Data,
	Element,
}

#[allow(dead_code)]
#[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub enum ConstExprKind {
	Global,
	ElementOffset,
	ElementFunction,
	DataOffset,
	TableInit,
}

pub trait Translator {
	fn as_obj(&mut self) -> &mut dyn Translator;

	fn translate_type_def(&mut self, ty: Type, s: &mut TypeSection) -> Result<()> {
		type_def(self.as_obj(), ty, s)
	}

	fn translate_import(&mut self, import: Import, s: &mut ImportSection) -> Result<()> {
		import_def(self.as_obj(), import, s)
	}

	fn translate_table_type(
		&mut self,
		ty: &wasmparser::TableType,
	) -> Result<wasm_encoder::TableType> {
		table_type(self.as_obj(), ty)
	}

	fn translate_memory_type(
		&mut self,
		ty: &wasmparser::MemoryType,
	) -> Result<wasm_encoder::MemoryType> {
		memory_type(self.as_obj(), ty)
	}

	fn translate_global_type(
		&mut self,
		ty: &wasmparser::GlobalType,
	) -> Result<wasm_encoder::GlobalType> {
		global_type(self.as_obj(), ty)
	}

	fn translate_tag_type(&mut self, ty: &wasmparser::TagType) -> Result<wasm_encoder::TagType> {
		tag_type(self.as_obj(), ty)
	}

	fn translate_ty(&mut self, t: &wasmparser::ValType) -> Result<ValType> {
		ty(self.as_obj(), t)
	}

	fn translate_ref_ty(&mut self, t: &wasmparser::RefType) -> Result<RefType> {
		ref_ty(self.as_obj(), t)
	}

	fn translate_heap_ty(&mut self, t: &wasmparser::HeapType) -> Result<HeapType> {
		heap_ty(self.as_obj(), t)
	}

	fn translate_global(&mut self, g: Global, s: &mut GlobalSection) -> Result<()> {
		global(self.as_obj(), g, s)
	}

	fn translate_export_kind(
		&mut self,
		g: wasmparser::ExternalKind,
	) -> Result<wasm_encoder::ExportKind> {
		export_kind(self.as_obj(), g)
	}

	fn translate_export(
		&mut self,
		e: &wasmparser::Export,
		sec: &mut wasm_encoder::ExportSection,
	) -> Result<()> {
		export(self.as_obj(), e, sec)
	}

	fn translate_const_expr(
		&mut self,
		e: &wasmparser::ConstExpr<'_>,
		_ty: &wasmparser::ValType,
		ctx: ConstExprKind,
	) -> Result<wasm_encoder::ConstExpr> {
		const_expr(self.as_obj(), e, ctx)
	}

	fn translate_element(
		&mut self,
		e: wasmparser::Element<'_>,
		s: &mut ElementSection,
	) -> Result<()> {
		element(self.as_obj(), e, s)
	}

	fn translate_data(&mut self, d: wasmparser::Data<'_>, s: &mut DataSection) -> Result<()> {
		data(self.as_obj(), d, s)
	}

	fn translate_code(&mut self, body: FunctionBody<'_>, s: &mut CodeSection) -> Result<()> {
		code(self.as_obj(), body, s)
	}

	fn translate_op(&mut self, e: &Operator<'_>) -> Result<Instruction<'static>> {
		op(self.as_obj(), e)
	}

	fn translate_block_type(&mut self, ty: &wasmparser::BlockType) -> Result<BlockType> {
		block_type(self.as_obj(), ty)
	}

	fn translate_memarg(&mut self, arg: &wasmparser::MemArg) -> Result<MemArg> {
		memarg(self.as_obj(), arg)
	}
}

pub struct DefaultTranslator;

impl Translator for DefaultTranslator {
	fn as_obj(&mut self) -> &mut dyn Translator {
		self
	}
}

pub fn type_def(t: &mut dyn Translator, ty: Type, s: &mut TypeSection) -> Result<()> {
	match ty {
		Type::Func(f) => {
			s.function(
				f.params().iter().map(|ty| t.translate_ty(ty)).collect::<Result<Vec<_>>>()?,
				f.results().iter().map(|ty| t.translate_ty(ty)).collect::<Result<Vec<_>>>()?,
			);
			Ok(())
		},
		Type::Array(_) => unimplemented!("Array and struct types are not supported yet."),
	}
}

#[allow(dead_code)]
pub fn import_def(t: &mut dyn Translator, ty: Import, s: &mut ImportSection) -> Result<()> {
	let new_ty = match ty.ty {
		wasmparser::TypeRef::Func(v) => EntityType::Function(v),
		wasmparser::TypeRef::Tag(v) => EntityType::Tag(t.translate_tag_type(&v)?),
		wasmparser::TypeRef::Global(v) => EntityType::Global(t.translate_global_type(&v)?),
		wasmparser::TypeRef::Table(v) => EntityType::Table(t.translate_table_type(&v)?),
		wasmparser::TypeRef::Memory(v) => EntityType::Memory(t.translate_memory_type(&v)?),
	};
	s.import(ty.module, ty.name, new_ty);
	Ok(())
}

#[allow(dead_code)]
pub fn table_type(
	t: &mut dyn Translator,
	ty: &wasmparser::TableType,
) -> Result<wasm_encoder::TableType> {
	Ok(wasm_encoder::TableType {
		element_type: t.translate_ref_ty(&ty.element_type)?,
		minimum: ty.initial,
		maximum: ty.maximum,
	})
}

#[allow(dead_code)]
pub fn memory_type(
	_t: &mut dyn Translator,
	ty: &wasmparser::MemoryType,
) -> Result<wasm_encoder::MemoryType> {
	Ok(wasm_encoder::MemoryType {
		memory64: ty.memory64,
		minimum: ty.initial,
		maximum: ty.maximum,
		shared: ty.shared,
	})
}

pub fn global_type(
	t: &mut dyn Translator,
	ty: &wasmparser::GlobalType,
) -> Result<wasm_encoder::GlobalType> {
	Ok(wasm_encoder::GlobalType {
		val_type: t.translate_ty(&ty.content_type)?,
		mutable: ty.mutable,
	})
}

#[allow(dead_code)]
pub fn tag_type(
	_t: &mut dyn Translator,
	ty: &wasmparser::TagType,
) -> Result<wasm_encoder::TagType> {
	Ok(wasm_encoder::TagType { kind: TagKind::Exception, func_type_idx: ty.func_type_idx })
}

pub fn ty(t: &mut dyn Translator, ty: &wasmparser::ValType) -> Result<ValType> {
	match ty {
		wasmparser::ValType::I32 => Ok(ValType::I32),
		wasmparser::ValType::I64 => Ok(ValType::I64),
		wasmparser::ValType::F32 => Ok(ValType::F32),
		wasmparser::ValType::F64 => Ok(ValType::F64),
		wasmparser::ValType::V128 => Ok(ValType::V128),
		wasmparser::ValType::Ref(ty) => Ok(ValType::Ref(t.translate_ref_ty(ty)?)),
	}
}

pub fn ref_ty(t: &mut dyn Translator, ty: &wasmparser::RefType) -> Result<RefType> {
	Ok(RefType { nullable: ty.is_nullable(), heap_type: t.translate_heap_ty(&ty.heap_type())? })
}

pub fn heap_ty(_t: &mut dyn Translator, ty: &wasmparser::HeapType) -> Result<HeapType> {
	match ty {
		wasmparser::HeapType::Func => Ok(HeapType::Func),
		wasmparser::HeapType::Extern => Ok(HeapType::Extern),
		wasmparser::HeapType::Any => Ok(HeapType::Any),
		wasmparser::HeapType::None => Ok(HeapType::None),
		wasmparser::HeapType::NoExtern => Ok(HeapType::NoExtern),
		wasmparser::HeapType::NoFunc => Ok(HeapType::NoFunc),
		wasmparser::HeapType::Eq => Ok(HeapType::Eq),
		wasmparser::HeapType::Struct => Ok(HeapType::Struct),
		wasmparser::HeapType::Array => Ok(HeapType::Array),
		wasmparser::HeapType::I31 => Ok(HeapType::I31),
		wasmparser::HeapType::Indexed(i) => Ok(HeapType::Indexed(*i)),
	}
}

pub fn global(t: &mut dyn Translator, global: Global, s: &mut GlobalSection) -> Result<()> {
	let ty = t.translate_global_type(&global.ty)?;
	let insn =
		t.translate_const_expr(&global.init_expr, &global.ty.content_type, ConstExprKind::Global)?;
	s.global(ty, &insn);
	Ok(())
}

pub fn export_kind(_: &dyn Translator, kind: ExternalKind) -> Result<ExportKind> {
	match kind {
		ExternalKind::Table => Ok(ExportKind::Table),
		ExternalKind::Global => Ok(ExportKind::Global),
		ExternalKind::Tag => Ok(ExportKind::Tag),
		ExternalKind::Func => Ok(ExportKind::Func),
		ExternalKind::Memory => Ok(ExportKind::Memory),
	}
}

#[allow(unused)]
pub fn export(
	t: &mut dyn Translator,
	e: &wasmparser::Export<'_>,
	sec: &mut wasm_encoder::ExportSection,
) -> Result<()> {
	sec.export(e.name, t.translate_export_kind(e.kind)?, e.index);
	Ok(())
}

pub fn const_expr(
	t: &mut dyn Translator,
	e: &wasmparser::ConstExpr<'_>,
	ctx: ConstExprKind,
) -> Result<wasm_encoder::ConstExpr> {
	let mut e = e.get_operators_reader();
	let mut offset_bytes = Vec::new();
	let op = e.read()?;
	if let ConstExprKind::ElementFunction = ctx {
		match op {
			Operator::RefFunc { .. } |
			Operator::RefNull { hty: wasmparser::HeapType::Func, .. } |
			Operator::GlobalGet { .. } => {},
			_ => return Err(TranslatorError::NoMutationsApplicable),
		}
	}
	t.translate_op(&op)?.encode(&mut offset_bytes);
	match e.read()? {
		Operator::End if e.eof() => {},
		_ => return Err(TranslatorError::NoMutationsApplicable),
	}
	Ok(wasm_encoder::ConstExpr::raw(offset_bytes))
}

#[allow(dead_code)]
pub fn element(
	t: &mut dyn Translator,
	element: wasmparser::Element<'_>,
	s: &mut ElementSection,
) -> Result<()> {
	let offset;
	let mode = match &element.kind {
		ElementKind::Active { table_index, offset_expr } => {
			offset = t.translate_const_expr(
				offset_expr,
				&wasmparser::ValType::I32,
				ConstExprKind::ElementOffset,
			)?;
			ElementMode::Active { table: *table_index, offset: &offset }
		},
		ElementKind::Passive => ElementMode::Passive,
		ElementKind::Declared => ElementMode::Declared,
	};
	let element_type = t.translate_ref_ty(&element.ty)?;
	let functions;
	let exprs;
	let elements = match element.items {
		wasmparser::ElementItems::Functions(reader) => {
			functions = reader.into_iter().collect::<wasmparser::Result<Vec<_>, _>>()?;
			Elements::Functions(&functions)
		},
		wasmparser::ElementItems::Expressions(reader) => {
			exprs = reader
				.into_iter()
				.map(|f| {
					t.translate_const_expr(
						&f?,
						&wasmparser::ValType::Ref(element.ty),
						ConstExprKind::ElementFunction,
					)
				})
				.collect::<wasmparser::Result<Vec<_>, _>>()?;
			Elements::Expressions(&exprs)
		},
	};
	s.segment(ElementSegment { mode, element_type, elements });
	Ok(())
}

/// This is a pretty gnarly function that translates from `wasmparser`
/// operators to `wasm_encoder` operators. It's quite large because there's
/// quite a few wasm instructions. The theory though is that at least each
/// individual case is pretty self-contained.
pub fn op(t: &mut dyn Translator, op: &Operator<'_>) -> Result<Instruction<'static>> {
	use wasm_encoder::Instruction as I;
	use wasmparser::Operator as O;
	Ok(match op {
		O::Unreachable => I::Unreachable,
		O::Nop => I::Nop,

		O::Block { blockty } => I::Block(t.translate_block_type(blockty)?),
		O::Loop { blockty } => I::Loop(t.translate_block_type(blockty)?),
		O::If { blockty } => I::If(t.translate_block_type(blockty)?),
		O::Else => I::Else,

		O::Try { blockty } => I::Try(t.translate_block_type(blockty)?),
		O::Catch { tag_index } => I::Catch(*tag_index),
		O::Throw { tag_index } => I::Throw(*tag_index),
		O::Rethrow { relative_depth } => I::Rethrow(*relative_depth),
		O::End => I::End,
		O::Br { relative_depth } => I::Br(*relative_depth),
		O::BrIf { relative_depth } => I::BrIf(*relative_depth),
		O::BrTable { targets } => I::BrTable(
			targets.targets().collect::<wasmparser::Result<Vec<_>, _>>()?.into(),
			targets.default(),
		),

		O::Return => I::Return,
		O::Call { function_index } => I::Call(*function_index),
		O::CallIndirect { type_index, table_index, table_byte: _ } =>
			I::CallIndirect { ty: *type_index, table: *table_index },
		O::ReturnCall { function_index } => I::ReturnCall(*function_index),
		O::ReturnCallIndirect { type_index, table_index } =>
			I::ReturnCallIndirect { ty: *type_index, table: *table_index },
		O::Delegate { relative_depth } => I::Delegate(*relative_depth),
		O::CatchAll => I::CatchAll,
		O::Drop => I::Drop,
		O::Select => I::Select,
		O::TypedSelect { ty } => I::TypedSelect(t.translate_ty(ty)?),

		O::LocalGet { local_index } => I::LocalGet(*local_index),
		O::LocalSet { local_index } => I::LocalSet(*local_index),
		O::LocalTee { local_index } => I::LocalTee(*local_index),

		O::GlobalGet { global_index } => I::GlobalGet(*global_index),
		O::GlobalSet { global_index } => I::GlobalSet(*global_index),

		O::I32Load { memarg } => I::I32Load(t.translate_memarg(memarg)?),
		O::I64Load { memarg } => I::I64Load(t.translate_memarg(memarg)?),
		O::F32Load { memarg } => I::F32Load(t.translate_memarg(memarg)?),
		O::F64Load { memarg } => I::F64Load(t.translate_memarg(memarg)?),
		O::I32Load8S { memarg } => I::I32Load8S(t.translate_memarg(memarg)?),
		O::I32Load8U { memarg } => I::I32Load8U(t.translate_memarg(memarg)?),
		O::I32Load16S { memarg } => I::I32Load16S(t.translate_memarg(memarg)?),
		O::I32Load16U { memarg } => I::I32Load16U(t.translate_memarg(memarg)?),
		O::I64Load8S { memarg } => I::I64Load8S(t.translate_memarg(memarg)?),
		O::I64Load8U { memarg } => I::I64Load8U(t.translate_memarg(memarg)?),
		O::I64Load16S { memarg } => I::I64Load16S(t.translate_memarg(memarg)?),
		O::I64Load16U { memarg } => I::I64Load16U(t.translate_memarg(memarg)?),
		O::I64Load32S { memarg } => I::I64Load32S(t.translate_memarg(memarg)?),
		O::I64Load32U { memarg } => I::I64Load32U(t.translate_memarg(memarg)?),
		O::I32Store { memarg } => I::I32Store(t.translate_memarg(memarg)?),
		O::I64Store { memarg } => I::I64Store(t.translate_memarg(memarg)?),
		O::F32Store { memarg } => I::F32Store(t.translate_memarg(memarg)?),
		O::F64Store { memarg } => I::F64Store(t.translate_memarg(memarg)?),
		O::I32Store8 { memarg } => I::I32Store8(t.translate_memarg(memarg)?),
		O::I32Store16 { memarg } => I::I32Store16(t.translate_memarg(memarg)?),
		O::I64Store8 { memarg } => I::I64Store8(t.translate_memarg(memarg)?),
		O::I64Store16 { memarg } => I::I64Store16(t.translate_memarg(memarg)?),
		O::I64Store32 { memarg } => I::I64Store32(t.translate_memarg(memarg)?),

		O::MemorySize { mem, .. } => I::MemorySize(*mem),
		O::MemoryGrow { mem, .. } => I::MemoryGrow(*mem),

		O::I32Const { value } => I::I32Const(*value),
		O::I64Const { value } => I::I64Const(*value),
		O::F32Const { value } => I::F32Const(f32::from_bits(value.bits())),
		O::F64Const { value } => I::F64Const(f64::from_bits(value.bits())),

		O::RefNull { hty } => I::RefNull(t.translate_heap_ty(hty)?),
		O::RefIsNull => I::RefIsNull,
		O::RefFunc { function_index } => I::RefFunc(*function_index),

		O::I31New => I::I31New,
		O::I31GetS => I::I31GetS,
		O::I31GetU => I::I31GetU,

		O::I32Eqz => I::I32Eqz,
		O::I32Eq => I::I32Eq,
		O::I32Ne => I::I32Ne,
		O::I32LtS => I::I32LtS,
		O::I32LtU => I::I32LtU,
		O::I32GtS => I::I32GtS,
		O::I32GtU => I::I32GtU,
		O::I32LeS => I::I32LeS,
		O::I32LeU => I::I32LeU,
		O::I32GeS => I::I32GeS,
		O::I32GeU => I::I32GeU,
		O::I64Eqz => I::I64Eqz,
		O::I64Eq => I::I64Eq,
		O::I64Ne => I::I64Ne,
		O::I64LtS => I::I64LtS,
		O::I64LtU => I::I64LtU,
		O::I64GtS => I::I64GtS,
		O::I64GtU => I::I64GtU,
		O::I64LeS => I::I64LeS,
		O::I64LeU => I::I64LeU,
		O::I64GeS => I::I64GeS,
		O::I64GeU => I::I64GeU,
		O::F32Eq => I::F32Eq,
		O::F32Ne => I::F32Ne,
		O::F32Lt => I::F32Lt,
		O::F32Gt => I::F32Gt,
		O::F32Le => I::F32Le,
		O::F32Ge => I::F32Ge,
		O::F64Eq => I::F64Eq,
		O::F64Ne => I::F64Ne,
		O::F64Lt => I::F64Lt,
		O::F64Gt => I::F64Gt,
		O::F64Le => I::F64Le,
		O::F64Ge => I::F64Ge,
		O::I32Clz => I::I32Clz,
		O::I32Ctz => I::I32Ctz,
		O::I32Popcnt => I::I32Popcnt,
		O::I32Add => I::I32Add,
		O::I32Sub => I::I32Sub,
		O::I32Mul => I::I32Mul,
		O::I32DivS => I::I32DivS,
		O::I32DivU => I::I32DivU,
		O::I32RemS => I::I32RemS,
		O::I32RemU => I::I32RemU,
		O::I32And => I::I32And,
		O::I32Or => I::I32Or,
		O::I32Xor => I::I32Xor,
		O::I32Shl => I::I32Shl,
		O::I32ShrS => I::I32ShrS,
		O::I32ShrU => I::I32ShrU,
		O::I32Rotl => I::I32Rotl,
		O::I32Rotr => I::I32Rotr,
		O::I64Clz => I::I64Clz,
		O::I64Ctz => I::I64Ctz,
		O::I64Popcnt => I::I64Popcnt,
		O::I64Add => I::I64Add,
		O::I64Sub => I::I64Sub,
		O::I64Mul => I::I64Mul,
		O::I64DivS => I::I64DivS,
		O::I64DivU => I::I64DivU,
		O::I64RemS => I::I64RemS,
		O::I64RemU => I::I64RemU,
		O::I64And => I::I64And,
		O::I64Or => I::I64Or,
		O::I64Xor => I::I64Xor,
		O::I64Shl => I::I64Shl,
		O::I64ShrS => I::I64ShrS,
		O::I64ShrU => I::I64ShrU,
		O::I64Rotl => I::I64Rotl,
		O::I64Rotr => I::I64Rotr,
		O::F32Abs => I::F32Abs,
		O::F32Neg => I::F32Neg,
		O::F32Ceil => I::F32Ceil,
		O::F32Floor => I::F32Floor,
		O::F32Trunc => I::F32Trunc,
		O::F32Nearest => I::F32Nearest,
		O::F32Sqrt => I::F32Sqrt,
		O::F32Add => I::F32Add,
		O::F32Sub => I::F32Sub,
		O::F32Mul => I::F32Mul,
		O::F32Div => I::F32Div,
		O::F32Min => I::F32Min,
		O::F32Max => I::F32Max,
		O::F32Copysign => I::F32Copysign,
		O::F64Abs => I::F64Abs,
		O::F64Neg => I::F64Neg,
		O::F64Ceil => I::F64Ceil,
		O::F64Floor => I::F64Floor,
		O::F64Trunc => I::F64Trunc,
		O::F64Nearest => I::F64Nearest,
		O::F64Sqrt => I::F64Sqrt,
		O::F64Add => I::F64Add,
		O::F64Sub => I::F64Sub,
		O::F64Mul => I::F64Mul,
		O::F64Div => I::F64Div,
		O::F64Min => I::F64Min,
		O::F64Max => I::F64Max,
		O::F64Copysign => I::F64Copysign,
		O::I32WrapI64 => I::I32WrapI64,
		O::I32TruncF32S => I::I32TruncF32S,
		O::I32TruncF32U => I::I32TruncF32U,
		O::I32TruncF64S => I::I32TruncF64S,
		O::I32TruncF64U => I::I32TruncF64U,
		O::I64ExtendI32S => I::I64ExtendI32S,
		O::I64ExtendI32U => I::I64ExtendI32U,
		O::I64TruncF32S => I::I64TruncF32S,
		O::I64TruncF32U => I::I64TruncF32U,
		O::I64TruncF64S => I::I64TruncF64S,
		O::I64TruncF64U => I::I64TruncF64U,
		O::F32ConvertI32S => I::F32ConvertI32S,
		O::F32ConvertI32U => I::F32ConvertI32U,
		O::F32ConvertI64S => I::F32ConvertI64S,
		O::F32ConvertI64U => I::F32ConvertI64U,
		O::F32DemoteF64 => I::F32DemoteF64,
		O::F64ConvertI32S => I::F64ConvertI32S,
		O::F64ConvertI32U => I::F64ConvertI32U,
		O::F64ConvertI64S => I::F64ConvertI64S,
		O::F64ConvertI64U => I::F64ConvertI64U,
		O::F64PromoteF32 => I::F64PromoteF32,
		O::I32ReinterpretF32 => I::I32ReinterpretF32,
		O::I64ReinterpretF64 => I::I64ReinterpretF64,
		O::F32ReinterpretI32 => I::F32ReinterpretI32,
		O::F64ReinterpretI64 => I::F64ReinterpretI64,
		O::I32Extend8S => I::I32Extend8S,
		O::I32Extend16S => I::I32Extend16S,
		O::I64Extend8S => I::I64Extend8S,
		O::I64Extend16S => I::I64Extend16S,
		O::I64Extend32S => I::I64Extend32S,

		O::I32TruncSatF32S => I::I32TruncSatF32S,
		O::I32TruncSatF32U => I::I32TruncSatF32U,
		O::I32TruncSatF64S => I::I32TruncSatF64S,
		O::I32TruncSatF64U => I::I32TruncSatF64U,
		O::I64TruncSatF32S => I::I64TruncSatF32S,
		O::I64TruncSatF32U => I::I64TruncSatF32U,
		O::I64TruncSatF64S => I::I64TruncSatF64S,
		O::I64TruncSatF64U => I::I64TruncSatF64U,

		O::MemoryInit { data_index, mem } => I::MemoryInit { data_index: *data_index, mem: *mem },
		O::DataDrop { data_index } => I::DataDrop(*data_index),
		O::MemoryCopy { dst_mem, src_mem } =>
			I::MemoryCopy { src_mem: *src_mem, dst_mem: *dst_mem },
		O::MemoryDiscard { mem } => I::MemoryDiscard(*mem),
		O::MemoryFill { mem, .. } => I::MemoryFill(*mem),

		O::TableInit { elem_index, table } =>
			I::TableInit { elem_index: *elem_index, table: *table },
		O::ElemDrop { elem_index } => I::ElemDrop(*elem_index),
		O::TableCopy { dst_table, src_table } =>
			I::TableCopy { dst_table: *dst_table, src_table: *src_table },
		O::TableFill { table } => I::TableFill(*table),
		O::TableGet { table } => I::TableGet(*table),
		O::TableSet { table } => I::TableSet(*table),
		O::TableGrow { table } => I::TableGrow(*table),
		O::TableSize { table } => I::TableSize(*table),

		O::V128Load { memarg } => I::V128Load(t.translate_memarg(memarg)?),
		O::V128Load8x8S { memarg } => I::V128Load8x8S(t.translate_memarg(memarg)?),
		O::V128Load8x8U { memarg } => I::V128Load8x8U(t.translate_memarg(memarg)?),
		O::V128Load16x4S { memarg } => I::V128Load16x4S(t.translate_memarg(memarg)?),
		O::V128Load16x4U { memarg } => I::V128Load16x4U(t.translate_memarg(memarg)?),
		O::V128Load32x2S { memarg } => I::V128Load32x2S(t.translate_memarg(memarg)?),
		O::V128Load32x2U { memarg } => I::V128Load32x2U(t.translate_memarg(memarg)?),
		O::V128Load8Splat { memarg } => I::V128Load8Splat(t.translate_memarg(memarg)?),
		O::V128Load16Splat { memarg } => I::V128Load16Splat(t.translate_memarg(memarg)?),
		O::V128Load32Splat { memarg } => I::V128Load32Splat(t.translate_memarg(memarg)?),
		O::V128Load64Splat { memarg } => I::V128Load64Splat(t.translate_memarg(memarg)?),
		O::V128Load32Zero { memarg } => I::V128Load32Zero(t.translate_memarg(memarg)?),
		O::V128Load64Zero { memarg } => I::V128Load64Zero(t.translate_memarg(memarg)?),
		O::V128Store { memarg } => I::V128Store(t.translate_memarg(memarg)?),
		O::V128Load8Lane { memarg, lane } =>
			I::V128Load8Lane { memarg: t.translate_memarg(memarg)?, lane: *lane },
		O::V128Load16Lane { memarg, lane } =>
			I::V128Load16Lane { memarg: t.translate_memarg(memarg)?, lane: *lane },
		O::V128Load32Lane { memarg, lane } =>
			I::V128Load32Lane { memarg: t.translate_memarg(memarg)?, lane: *lane },
		O::V128Load64Lane { memarg, lane } =>
			I::V128Load64Lane { memarg: t.translate_memarg(memarg)?, lane: *lane },
		O::V128Store8Lane { memarg, lane } =>
			I::V128Store8Lane { memarg: t.translate_memarg(memarg)?, lane: *lane },
		O::V128Store16Lane { memarg, lane } =>
			I::V128Store16Lane { memarg: t.translate_memarg(memarg)?, lane: *lane },
		O::V128Store32Lane { memarg, lane } =>
			I::V128Store32Lane { memarg: t.translate_memarg(memarg)?, lane: *lane },
		O::V128Store64Lane { memarg, lane } =>
			I::V128Store64Lane { memarg: t.translate_memarg(memarg)?, lane: *lane },

		O::V128Const { value } => I::V128Const(value.i128()),
		O::I8x16Shuffle { lanes } => I::I8x16Shuffle(*lanes),
		O::I8x16ExtractLaneS { lane } => I::I8x16ExtractLaneS(*lane),
		O::I8x16ExtractLaneU { lane } => I::I8x16ExtractLaneU(*lane),
		O::I8x16ReplaceLane { lane } => I::I8x16ReplaceLane(*lane),
		O::I16x8ExtractLaneS { lane } => I::I16x8ExtractLaneS(*lane),
		O::I16x8ExtractLaneU { lane } => I::I16x8ExtractLaneU(*lane),
		O::I16x8ReplaceLane { lane } => I::I16x8ReplaceLane(*lane),
		O::I32x4ExtractLane { lane } => I::I32x4ExtractLane(*lane),
		O::I32x4ReplaceLane { lane } => I::I32x4ReplaceLane(*lane),
		O::I64x2ExtractLane { lane } => I::I64x2ExtractLane(*lane),
		O::I64x2ReplaceLane { lane } => I::I64x2ReplaceLane(*lane),
		O::F32x4ExtractLane { lane } => I::F32x4ExtractLane(*lane),
		O::F32x4ReplaceLane { lane } => I::F32x4ReplaceLane(*lane),
		O::F64x2ExtractLane { lane } => I::F64x2ExtractLane(*lane),
		O::F64x2ReplaceLane { lane } => I::F64x2ReplaceLane(*lane),

		O::I8x16Swizzle => I::I8x16Swizzle,
		O::I8x16Splat => I::I8x16Splat,
		O::I16x8Splat => I::I16x8Splat,
		O::I32x4Splat => I::I32x4Splat,
		O::I64x2Splat => I::I64x2Splat,
		O::F32x4Splat => I::F32x4Splat,
		O::F64x2Splat => I::F64x2Splat,
		O::I8x16Eq => I::I8x16Eq,
		O::I8x16Ne => I::I8x16Ne,
		O::I8x16LtS => I::I8x16LtS,
		O::I8x16LtU => I::I8x16LtU,
		O::I8x16GtS => I::I8x16GtS,
		O::I8x16GtU => I::I8x16GtU,
		O::I8x16LeS => I::I8x16LeS,
		O::I8x16LeU => I::I8x16LeU,
		O::I8x16GeS => I::I8x16GeS,
		O::I8x16GeU => I::I8x16GeU,
		O::I16x8Eq => I::I16x8Eq,
		O::I16x8Ne => I::I16x8Ne,
		O::I16x8LtS => I::I16x8LtS,
		O::I16x8LtU => I::I16x8LtU,
		O::I16x8GtS => I::I16x8GtS,
		O::I16x8GtU => I::I16x8GtU,
		O::I16x8LeS => I::I16x8LeS,
		O::I16x8LeU => I::I16x8LeU,
		O::I16x8GeS => I::I16x8GeS,
		O::I16x8GeU => I::I16x8GeU,
		O::I32x4Eq => I::I32x4Eq,
		O::I32x4Ne => I::I32x4Ne,
		O::I32x4LtS => I::I32x4LtS,
		O::I32x4LtU => I::I32x4LtU,
		O::I32x4GtS => I::I32x4GtS,
		O::I32x4GtU => I::I32x4GtU,
		O::I32x4LeS => I::I32x4LeS,
		O::I32x4LeU => I::I32x4LeU,
		O::I32x4GeS => I::I32x4GeS,
		O::I32x4GeU => I::I32x4GeU,
		O::I64x2Eq => I::I64x2Eq,
		O::I64x2Ne => I::I64x2Ne,
		O::I64x2LtS => I::I64x2LtS,
		O::I64x2GtS => I::I64x2GtS,
		O::I64x2LeS => I::I64x2LeS,
		O::I64x2GeS => I::I64x2GeS,
		O::F32x4Eq => I::F32x4Eq,
		O::F32x4Ne => I::F32x4Ne,
		O::F32x4Lt => I::F32x4Lt,
		O::F32x4Gt => I::F32x4Gt,
		O::F32x4Le => I::F32x4Le,
		O::F32x4Ge => I::F32x4Ge,
		O::F64x2Eq => I::F64x2Eq,
		O::F64x2Ne => I::F64x2Ne,
		O::F64x2Lt => I::F64x2Lt,
		O::F64x2Gt => I::F64x2Gt,
		O::F64x2Le => I::F64x2Le,
		O::F64x2Ge => I::F64x2Ge,
		O::V128Not => I::V128Not,
		O::V128And => I::V128And,
		O::V128AndNot => I::V128AndNot,
		O::V128Or => I::V128Or,
		O::V128Xor => I::V128Xor,
		O::V128Bitselect => I::V128Bitselect,
		O::V128AnyTrue => I::V128AnyTrue,
		O::I8x16Abs => I::I8x16Abs,
		O::I8x16Neg => I::I8x16Neg,
		O::I8x16Popcnt => I::I8x16Popcnt,
		O::I8x16AllTrue => I::I8x16AllTrue,
		O::I8x16Bitmask => I::I8x16Bitmask,
		O::I8x16NarrowI16x8S => I::I8x16NarrowI16x8S,
		O::I8x16NarrowI16x8U => I::I8x16NarrowI16x8U,
		O::I8x16Shl => I::I8x16Shl,
		O::I8x16ShrS => I::I8x16ShrS,
		O::I8x16ShrU => I::I8x16ShrU,
		O::I8x16Add => I::I8x16Add,
		O::I8x16AddSatS => I::I8x16AddSatS,
		O::I8x16AddSatU => I::I8x16AddSatU,
		O::I8x16Sub => I::I8x16Sub,
		O::I8x16SubSatS => I::I8x16SubSatS,
		O::I8x16SubSatU => I::I8x16SubSatU,
		O::I8x16MinS => I::I8x16MinS,
		O::I8x16MinU => I::I8x16MinU,
		O::I8x16MaxS => I::I8x16MaxS,
		O::I8x16MaxU => I::I8x16MaxU,
		O::I8x16AvgrU => I::I8x16AvgrU,
		O::I16x8ExtAddPairwiseI8x16S => I::I16x8ExtAddPairwiseI8x16S,
		O::I16x8ExtAddPairwiseI8x16U => I::I16x8ExtAddPairwiseI8x16U,
		O::I16x8Abs => I::I16x8Abs,
		O::I16x8Neg => I::I16x8Neg,
		O::I16x8Q15MulrSatS => I::I16x8Q15MulrSatS,
		O::I16x8AllTrue => I::I16x8AllTrue,
		O::I16x8Bitmask => I::I16x8Bitmask,
		O::I16x8NarrowI32x4S => I::I16x8NarrowI32x4S,
		O::I16x8NarrowI32x4U => I::I16x8NarrowI32x4U,
		O::I16x8ExtendLowI8x16S => I::I16x8ExtendLowI8x16S,
		O::I16x8ExtendHighI8x16S => I::I16x8ExtendHighI8x16S,
		O::I16x8ExtendLowI8x16U => I::I16x8ExtendLowI8x16U,
		O::I16x8ExtendHighI8x16U => I::I16x8ExtendHighI8x16U,
		O::I16x8Shl => I::I16x8Shl,
		O::I16x8ShrS => I::I16x8ShrS,
		O::I16x8ShrU => I::I16x8ShrU,
		O::I16x8Add => I::I16x8Add,
		O::I16x8AddSatS => I::I16x8AddSatS,
		O::I16x8AddSatU => I::I16x8AddSatU,
		O::I16x8Sub => I::I16x8Sub,
		O::I16x8SubSatS => I::I16x8SubSatS,
		O::I16x8SubSatU => I::I16x8SubSatU,
		O::I16x8Mul => I::I16x8Mul,
		O::I16x8MinS => I::I16x8MinS,
		O::I16x8MinU => I::I16x8MinU,
		O::I16x8MaxS => I::I16x8MaxS,
		O::I16x8MaxU => I::I16x8MaxU,
		O::I16x8AvgrU => I::I16x8AvgrU,
		O::I16x8ExtMulLowI8x16S => I::I16x8ExtMulLowI8x16S,
		O::I16x8ExtMulHighI8x16S => I::I16x8ExtMulHighI8x16S,
		O::I16x8ExtMulLowI8x16U => I::I16x8ExtMulLowI8x16U,
		O::I16x8ExtMulHighI8x16U => I::I16x8ExtMulHighI8x16U,
		O::I32x4ExtAddPairwiseI16x8S => I::I32x4ExtAddPairwiseI16x8S,
		O::I32x4ExtAddPairwiseI16x8U => I::I32x4ExtAddPairwiseI16x8U,
		O::I32x4Abs => I::I32x4Abs,
		O::I32x4Neg => I::I32x4Neg,
		O::I32x4AllTrue => I::I32x4AllTrue,
		O::I32x4Bitmask => I::I32x4Bitmask,
		O::I32x4ExtendLowI16x8S => I::I32x4ExtendLowI16x8S,
		O::I32x4ExtendHighI16x8S => I::I32x4ExtendHighI16x8S,
		O::I32x4ExtendLowI16x8U => I::I32x4ExtendLowI16x8U,
		O::I32x4ExtendHighI16x8U => I::I32x4ExtendHighI16x8U,
		O::I32x4Shl => I::I32x4Shl,
		O::I32x4ShrS => I::I32x4ShrS,
		O::I32x4ShrU => I::I32x4ShrU,
		O::I32x4Add => I::I32x4Add,
		O::I32x4Sub => I::I32x4Sub,
		O::I32x4Mul => I::I32x4Mul,
		O::I32x4MinS => I::I32x4MinS,
		O::I32x4MinU => I::I32x4MinU,
		O::I32x4MaxS => I::I32x4MaxS,
		O::I32x4MaxU => I::I32x4MaxU,
		O::I32x4DotI16x8S => I::I32x4DotI16x8S,
		O::I32x4ExtMulLowI16x8S => I::I32x4ExtMulLowI16x8S,
		O::I32x4ExtMulHighI16x8S => I::I32x4ExtMulHighI16x8S,
		O::I32x4ExtMulLowI16x8U => I::I32x4ExtMulLowI16x8U,
		O::I32x4ExtMulHighI16x8U => I::I32x4ExtMulHighI16x8U,
		O::I64x2Abs => I::I64x2Abs,
		O::I64x2Neg => I::I64x2Neg,
		O::I64x2AllTrue => I::I64x2AllTrue,
		O::I64x2Bitmask => I::I64x2Bitmask,
		O::I64x2ExtendLowI32x4S => I::I64x2ExtendLowI32x4S,
		O::I64x2ExtendHighI32x4S => I::I64x2ExtendHighI32x4S,
		O::I64x2ExtendLowI32x4U => I::I64x2ExtendLowI32x4U,
		O::I64x2ExtendHighI32x4U => I::I64x2ExtendHighI32x4U,
		O::I64x2Shl => I::I64x2Shl,
		O::I64x2ShrS => I::I64x2ShrS,
		O::I64x2ShrU => I::I64x2ShrU,
		O::I64x2Add => I::I64x2Add,
		O::I64x2Sub => I::I64x2Sub,
		O::I64x2Mul => I::I64x2Mul,
		O::I64x2ExtMulLowI32x4S => I::I64x2ExtMulLowI32x4S,
		O::I64x2ExtMulHighI32x4S => I::I64x2ExtMulHighI32x4S,
		O::I64x2ExtMulLowI32x4U => I::I64x2ExtMulLowI32x4U,
		O::I64x2ExtMulHighI32x4U => I::I64x2ExtMulHighI32x4U,
		O::F32x4Ceil => I::F32x4Ceil,
		O::F32x4Floor => I::F32x4Floor,
		O::F32x4Trunc => I::F32x4Trunc,
		O::F32x4Nearest => I::F32x4Nearest,
		O::F32x4Abs => I::F32x4Abs,
		O::F32x4Neg => I::F32x4Neg,
		O::F32x4Sqrt => I::F32x4Sqrt,
		O::F32x4Add => I::F32x4Add,
		O::F32x4Sub => I::F32x4Sub,
		O::F32x4Mul => I::F32x4Mul,
		O::F32x4Div => I::F32x4Div,
		O::F32x4Min => I::F32x4Min,
		O::F32x4Max => I::F32x4Max,
		O::F32x4PMin => I::F32x4PMin,
		O::F32x4PMax => I::F32x4PMax,
		O::F64x2Ceil => I::F64x2Ceil,
		O::F64x2Floor => I::F64x2Floor,
		O::F64x2Trunc => I::F64x2Trunc,
		O::F64x2Nearest => I::F64x2Nearest,
		O::F64x2Abs => I::F64x2Abs,
		O::F64x2Neg => I::F64x2Neg,
		O::F64x2Sqrt => I::F64x2Sqrt,
		O::F64x2Add => I::F64x2Add,
		O::F64x2Sub => I::F64x2Sub,
		O::F64x2Mul => I::F64x2Mul,
		O::F64x2Div => I::F64x2Div,
		O::F64x2Min => I::F64x2Min,
		O::F64x2Max => I::F64x2Max,
		O::F64x2PMin => I::F64x2PMin,
		O::F64x2PMax => I::F64x2PMax,
		O::I32x4TruncSatF32x4S => I::I32x4TruncSatF32x4S,
		O::I32x4TruncSatF32x4U => I::I32x4TruncSatF32x4U,
		O::F32x4ConvertI32x4S => I::F32x4ConvertI32x4S,
		O::F32x4ConvertI32x4U => I::F32x4ConvertI32x4U,
		O::I32x4TruncSatF64x2SZero => I::I32x4TruncSatF64x2SZero,
		O::I32x4TruncSatF64x2UZero => I::I32x4TruncSatF64x2UZero,
		O::F64x2ConvertLowI32x4S => I::F64x2ConvertLowI32x4S,
		O::F64x2ConvertLowI32x4U => I::F64x2ConvertLowI32x4U,
		O::F32x4DemoteF64x2Zero => I::F32x4DemoteF64x2Zero,
		O::F64x2PromoteLowF32x4 => I::F64x2PromoteLowF32x4,
		O::I8x16RelaxedSwizzle => I::I8x16RelaxedSwizzle,
		O::I32x4RelaxedTruncF32x4S => I::I32x4RelaxedTruncF32x4S,
		O::I32x4RelaxedTruncF32x4U => I::I32x4RelaxedTruncF32x4U,
		O::I32x4RelaxedTruncF64x2SZero => I::I32x4RelaxedTruncF64x2SZero,
		O::I32x4RelaxedTruncF64x2UZero => I::I32x4RelaxedTruncF64x2UZero,
		O::F32x4RelaxedMadd => I::F32x4RelaxedMadd,
		O::F32x4RelaxedNmadd => I::F32x4RelaxedNmadd,
		O::F64x2RelaxedMadd => I::F64x2RelaxedMadd,
		O::F64x2RelaxedNmadd => I::F64x2RelaxedNmadd,
		O::I8x16RelaxedLaneselect => I::I8x16RelaxedLaneselect,
		O::I16x8RelaxedLaneselect => I::I16x8RelaxedLaneselect,
		O::I32x4RelaxedLaneselect => I::I32x4RelaxedLaneselect,
		O::I64x2RelaxedLaneselect => I::I64x2RelaxedLaneselect,
		O::F32x4RelaxedMin => I::F32x4RelaxedMin,
		O::F32x4RelaxedMax => I::F32x4RelaxedMax,
		O::F64x2RelaxedMin => I::F64x2RelaxedMin,
		O::F64x2RelaxedMax => I::F64x2RelaxedMax,
		O::I16x8RelaxedQ15mulrS => I::I16x8RelaxedQ15mulrS,
		O::I16x8RelaxedDotI8x16I7x16S => I::I16x8RelaxedDotI8x16I7x16S,
		O::I32x4RelaxedDotI8x16I7x16AddS => I::I32x4RelaxedDotI8x16I7x16AddS,

		O::CallRef { type_index } => I::CallRef(*type_index),
		O::ReturnCallRef { type_index } => I::ReturnCallRef(*type_index),
		O::RefAsNonNull => I::RefAsNonNull,
		O::BrOnNull { relative_depth } => I::BrOnNull(*relative_depth),
		O::BrOnNonNull { relative_depth } => I::BrOnNonNull(*relative_depth),

		O::MemoryAtomicNotify { memarg } => I::MemoryAtomicNotify(t.translate_memarg(memarg)?),
		O::MemoryAtomicWait32 { memarg } => I::MemoryAtomicWait32(t.translate_memarg(memarg)?),
		O::MemoryAtomicWait64 { memarg } => I::MemoryAtomicWait64(t.translate_memarg(memarg)?),
		O::I32AtomicLoad { memarg } => I::I32AtomicLoad(t.translate_memarg(memarg)?),
		O::I32AtomicLoad8U { memarg } => I::I32AtomicLoad8U(t.translate_memarg(memarg)?),
		O::I32AtomicLoad16U { memarg } => I::I32AtomicLoad16U(t.translate_memarg(memarg)?),
		O::I64AtomicLoad { memarg } => I::I64AtomicLoad(t.translate_memarg(memarg)?),
		O::I64AtomicLoad8U { memarg } => I::I64AtomicLoad8U(t.translate_memarg(memarg)?),
		O::I64AtomicLoad16U { memarg } => I::I64AtomicLoad16U(t.translate_memarg(memarg)?),
		O::I64AtomicLoad32U { memarg } => I::I64AtomicLoad32U(t.translate_memarg(memarg)?),
		O::I32AtomicStore { memarg } => I::I32AtomicStore(t.translate_memarg(memarg)?),
		O::I64AtomicStore { memarg } => I::I64AtomicStore(t.translate_memarg(memarg)?),
		O::I32AtomicStore8 { memarg } => I::I32AtomicStore8(t.translate_memarg(memarg)?),
		O::I32AtomicStore16 { memarg } => I::I32AtomicStore16(t.translate_memarg(memarg)?),
		O::I64AtomicStore8 { memarg } => I::I64AtomicStore8(t.translate_memarg(memarg)?),
		O::I64AtomicStore16 { memarg } => I::I64AtomicStore16(t.translate_memarg(memarg)?),
		O::I64AtomicStore32 { memarg } => I::I64AtomicStore32(t.translate_memarg(memarg)?),
		O::I32AtomicRmwAdd { memarg } => I::I32AtomicRmwAdd(t.translate_memarg(memarg)?),
		O::I64AtomicRmwAdd { memarg } => I::I64AtomicRmwAdd(t.translate_memarg(memarg)?),
		O::I32AtomicRmw8AddU { memarg } => I::I32AtomicRmw8AddU(t.translate_memarg(memarg)?),
		O::I32AtomicRmw16AddU { memarg } => I::I32AtomicRmw16AddU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw8AddU { memarg } => I::I64AtomicRmw8AddU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw16AddU { memarg } => I::I64AtomicRmw16AddU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw32AddU { memarg } => I::I64AtomicRmw32AddU(t.translate_memarg(memarg)?),
		O::I32AtomicRmwSub { memarg } => I::I32AtomicRmwSub(t.translate_memarg(memarg)?),
		O::I64AtomicRmwSub { memarg } => I::I64AtomicRmwSub(t.translate_memarg(memarg)?),
		O::I32AtomicRmw8SubU { memarg } => I::I32AtomicRmw8SubU(t.translate_memarg(memarg)?),
		O::I32AtomicRmw16SubU { memarg } => I::I32AtomicRmw16SubU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw8SubU { memarg } => I::I64AtomicRmw8SubU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw16SubU { memarg } => I::I64AtomicRmw16SubU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw32SubU { memarg } => I::I64AtomicRmw32SubU(t.translate_memarg(memarg)?),
		O::I32AtomicRmwAnd { memarg } => I::I32AtomicRmwAnd(t.translate_memarg(memarg)?),
		O::I64AtomicRmwAnd { memarg } => I::I64AtomicRmwAnd(t.translate_memarg(memarg)?),
		O::I32AtomicRmw8AndU { memarg } => I::I32AtomicRmw8AndU(t.translate_memarg(memarg)?),
		O::I32AtomicRmw16AndU { memarg } => I::I32AtomicRmw16AndU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw8AndU { memarg } => I::I64AtomicRmw8AndU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw16AndU { memarg } => I::I64AtomicRmw16AndU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw32AndU { memarg } => I::I64AtomicRmw32AndU(t.translate_memarg(memarg)?),
		O::I32AtomicRmwOr { memarg } => I::I32AtomicRmwOr(t.translate_memarg(memarg)?),
		O::I64AtomicRmwOr { memarg } => I::I64AtomicRmwOr(t.translate_memarg(memarg)?),
		O::I32AtomicRmw8OrU { memarg } => I::I32AtomicRmw8OrU(t.translate_memarg(memarg)?),
		O::I32AtomicRmw16OrU { memarg } => I::I32AtomicRmw16OrU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw8OrU { memarg } => I::I64AtomicRmw8OrU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw16OrU { memarg } => I::I64AtomicRmw16OrU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw32OrU { memarg } => I::I64AtomicRmw32OrU(t.translate_memarg(memarg)?),
		O::I32AtomicRmwXor { memarg } => I::I32AtomicRmwXor(t.translate_memarg(memarg)?),
		O::I64AtomicRmwXor { memarg } => I::I64AtomicRmwXor(t.translate_memarg(memarg)?),
		O::I32AtomicRmw8XorU { memarg } => I::I32AtomicRmw8XorU(t.translate_memarg(memarg)?),
		O::I32AtomicRmw16XorU { memarg } => I::I32AtomicRmw16XorU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw8XorU { memarg } => I::I64AtomicRmw8XorU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw16XorU { memarg } => I::I64AtomicRmw16XorU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw32XorU { memarg } => I::I64AtomicRmw32XorU(t.translate_memarg(memarg)?),
		O::I32AtomicRmwXchg { memarg } => I::I32AtomicRmwXchg(t.translate_memarg(memarg)?),
		O::I64AtomicRmwXchg { memarg } => I::I64AtomicRmwXchg(t.translate_memarg(memarg)?),
		O::I32AtomicRmw8XchgU { memarg } => I::I32AtomicRmw8XchgU(t.translate_memarg(memarg)?),
		O::I32AtomicRmw16XchgU { memarg } => I::I32AtomicRmw16XchgU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw8XchgU { memarg } => I::I64AtomicRmw8XchgU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw16XchgU { memarg } => I::I64AtomicRmw16XchgU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw32XchgU { memarg } => I::I64AtomicRmw32XchgU(t.translate_memarg(memarg)?),
		O::I32AtomicRmwCmpxchg { memarg } => I::I32AtomicRmwCmpxchg(t.translate_memarg(memarg)?),
		O::I64AtomicRmwCmpxchg { memarg } => I::I64AtomicRmwCmpxchg(t.translate_memarg(memarg)?),
		O::I32AtomicRmw8CmpxchgU { memarg } =>
			I::I32AtomicRmw8CmpxchgU(t.translate_memarg(memarg)?),
		O::I32AtomicRmw16CmpxchgU { memarg } =>
			I::I32AtomicRmw16CmpxchgU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw8CmpxchgU { memarg } =>
			I::I64AtomicRmw8CmpxchgU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw16CmpxchgU { memarg } =>
			I::I64AtomicRmw16CmpxchgU(t.translate_memarg(memarg)?),
		O::I64AtomicRmw32CmpxchgU { memarg } =>
			I::I64AtomicRmw32CmpxchgU(t.translate_memarg(memarg)?),
		O::AtomicFence => I::AtomicFence,
	})
}

pub fn block_type(t: &mut dyn Translator, ty: &wasmparser::BlockType) -> Result<BlockType> {
	match ty {
		wasmparser::BlockType::Empty => Ok(BlockType::Empty),
		wasmparser::BlockType::Type(ty) => Ok(BlockType::Result(t.translate_ty(ty)?)),
		wasmparser::BlockType::FuncType(f) => Ok(BlockType::FunctionType(*f)),
	}
}

pub fn memarg(_t: &mut dyn Translator, memarg: &wasmparser::MemArg) -> Result<MemArg> {
	Ok(MemArg { offset: memarg.offset, align: memarg.align.into(), memory_index: memarg.memory })
}

#[allow(dead_code)]
pub fn data(t: &mut dyn Translator, data: wasmparser::Data<'_>, s: &mut DataSection) -> Result<()> {
	let offset;
	let mode = match &data.kind {
		DataKind::Active { memory_index, offset_expr } => {
			offset = t.translate_const_expr(
				offset_expr,
				&wasmparser::ValType::I32,
				ConstExprKind::DataOffset,
			)?;
			DataSegmentMode::Active { memory_index: *memory_index, offset: &offset }
		},
		DataKind::Passive => DataSegmentMode::Passive,
	};
	s.segment(DataSegment { mode, data: data.data.iter().copied() });
	Ok(())
}

pub fn code(t: &mut dyn Translator, body: FunctionBody<'_>, s: &mut CodeSection) -> Result<()> {
	let locals = body
		.get_locals_reader()?
		.into_iter()
		.map(|local| {
			let (cnt, ty) = local?;
			Ok((cnt, t.translate_ty(&ty)?))
		})
		.collect::<Result<Vec<_>>>()?;
	let mut func = Function::new(locals);

	let mut reader = body.get_operators_reader()?;
	reader.allow_memarg64(true);
	for op in reader {
		let op = op?;
		func.instruction(&t.translate_op(&op)?);
	}
	s.function(&func);
	Ok(())
}
