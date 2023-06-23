//! This module is used to instrument a Wasm module with the gas metering code.
//!
//! The primary public interface is the [`inject`] function which transforms a given
//! module into one that charges gas for code to be executed. See function documentation for usage
//! and details.

mod backend;

pub use backend::{host_function, mutable_global, Backend, GasMeter};

//#[cfg(test)]
//mod validation;

use crate::parser::{
	copy_locals,
	translator::{ConstExprKind, DefaultTranslator, Translator},
	truncate_len_from_encoder, ModuleInfo,
};
use alloc::{vec, vec::Vec};
use anyhow::{anyhow, Result};
use core::{cmp::min, mem, num::NonZeroU32};
use wasm_encoder::{
	ElementMode, ElementSection, ElementSegment, Elements, ExportKind, ExportSection, Function,
	Instruction, SectionId, StartSection,
};
use wasmparser::{
	CodeSectionReader, ElementItems, ElementKind, ElementSectionReader, ExternalKind, FuncType,
	FunctionBody, GlobalType, Operator, Type, ValType,
};

/// An interface that describes instruction costs.
pub trait Rules {
	/// Returns the cost for the passed `instruction`.
	///
	/// Returning `None` makes the gas instrumention end with an error. This is meant
	/// as a way to have a partial rule set where any instruction that is not specifed
	/// is considered as forbidden.
	fn instruction_cost(&self, instruction: &Operator) -> Option<u32>;

	/// Returns the costs for growing the memory using the `memory.grow` instruction.
	///
	/// Please note that these costs are in addition to the costs specified by `instruction_cost`
	/// for the `memory.grow` instruction. Those are meant as dynamic costs which take the
	/// amount of pages that the memory is grown by into consideration. This is not possible
	/// using `instruction_cost` because those costs depend on the stack and must be injected as
	/// code into the function calling `memory.grow`. Therefore returning anything but
	/// [`MemoryGrowCost::Free`] introduces some overhead to the `memory.grow` instruction.
	fn memory_grow_cost(&self) -> MemoryGrowCost;

	/// A surcharge cost to calling a function that is added per local of that function.
	fn call_per_local_cost(&self) -> u32;
}

/// Dynamic costs for memory growth.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MemoryGrowCost {
	/// Skip per page charge.
	///
	/// # Note
	///
	/// This makes sense when the amount of pages that a module is allowed to use is limited
	/// to a rather small number by static validation. In that case it is viable to
	/// benchmark the costs of `memory.grow` as the worst case (growing to to the maximum
	/// number of pages).
	Free,
	/// Charge the specified amount for each page that the memory is grown by.
	Linear(NonZeroU32),
}

impl MemoryGrowCost {
	/// True iff memory growths code needs to be injected.
	fn enabled(&self) -> bool {
		match self {
			Self::Free => false,
			Self::Linear(_) => true,
		}
	}
}

/// A type that implements [`Rules`] so that every instruction costs the same.
///
/// This is a simplification that is mostly useful for development and testing.
///
/// # Note
///
/// In a production environment it usually makes no sense to assign every instruction
/// the same cost. A proper implemention of [`Rules`] should be provided that is probably
/// created by benchmarking.
pub struct ConstantCostRules {
	instruction_cost: u32,
	memory_grow_cost: u32,
	call_per_local_cost: u32,
}

impl ConstantCostRules {
	/// Create a new [`ConstantCostRules`].
	///
	/// Uses `instruction_cost` for every instruction and `memory_grow_cost` to dynamically
	/// meter the memory growth instruction.
	pub fn new(instruction_cost: u32, memory_grow_cost: u32, call_per_local_cost: u32) -> Self {
		Self { instruction_cost, memory_grow_cost, call_per_local_cost }
	}
}

impl Default for ConstantCostRules {
	/// Uses instruction cost of `1` and disables memory growth instrumentation.
	fn default() -> Self {
		Self { instruction_cost: 1, memory_grow_cost: 0, call_per_local_cost: 1 }
	}
}

impl Rules for ConstantCostRules {
	fn instruction_cost(&self, _: &Operator) -> Option<u32> {
		Some(self.instruction_cost)
	}

	fn memory_grow_cost(&self) -> MemoryGrowCost {
		NonZeroU32::new(self.memory_grow_cost).map_or(MemoryGrowCost::Free, MemoryGrowCost::Linear)
	}

	fn call_per_local_cost(&self) -> u32 {
		self.call_per_local_cost
	}
}

/// Transforms a given module into one that tracks the gas charged during its execution.
///
/// The output module uses the `gas` function to track the gas spent. The function could be either
/// an imported or a local one modifying a mutable global. The argument is the amount of gas
/// required to continue execution. The execution engine is meant to keep track of the total amount
/// of gas used and trap or otherwise halt execution of the runtime if the gas usage exceeds some
/// allowed limit.
///
/// The body of each function of the original module is divided into metered blocks, and the calls
/// to charge gas are inserted at the beginning of every such block of code. A metered block is
/// defined so that, unless there is a trap, either all of the instructions are executed or none
/// are. These are similar to basic blocks in a control flow graph, except that in some cases
/// multiple basic blocks can be merged into a single metered block. This is the case if any path
/// through the control flow graph containing one basic block also contains another.
///
/// Charging gas at the beginning of each metered block ensures that 1) all instructions
/// executed are already paid for, 2) instructions that will not be executed are not charged for
/// unless execution traps, and 3) the number of calls to `gas` is minimized. The corollary is
/// that modules instrumented with this metering code may charge gas for instructions not
/// executed in the event of a trap.
///
/// Additionally, each `memory.grow` instruction found in the module is instrumented to first
/// make a call to charge gas for the additional pages requested. This cannot be done as part of
/// the block level gas charges as the gas cost is not static and depends on the stack argument
/// to `memory.grow`.
///
/// The above transformations are performed for every function body defined in the module. This
/// function also rewrites all function indices references by code, table elements, etc., since
/// the addition of an imported functions changes the indices of module-defined functions. If
/// the module has a `NameSection`, added by calling `parse_names`, the indices will also be
/// updated.
///
/// Syncronizing the amount of gas charged with the execution engine can be done in two ways. The
/// first way is by calling the imported `gas` host function, see [`host_function`] for details. The
/// second way is by using a local `gas` function together with a mutable global, see
/// [`mutable_global`] for details.
///
/// This routine runs in time linear in the size of the input module.
///
/// The function fails if the module contains any operation forbidden by gas rule set, returning
/// the original module as an `Err`.
pub fn inject<R: Rules, B: Backend>(
	module_info: &mut ModuleInfo,
	backend: B,
	rules: &R,
) -> Result<Vec<u8>> {
	// Prepare module and return the gas function
	let gas_meter = backend.gas_meter(module_info, rules);

	let import_count = module_info.imported_functions_count;
	let functions_space = module_info.num_functions();

	// Calculate the indexes and gas function cost,
	// for external gas function the cost is counted on the host side
	let (gas_func_idx, total_func, gas_fn_cost, inject_targets) = match gas_meter {
		GasMeter::External { module: gas_module, function } => {
			// Inject the import of the gas function
			let ty = Type::Func(FuncType::new(vec![ValType::I64], vec![]));
			module_info.add_import_func(gas_module, function, ty)?;

			(import_count, functions_space + 1, 0, module_info.num_local_functions())
		},
		GasMeter::Internal { module: gas_module, global: global_name, ref func, cost } => {
			let gas_global_idx = module_info.num_globals();

			// Inject the gas counting global
			module_info.add_global(
				GlobalType { content_type: ValType::I64, mutable: true },
				&wasm_encoder::ConstExpr::i64_const(0),
			)?;

			module_info.add_export(global_name, ExportKind::Global, gas_global_idx)?;

			// Inject the local gas function
			let ty = Type::Func(FuncType::new(vec![ValType::I64], vec![]));
			module_info.add_func(ty, func)?;

			// Don't inject counters to the local gas function, which is the last one as
			// it's just added. Cost for its execution is added statically before each
			// invocation (see `inject_counter()`).
			let inject_targets = module_info.num_local_functions() - 1;
			let func_idx = functions_space;

			(func_idx, func_idx + 1, cost, inject_targets)
		},
	};

	// TODO: what should be the proper value?
	let grow_cnt_func = total_func;
	let mut need_grow_counter = false;
	let mut error = false;

	// Iterate over module sections and perform needed transformations.
	// Indexes are needed to be fixed up in `GasMeter::External` case, as it adds an imported
	// function, which goes to the beginning of the module's functions space.
	if let Some(code_section) = module_info.raw_sections.get_mut(&SectionId::Code.into()) {
		let mut code_section_builder = wasm_encoder::CodeSection::new();
		let code_sec_reader = CodeSectionReader::new(&code_section.data, 0)?;
		let code_item_count = code_sec_reader.count();

		for (func_body, is_last) in code_sec_reader
			.into_iter()
			.enumerate()
			.map(|(index, item)| (item, index as u32 == code_item_count - 1))
		{
			let func_body = func_body?;
			let current_locals = copy_locals(&func_body)?;

			let locals_count = current_locals.iter().map(|(count, _)| count).sum();

			let mut func_builder = wasm_encoder::Function::new(copy_locals(&func_body)?);

			let operator_reader = func_body.get_operators_reader()?;
			for op in operator_reader {
				let op = op?;
				if let GasMeter::External { .. } = gas_meter {
					if let Operator::Call { function_index } = op {
						if function_index >= gas_func_idx {
							func_builder.instruction(&Instruction::Call(function_index + 1));
						}
					} else {
						func_builder.instruction(&DefaultTranslator.translate_op(&op)?);
					}
				} else {
					func_builder.instruction(&DefaultTranslator.translate_op(&op)?);
				}
			}

			if let GasMeter::Internal { .. } = gas_meter {
				// If GasMeter::Internal then don't inject counters to the local gas function, which is the last one as
				// it's just added. Cost for its execution is added statically before each
				// invocation (see `inject_counter()`).
				if is_last {
					code_section_builder.function(&func_builder);
					continue;
				}
			}

			match inject_counter(
				&FunctionBody::new(0, &truncate_len_from_encoder(&func_builder)?),
				gas_fn_cost,
				locals_count,
				rules,
				gas_func_idx,
			) {
				Ok(new_builder) => func_builder = new_builder,
				Err(_) => {
					error = true;
					break;
				},
			}
			if rules.memory_grow_cost().enabled() {
				let counter;
				(func_builder, counter) = inject_grow_counter(
					&FunctionBody::new(0, &truncate_len_from_encoder(&func_builder)?),
					grow_cnt_func,
				)?;
				if counter > 0 {
					need_grow_counter = true;
				}
			}
			code_section_builder.function(&func_builder);
		}
		module_info.replace_section(SectionId::Code.into(), &code_section_builder)?;
	}

	if let Some(_) = module_info.raw_sections.get_mut(&SectionId::Export.into()) {
		if let GasMeter::External { .. } = gas_meter {
			let mut export_sec_builder = ExportSection::new();

			let exports = module_info.export_section()?;
			for export in exports {
				let mut export_index = export.index;
				if let ExternalKind::Func = export.kind {
					if export_index >= gas_func_idx {
						export_index += 1;
					}
				}
				export_sec_builder.export(
					export.name,
					DefaultTranslator.translate_export_kind(export.kind)?,
					export_index,
				);
			}
			module_info
				.replace_section(SectionId::Export.into(), &export_sec_builder)
				.unwrap();
		}
	}

	if let Some(ele_section) = module_info.raw_sections.get_mut(&SectionId::Element.into()) {
		// Note that we do not need to check the element type referenced because in the
		// WebAssembly 1.0 spec, the only allowed element type is funcref.
		if let GasMeter::External { .. } = gas_meter {
			let mut ele_sec_builder = ElementSection::new();
			let ele_sec_reader = ElementSectionReader::new(&ele_section.data, 0)?;

			for elem in ele_sec_reader {
				let elem = elem?;
				let mut functions = vec![];
				if let ElementItems::Functions(func_indexes) = elem.items {
					for func_idx in func_indexes {
						let mut func_idx = func_idx?;
						if func_idx >= gas_func_idx {
							func_idx += 1
						}
						functions.push(func_idx);
					}
				}

				let offset;
				//todo edit element is little complex,
				let mode = match elem.kind {
					ElementKind::Active { table_index, offset_expr } => {
						offset = DefaultTranslator.translate_const_expr(
							&offset_expr,
							&ValType::I32,
							ConstExprKind::ElementOffset,
						)?;

						ElementMode::Active { table: table_index, offset: &offset }
					},
					ElementKind::Passive => ElementMode::Passive,
					ElementKind::Declared => ElementMode::Declared,
				};

				let element_type = DefaultTranslator.translate_refty(&elem.ty)?;
				let elements = Elements::Functions(&functions);

				ele_sec_builder.segment(ElementSegment {
					mode,
					/// The element type.
					element_type,
					/// The element functions.
					elements,
				});
			}
			module_info.replace_section(SectionId::Element.into(), &ele_sec_builder)?;
		}
	}

	if let Some(_) = module_info.raw_sections.get_mut(&SectionId::Start.into()) {
		if let GasMeter::External { .. } = gas_meter {
			if let Some(func_idx) = module_info.start_function {
				if func_idx > gas_func_idx {
					let start_section = StartSection { function_index: func_idx + 1 };
					module_info.replace_section(SectionId::Start.into(), &start_section)?;
				}
			}
		}
	}

	if error {
		return Err(anyhow!("inject fail"));
	}

	if need_grow_counter {
		if let Some((func, grow_counter_func)) = generate_grow_counter(rules, gas_func_idx) {
			module_info.add_func(func, &grow_counter_func)?;
		}
	}
	Ok(module_info.bytes())
}

/// A control flow block is opened with the `block`, `loop`, and `if` instructions and is closed
/// with `end`. Each block implicitly defines a new label. The control blocks form a stack during
/// program execution.
///
/// An example of block:
///
/// ```ignore
/// loop
///   i32.const 1
///   get_local 0
///   i32.sub
///   tee_local 0
///   br_if 0
/// end
/// ```
///
/// The start of the block is `i32.const 1`.
#[derive(Debug)]
struct ControlBlock {
	/// The lowest control stack index corresponding to a forward jump targeted by a br, br_if, or
	/// br_table instruction within this control block. The index must refer to a control block
	/// that is not a loop, meaning it is a forward jump. Given the way Wasm control flow is
	/// structured, the lowest index on the stack represents the furthest forward branch target.
	///
	/// This value will always be at most the index of the block itself, even if there is no
	/// explicit br instruction targeting this control block. This does not affect how the value is
	/// used in the metering algorithm.
	lowest_forward_br_target: usize,

	/// The active metering block that new instructions contribute a gas cost towards.
	active_metered_block: MeteredBlock,

	/// Whether the control block is a loop. Loops have the distinguishing feature that branches to
	/// them jump to the beginning of the block, not the end as with the other control blocks.
	is_loop: bool,
}

/// A block of code that metering instructions will be inserted at the beginning of. Metered blocks
/// are constructed with the property that, in the absence of any traps, either all instructions in
/// the block are executed or none are.
#[derive(Debug)]
struct MeteredBlock {
	/// Index of the first instruction (aka `Opcode`) in the block.
	start_pos: usize,
	/// Sum of costs of all instructions until end of the block.
	cost: u64,
}

/// Counter is used to manage state during the gas metering algorithm implemented by
/// `inject_counter`.
struct Counter {
	/// A stack of control blocks. This stack grows when new control blocks are opened with
	/// `block`, `loop`, and `if` and shrinks when control blocks are closed with `end`. The first
	/// block on the stack corresponds to the function body, not to any labelled block. Therefore
	/// the actual Wasm label index associated with each control block is 1 less than its position
	/// in this stack.
	stack: Vec<ControlBlock>,

	/// A list of metered blocks that have been finalized, meaning they will no longer change.
	finalized_blocks: Vec<MeteredBlock>,
}

impl Counter {
	fn new() -> Counter {
		Counter { stack: Vec::new(), finalized_blocks: Vec::new() }
	}

	/// Open a new control block. The cursor is the position of the first instruction in the block.
	fn begin_control_block(&mut self, cursor: usize, is_loop: bool) {
		let index = self.stack.len();
		self.stack.push(ControlBlock {
			lowest_forward_br_target: index,
			active_metered_block: MeteredBlock { start_pos: cursor, cost: 0 },
			is_loop,
		})
	}

	/// Close the last control block. The cursor is the position of the final (pseudo-)instruction
	/// in the block.
	fn finalize_control_block(&mut self, cursor: usize) -> Result<()> {
		// This either finalizes the active metered block or merges its cost into the active
		// metered block in the previous control block on the stack.
		self.finalize_metered_block(cursor)?;

		// Pop the control block stack.
		let closing_control_block = self.stack.pop().ok_or_else(|| anyhow!("stack not found"))?;
		let closing_control_index = self.stack.len();

		if self.stack.is_empty() {
			return Ok(());
		}

		// Update the lowest_forward_br_target for the control block now on top of the stack.
		{
			let control_block = self.stack.last_mut().ok_or_else(|| anyhow!("stack not found"))?;
			control_block.lowest_forward_br_target = min(
				control_block.lowest_forward_br_target,
				closing_control_block.lowest_forward_br_target,
			);
		}

		// If there may have been a branch to a lower index, then also finalize the active metered
		// block for the previous control block. Otherwise, finalize it and begin a new one.
		let may_br_out = closing_control_block.lowest_forward_br_target < closing_control_index;
		if may_br_out {
			self.finalize_metered_block(cursor)?;
		}

		Ok(())
	}

	/// Finalize the current active metered block.
	///
	/// Finalized blocks have final cost which will not change later.
	fn finalize_metered_block(&mut self, cursor: usize) -> Result<()> {
		let closing_metered_block = {
			let control_block = self.stack.last_mut().ok_or_else(|| anyhow!("stack not found"))?;
			mem::replace(
				&mut control_block.active_metered_block,
				MeteredBlock { start_pos: cursor + 1, cost: 0 },
			)
		};

		// If the block was opened with a `block`, then its start position will be set to that of
		// the active metered block in the control block one higher on the stack. This is because
		// any instructions between a `block` and the first branch are part of the same basic block
		// as the preceding instruction. In this case, instead of finalizing the block, merge its
		// cost into the other active metered block to avoid injecting unnecessary instructions.
		let last_index = self.stack.len() - 1;
		if last_index > 0 {
			let prev_control_block = self
				.stack
				.get_mut(last_index - 1)
				.expect("last_index is greater than 0; last_index is stack size - 1; qed");
			let prev_metered_block = &mut prev_control_block.active_metered_block;
			if closing_metered_block.start_pos == prev_metered_block.start_pos {
				prev_metered_block.cost = prev_metered_block
					.cost
					.checked_add(closing_metered_block.cost)
					.ok_or_else(|| anyhow!("overflow occured"))?;
				return Ok(());
			}
		}

		if closing_metered_block.cost > 0 {
			self.finalized_blocks.push(closing_metered_block);
		}
		Ok(())
	}

	/// Handle a branch instruction in the program. The cursor is the index of the branch
	/// instruction in the program. The indices are the stack positions of the target control
	/// blocks. Recall that the index is 0 for a `return` and relatively indexed from the top of
	/// the stack by the label of `br`, `br_if`, and `br_table` instructions.
	fn branch(&mut self, cursor: usize, indices: &[usize]) -> Result<()> {
		self.finalize_metered_block(cursor)?;

		// Update the lowest_forward_br_target of the current control block.
		for &index in indices {
			let target_is_loop = {
				let target_block =
					self.stack.get(index).ok_or_else(|| anyhow!("unable to find stack index"))?;
				target_block.is_loop
			};
			if target_is_loop {
				continue;
			}

			let control_block =
				self.stack.last_mut().ok_or_else(|| anyhow!("stack does not exist"))?;
			control_block.lowest_forward_br_target =
				min(control_block.lowest_forward_br_target, index);
		}

		Ok(())
	}

	/// Returns the stack index of the active control block. Returns None if stack is empty.
	fn active_control_block_index(&self) -> Option<usize> {
		self.stack.len().checked_sub(1)
	}

	/// Get a reference to the currently active metered block.
	fn active_metered_block(&mut self) -> Result<&mut MeteredBlock> {
		let top_block = self.stack.last_mut().ok_or_else(|| anyhow!("stack does not exist"))?;
		Ok(&mut top_block.active_metered_block)
	}

	/// Increment the cost of the current block by the specified value.
	fn increment(&mut self, val: u32) -> Result<()> {
		let top_block = self.active_metered_block()?;
		top_block.cost = top_block
			.cost
			.checked_add(val.into())
			.ok_or_else(|| anyhow!("add cost overflow"))?;
		Ok(())
	}
}

fn inject_grow_counter(
	func_body: &FunctionBody,
	grow_counter_func: u32,
) -> Result<(Function, usize)> {
	let mut counter = 0;
	let mut new_func = Function::new(copy_locals(func_body)?);
	let mut operator_reader = func_body.get_operators_reader()?;
	while !operator_reader.eof() {
		let op = operator_reader.read()?;
		match op {
			Operator::MemoryGrow { .. } => {
				//todo Bulk memories
				new_func.instruction(&wasm_encoder::Instruction::Call(grow_counter_func));
				counter += 1;
			},
			op => {
				new_func.instruction(&DefaultTranslator.translate_op(&op)?);
			},
		}
	}
	Ok((new_func, counter))
}

fn generate_grow_counter<R: Rules>(rules: &R, gas_func: u32) -> Option<(Type, Function)> {
	let cost = match rules.memory_grow_cost() {
		MemoryGrowCost::Free => return None,
		MemoryGrowCost::Linear(val) => val.get(),
	};

	let mut func = wasm_encoder::Function::new(None);
	func.instruction(&wasm_encoder::Instruction::LocalGet(0));
	func.instruction(&wasm_encoder::Instruction::LocalGet(0));
	func.instruction(&wasm_encoder::Instruction::I64ExtendI32U);
	func.instruction(&wasm_encoder::Instruction::I64Const(cost as i64));
	func.instruction(&wasm_encoder::Instruction::I64Mul);
	func.instruction(&wasm_encoder::Instruction::Call(gas_func));
	func.instruction(&wasm_encoder::Instruction::MemoryGrow(0));
	func.instruction(&wasm_encoder::Instruction::End);
	Some((Type::Func(FuncType::new(vec![ValType::I32], vec![ValType::I32])), func))
}

fn determine_metered_blocks<R: Rules>(
	func_body: &wasmparser::FunctionBody,
	rules: &R,
	locals_count: u32,
) -> Result<Vec<MeteredBlock>> {
	use wasmparser::Operator::*;

	let mut counter = Counter::new();

	// Begin an implicit function (i.e. `func...end`) block.
	counter.begin_control_block(0, false);
	// Add locals initialization cost to the function block.
	let locals_init_cost = rules
		.call_per_local_cost()
		.checked_mul(locals_count)
		.ok_or_else(|| anyhow!("overflow occured"))?;
	counter.increment(locals_init_cost)?;

	let operators = func_body
		.get_operators_reader()
		.unwrap()
		.into_iter()
		.collect::<wasmparser::Result<Vec<Operator>>>()
		.unwrap();
	for (cursor, instruction) in operators.iter().enumerate() {
		let instruction_cost = rules
			.instruction_cost(instruction)
			.ok_or_else(|| anyhow!("check gas rule fail"))?;
		match instruction {
			Block { blockty: _ } => {
				counter.increment(instruction_cost)?;

				// Begin new block. The cost of the following opcodes until `end` or `else` will
				// be included into this block. The start position is set to that of the previous
				// active metered block to signal that they should be merged in order to reduce
				// unnecessary metering instructions.
				let top_block_start_pos = counter.active_metered_block()?.start_pos;
				counter.begin_control_block(top_block_start_pos, false);
			},
			If { blockty: _ } => {
				counter.increment(instruction_cost)?;
				counter.begin_control_block(cursor + 1, false);
			},
			Loop { blockty: _ } => {
				counter.increment(instruction_cost)?;
				counter.begin_control_block(cursor + 1, true);
			},
			End => {
				counter.finalize_control_block(cursor)?;
			},
			Else => {
				counter.finalize_metered_block(cursor)?;
			},
			Br { relative_depth } | BrIf { relative_depth } => {
				counter.increment(instruction_cost)?;

				// Label is a relative index into the control stack.
				let active_index = counter
					.active_control_block_index()
					.ok_or_else(|| anyhow!("active control block not exit"))?;

				let target_index = active_index
					.checked_sub(*relative_depth as usize)
					.ok_or_else(|| anyhow!("index not found"))?;

				counter.branch(cursor, &[target_index])?;
			},
			BrTable { targets: br_table_data } => {
				counter.increment(instruction_cost)?;

				let active_index = counter
					.active_control_block_index()
					.ok_or_else(|| anyhow!("index not found"))?;
				let r = br_table_data.targets().collect::<wasmparser::Result<Vec<u32>>>().unwrap();
				let target_indices = [br_table_data.default()]
					.iter()
					.chain(r.iter())
					.map(|label| active_index.checked_sub(*label as usize))
					.collect::<Option<Vec<_>>>()
					.ok_or_else(|| anyhow!("to do check this error"))?;
				counter.branch(cursor, &target_indices)?;
			},
			Return => {
				counter.increment(instruction_cost)?;
				counter.branch(cursor, &[0])?;
			},
			_ => {
				// An ordinal non control flow instruction increments the cost of the current block.
				counter.increment(instruction_cost)?;
			},
		}
	}

	counter.finalized_blocks.sort_unstable_by_key(|block| block.start_pos);
	Ok(counter.finalized_blocks)
}

fn inject_counter<R: Rules>(
	func_body: &FunctionBody,
	gas_function_cost: u64,
	locals_count: u32,
	rules: &R,
	gas_func: u32,
) -> Result<wasm_encoder::Function> {
	let blocks = determine_metered_blocks(func_body, rules, locals_count)?;
	insert_metering_calls(func_body, gas_function_cost, blocks, gas_func)
}

// Then insert metering calls into a sequence of instructions given the block locations and costs.
fn insert_metering_calls(
	func_body: &FunctionBody,
	gas_function_cost: u64,
	blocks: Vec<MeteredBlock>,
	gas_func: u32,
) -> Result<wasm_encoder::Function> {
	let mut new_func = wasm_encoder::Function::new(copy_locals(func_body)?);

	// To do this in linear time, construct a new vector of instructions, copying over old
	// instructions one by one and injecting new ones as required.
	let mut block_iter = blocks.into_iter().peekable();
	let operators = func_body
		.get_operators_reader()
		.unwrap()
		.into_iter()
		.collect::<wasmparser::Result<Vec<Operator>>>()?;
	for (original_pos, instr) in operators.iter().enumerate() {
		// If there the next block starts at this position, inject metering func_body.
		let used_block = if let Some(block) = block_iter.peek() {
			if block.start_pos == original_pos {
				let cost = block
					.cost
					.checked_add(gas_function_cost)
					.ok_or_else(|| anyhow!("block cost add overflow"))? as i64;
				new_func.instruction(&wasm_encoder::Instruction::I64Const(cost));
				new_func.instruction(&wasm_encoder::Instruction::Call(gas_func));
				true
			} else {
				false
			}
		} else {
			false
		};

		if used_block {
			block_iter.next();
		}

		// Copy over the original instruction.
		new_func.instruction(&DefaultTranslator.translate_op(instr)?);
	}

	if block_iter.next().is_some() {
		return Err(anyhow!("block should be consume all"));
	}
	Ok(new_func)
}

#[cfg(test)]
mod tests {
	use super::*;
	use wasm_encoder::{BlockType, Encode, Instruction::*};
	use wasmparser::FunctionBody;

	fn check_expect_function_body(
		raw_wasm: &[u8],
		index: usize,
		ops2: &[wasm_encoder::Instruction],
	) -> bool {
		let mut body_raw = vec![];
		ops2.iter().for_each(|v| v.encode(&mut body_raw));
		get_function_body(raw_wasm, index).eq(&body_raw)
	}

	fn get_function_body(raw_wasm: &[u8], index: usize) -> Vec<u8> {
		let mut module = ModuleInfo::new(raw_wasm).unwrap();
		let func_sec = module.raw_sections.get_mut(&SectionId::Code.into()).unwrap();
		let func_bodies = wasmparser::CodeSectionReader::new(&func_sec.data, 0)
			.unwrap()
			.into_iter()
			.collect::<wasmparser::Result<Vec<FunctionBody>>>()
			.unwrap();

		let func_body = func_bodies
			.get(index)
			.unwrap_or_else(|| panic!("module doesn't have function {} body", index));

		let list = func_body
			.get_operators_reader()
			.unwrap()
			.into_iter()
			.map(|op| DefaultTranslator.translate_op(&op.unwrap()).unwrap())
			.collect::<Vec<Instruction>>();

		let start = func_body.get_operators_reader().unwrap().original_position();
		func_sec.data[start..func_body.range().end].to_vec()
	}

	fn get_function_operators(raw_wasm: &[u8], index: usize) -> Vec<Instruction> {
		let mut module = ModuleInfo::new(raw_wasm).unwrap();
		let func_sec = module.raw_sections.get_mut(&SectionId::Code.into()).unwrap();
		let func_bodies = wasmparser::CodeSectionReader::new(&func_sec.data, 0)
			.unwrap()
			.into_iter()
			.collect::<wasmparser::Result<Vec<FunctionBody>>>()
			.unwrap();

		let func_body = func_bodies
			.get(index)
			.unwrap_or_else(|| panic!("module doesn't have function {} body", index));

		let operators = func_body
			.get_operators_reader()
			.unwrap()
			.into_iter()
			.map(|op| DefaultTranslator.translate_op(&op.unwrap()).unwrap())
			.collect::<Vec<Instruction>>();

		operators
	}

	fn parse_wat(source: &str) -> ModuleInfo {
		let module_bytes = wat::parse_str(source).unwrap();
		ModuleInfo::new(&module_bytes).unwrap()
	}

	#[test]
	fn simple_grow_host_fn() {
		let mut module = parse_wat(
			r#"(module
			(func (result i32)
			  global.get 0
			  memory.grow)
			(global i32 (i32.const 42))
			(memory 0 1)
			)"#,
		);

		let backend = host_function::Injector::new("env", "gas");
		let injected_raw_wasm =
			super::inject(&mut module, backend, &ConstantCostRules::new(1, 10_000, 1)).unwrap();

		// main function
		assert!(check_expect_function_body(
			&injected_raw_wasm,
			0,
			&[I64Const(2), Call(0), GlobalGet(0), Call(2), End,]
		));
		// grow counter
		assert!(check_expect_function_body(
			&injected_raw_wasm,
			1,
			&[
				LocalGet(0),
				LocalGet(0),
				I64ExtendI32U,
				I64Const(10000),
				I64Mul,
				Call(0),
				MemoryGrow(0),
				End,
			]
		));

		wasmparser::validate(&injected_raw_wasm).unwrap();
	}

	#[test]
	fn simple_grow_mut_global() {
		let mut module = parse_wat(
			r#"(module
				(func (result i32)
				  global.get 0
				  memory.grow)
				(global i32 (i32.const 42))
				(memory 0 1)
				)"#,
		);

		let backend = mutable_global::Injector::new("env", "gas_left");
		let injected_raw_wasm =
			super::inject(&mut module, backend, &ConstantCostRules::new(1, 10_000, 1)).unwrap();

		// gas_counter
		assert!(check_expect_function_body(
			&injected_raw_wasm,
			1,
			&[
				GlobalGet(1),
				LocalGet(0),
				I64GeU,
				If(BlockType::Empty),
				GlobalGet(1),
				LocalGet(0),
				I64Sub,
				GlobalSet(1),
				Else,
				I64Const(-1i64),
				GlobalSet(1),
				Unreachable,
				End,
				End
			]
		));

		// grow_counter
		assert!(check_expect_function_body(
			&injected_raw_wasm,
			2,
			&[
				LocalGet(0),
				LocalGet(0),
				I64ExtendI32U,
				I64Const(10000i64),
				I64Mul,
				Call(1),
				MemoryGrow(0),
				End
			]
		));

		wasmparser::validate(&injected_raw_wasm).unwrap();
	}

	#[test]
	fn grow_no_gas_no_track_host_fn() {
		let mut module = parse_wat(
			r"(module
				(func (result i32)
				  global.get 0
				  memory.grow)
				(global i32 (i32.const 42))
				(memory 0 1)
				)",
		);

		let backend = host_function::Injector::new("env", "gas");
		let injected_raw_wasm =
			super::inject(&mut module, backend, &ConstantCostRules::default()).unwrap();

		// main function
		assert!(check_expect_function_body(
			&injected_raw_wasm,
			0,
			&[I64Const(2), Call(0), GlobalGet(0), MemoryGrow(0), End,]
		));

		// Sum of local ('main') and imported functions ('gas') shall be 2
		assert_eq!(module.num_functions(), 2);

		wasmparser::validate(&injected_raw_wasm).unwrap();
	}
	#[test]
	fn grow_no_gas_no_track_mut_global() {
		let mut module = parse_wat(
			r"(module
					(func (result i32)
					  global.get 0
					  memory.grow)
					(global i32 (i32.const 42))
					(memory 0 1)
					)",
		);

		let backend = host_function::Injector::new("env", "gas");
		let injected_raw_wasm =
			super::inject(&mut module, backend, &ConstantCostRules::default()).unwrap();

		// main function
		assert!(check_expect_function_body(
			&injected_raw_wasm,
			0,
			&[I64Const(2), Call(0), GlobalGet(0), MemoryGrow(0), End,]
		));

		// Sum of local ('main') and imported functions ('gas') shall be 2
		assert_eq!(module.num_functions(), 2);

		wasmparser::validate(&injected_raw_wasm).unwrap();
	}
	#[test]
	fn call_index_host_fn() {
		let mut module = parse_wat(
			r"(module
				  (type (;0;) (func (result i32)))
				  (func (;0;) (type 0) (result i32))
				  (func (;1;) (type 0) (result i32)
					call 0
					if  ;; label = @1
					  call 0
					  call 0
					  call 0
					else
					  call 0
					  call 0
					end
					call 0
				  )
				  (global (;0;) i32 (i32.const 0))
				)",
		);

		let backend = host_function::Injector::new("env", "gas");
		let injected_raw_wasm =
			super::inject(&mut module, backend, &ConstantCostRules::default()).unwrap();

		// main function
		assert!(check_expect_function_body(
			&injected_raw_wasm,
			1,
			&vec![
				I64Const(3),
				Call(0),
				Call(1),
				If(BlockType::Empty),
				I64Const(3),
				Call(0),
				Call(1),
				Call(1),
				Call(1),
				Else,
				I64Const(2),
				Call(0),
				Call(1),
				Call(1),
				End,
				Call(1),
				End
			]
		));
	}
	#[test]
	fn call_index_mut_global() {
		let mut module = parse_wat(
			r"(module
				  (type (;0;) (func (result i32)))
				  (func (;0;) (type 0) (result i32))
				  (func (;1;) (type 0) (result i32)
					call 0
					if  ;; label = @1
					  call 0
					  call 0
					  call 0
					else
					  call 0
					  call 0
					end
					call 0
				  )
				  (global (;0;) i32 (i32.const 0))
				)",
		);

		let backend = mutable_global::Injector::new("env", "gas_left");
		let injected_raw_wasm =
			super::inject(&mut module, backend, &ConstantCostRules::default()).unwrap();

		// main function
		assert!(check_expect_function_body(
			&injected_raw_wasm,
			1,
			&vec![
				I64Const(14),
				Call(2),
				Call(0),
				If(BlockType::Empty),
				I64Const(14),
				Call(2),
				Call(0),
				Call(0),
				Call(0),
				Else,
				I64Const(13),
				Call(2),
				Call(0),
				Call(0),
				End,
				Call(0),
				End
			]
		));
	}

	macro_rules! test_gas_counter_injection {
		(names = ($name1:ident, $name2:ident); input = $input:expr; expected = $expected:expr) => {
			#[test]
			fn $name1() {
				let mut module = parse_wat($input);
				let expected_module = parse_wat($expected);
				let injected_wasm = super::inject(
					&mut module,
					host_function::Injector::new("env", "gas"),
					&ConstantCostRules::default(),
				)
				.expect("inject_gas_counter call failed");

				let actual_func_body = get_function_body(&injected_wasm, 0);
				let expected_func_body = get_function_body(&expected_module.bytes(), 0);

				assert_eq!(actual_func_body, expected_func_body);
			}

			#[test]
			fn $name2() {
				let mut module = parse_wat($input);
				let draft_module = parse_wat($expected);
				let gas_fun_cost = match mutable_global::Injector::new("env", "gas_left")
					.gas_meter(&mut module, &ConstantCostRules::default())
				{
					GasMeter::Internal { cost, .. } => cost as i64,
					_ => 0i64,
				};

				let injected_wasm = super::inject(
					&mut module,
					mutable_global::Injector::new("env", "gas_left"),
					&ConstantCostRules::default(),
				)
				.expect("inject_gas_counter call failed");

				let actual_func_body = get_function_body(&injected_wasm, 0);

				let expected_module_bytes = draft_module.bytes();
				let mut expected_func_operators = get_function_operators(&expected_module_bytes, 0);

				// modify expected instructions set for gas_metering::mutable_global
				let mut iter = expected_func_operators.iter_mut();
				while let Some(ins) = iter.next() {
					if let I64Const(cost) = ins {
						if let Some(ins_next) = iter.next() {
							if let Call(0) = ins_next {
								*cost += gas_fun_cost;
								*ins_next = Call(1);
							}
						}
					}
				}
				let mut expected_func_body = vec![];
				expected_func_operators.iter().for_each(|v| v.encode(&mut expected_func_body));

				assert_eq!(actual_func_body, expected_func_body);
			}
		};
	}

	test_gas_counter_injection! {
		names = (simple_host_fn, simple_mut_global);
		input = r#"
					(module
						(func (result i32)
							(get_global 0)))
					"#;
		expected = r#"
					(module
						(func (result i32)
							(call 0 (i64.const 1))
							(get_global 0)))
					"#
	}

	test_gas_counter_injection! {
		names = (nested_host_fn, nested_mut_global);
		input = r#"
					(module
						(func (result i32)
							(get_global 0)
							(block
								(get_global 0)
								(get_global 0)
								(get_global 0))
							(get_global 0)))
					"#;
		expected = r#"
					(module
						(func (result i32)
							(call 0 (i64.const 6))
							(get_global 0)
							(block
								(get_global 0)
								(get_global 0)
								(get_global 0))
							(get_global 0)))
					"#
	}

	test_gas_counter_injection! {
		names = (ifelse_host_fn, ifelse_mut_global);
		input = r#"
					(module
						(func (result i32)
							(get_global 0)
							(if
								(then
									(get_global 0)
									(get_global 0)
									(get_global 0))
								(else
									(get_global 0)
									(get_global 0)))
							(get_global 0)))
					"#;
		expected = r#"
					(module
						(func (result i32)
							(call 0 (i64.const 3))
							(get_global 0)
							(if
								(then
									(call 0 (i64.const 3))
									(get_global 0)
									(get_global 0)
									(get_global 0))
								(else
									(call 0 (i64.const 2))
									(get_global 0)
									(get_global 0)))
							(get_global 0)))
					"#
	}

	test_gas_counter_injection! {
		names = (branch_innermost_host_fn, branch_innermost_mut_global);
		input = r#"
					(module
						(func (result i32)
							(get_global 0)
							(block
								(get_global 0)
								(drop)
								(br 0)
								(get_global 0)
								(drop))
							(get_global 0)))
					"#;
		expected = r#"
					(module
						(func (result i32)
							(call 0 (i64.const 6))
							(get_global 0)
							(block
								(get_global 0)
								(drop)
								(br 0)
								(call 0 (i64.const 2))
								(get_global 0)
								(drop))
							(get_global 0)))
					"#
	}

	test_gas_counter_injection! {
		names = (branch_outer_block_host_fn, branch_outer_block_mut_global);
		input = r#"
					(module
						(func (result i32)
							(get_global 0)
							(block
								(get_global 0)
								(if
									(then
										(get_global 0)
										(get_global 0)
										(drop)
										(br_if 1)))
								(get_global 0)
								(drop))
							(get_global 0)))
					"#;
		expected = r#"
					(module
						(func (result i32)
							(call 0 (i64.const 5))
							(get_global 0)
							(block
								(get_global 0)
								(if
									(then
										(call 0 (i64.const 4))
										(get_global 0)
										(get_global 0)
										(drop)
										(br_if 1)))
								(call 0 (i64.const 2))
								(get_global 0)
								(drop))
							(get_global 0)))
					"#
	}

	test_gas_counter_injection! {
		names = (branch_outer_loop_host_fn, branch_outer_loop_mut_global);
		input = r#"
					(module
						(func (result i32)
							(get_global 0)
							(loop
								(get_global 0)
								(if
									(then
										(get_global 0)
										(br_if 0))
									(else
										(get_global 0)
										(get_global 0)
										(drop)
										(br_if 1)))
								(get_global 0)
								(drop))
							(get_global 0)))
					"#;
		expected = r#"
					(module
						(func (result i32)
							(call 0 (i64.const 3))
							(get_global 0)
							(loop
								(call 0 (i64.const 4))
								(get_global 0)
								(if
									(then
										(call 0 (i64.const 2))
										(get_global 0)
										(br_if 0))
									(else
										(call 0 (i64.const 4))
										(get_global 0)
										(get_global 0)
										(drop)
										(br_if 1)))
								(get_global 0)
								(drop))
							(get_global 0)))
					"#
	}

	test_gas_counter_injection! {
		names = (return_from_func_host_fn, return_from_func_mut_global);
		input = r#"
					(module
						(func (result i32)
							(get_global 0)
							(if
								(then
									(return)))
							(get_global 0)))
					"#;
		expected = r#"
					(module
						(func (result i32)
							(call 0 (i64.const 2))
							(get_global 0)
							(if
								(then
									(call 0 (i64.const 1))
									(return)))
							(call 0 (i64.const 1))
							(get_global 0)))
					"#
	}

	test_gas_counter_injection! {
		names = (branch_from_if_not_else_host_fn, branch_from_if_not_else_mut_global);
		input = r#"
				(module
					(func (result i32)
						(get_global 0)
						(block
							(get_global 0)
							(if
								(then (br 1))
								(else (br 0)))
							(get_global 0)
							(drop))
						(get_global 0)))
				"#;
		expected = r#"
				(module
					(func (result i32)
						(call 0 (i64.const 5))
						(get_global 0)
						(block
							(get_global 0)
							(if
								(then
									(call 0 (i64.const 1))
									(br 1))
								(else
									(call 0 (i64.const 1))
									(br 0)))
							(call 0 (i64.const 2))
							(get_global 0)
							(drop))
						(get_global 0)))
				"#
	}

	test_gas_counter_injection! {
		names = (empty_loop_host_fn, empty_loop_mut_global);
		input = r#"
			(module
				(func
					(loop
						(br 0)
					)
					unreachable
				)
			)
			"#;
		expected = r#"
			(module
				(func
					(call 0 (i64.const 2))
					(loop
						(call 0 (i64.const 1))
						(br 0)
					)
					unreachable
				)
			)
			"#
	}
}
