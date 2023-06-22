//! Provides backends for the gas metering instrumentation
use crate::parser::ModuleInfo;
use wasm_encoder::Function;
use wasmparser::Type;

/// Implementation details of the specific method of the gas metering.
#[derive(Clone)]
pub enum GasMeter {
	/// Gas metering with an external function.
	External {
		/// Name of the module to import the gas function from.
		module: &'static str,
		/// Name of the external gas function to be imported.
		function: &'static str,
	},
	/// Gas metering with a local function and a mutable global.
	Internal {
		/// The name of the module to import the gas global from.
		/// TODO: make sure if 'module' is really required
		module: &'static str,
		/// Name of the mutable global to be exported.
		global: &'static str,
		/// Body of the local gas counting function to be injected.
		func: Function,
		/// Cost of the gas function execution.
		cost: u64,
	},
}

use super::Rules;
/// Under the hood part of the gas metering mechanics.
pub trait Backend {
	/// Provides the gas metering implementation details.
	fn gas_meter<R: Rules>(self, module_info: &mut ModuleInfo, rules: &R) -> GasMeter;
}

/// Gas metering with an external host function.
pub mod host_function {
	use super::{Backend, GasMeter, Rules};
	use crate::parser::ModuleInfo;

	/// Injects invocations of the gas charging host function into each metering block.
	pub struct Injector {
		/// The name of the module to import the gas function from.
		module: &'static str,
		/// The name of the gas function to import.
		name: &'static str,
	}

	impl Injector {
		pub fn new(module: &'static str, name: &'static str) -> Self {
			Self { module, name }
		}
	}

	impl Backend for Injector {
		fn gas_meter<R: Rules>(self, module_info: &mut ModuleInfo, rules: &R) -> GasMeter {
			GasMeter::External { module: self.module, function: self.name }
		}
	}
}

/// Gas metering with a mutable global.
///
/// # Note
///
/// Not for all execution engines this method gives performance wins compared to using an [external
/// host function](host_function). See benchmarks and size overhead tests for examples of how to
/// make measurements needed to decide which gas metering method is better for your particular case.
///
/// # Warning
///
/// It is not recommended to apply [stack limiter](crate::inject_stack_limiter) instrumentation to a
/// module instrumented with this type of gas metering. This could lead to a massive module size
/// bloat. This is a known issue to be fixed in upcoming versions.
pub mod mutable_global {
	use super::{Backend, GasMeter, Rules};
	use crate::parser::{
		translator::{DefaultTranslator, Translator},
		ModuleInfo,
	};
	use alloc::vec;
	use wasmparser::{BlockType, Operator};

	/// Injects a mutable global variable and a local function to the module to track
	/// current gas left.
	///
	/// The function is called in every metering block. In case of falling out of gas, the global is
	/// set to the sentinel value `U64::MAX` and `unreachable` instruction is called. The execution
	/// engine should take care of getting the current global value and setting it back in order to
	/// sync the gas left value during an execution.
	pub struct Injector {
		/// The name of the module to import the gas global from.
		/// TODO: make sure if 'module' is really required
		module: &'static str,
		/// The export name of the gas tracking global.
		pub global_name: &'static str,
	}

	impl Injector {
		pub fn new(module: &'static str, global_name: &'static str) -> Self {
			Self { module, global_name }
		}
	}

	impl Backend for Injector {
		fn gas_meter<R: Rules>(self, module_info: &mut ModuleInfo, rules: &R) -> GasMeter {
			let gas_global_idx = module_info.num_globals();

			let mut func = wasm_encoder::Function::new(None);
			let operators = vec![
				Operator::GlobalGet { global_index: gas_global_idx },
				Operator::LocalGet { local_index: 0 },
				Operator::I64GeU,
				Operator::If { blockty: BlockType::Empty },
				Operator::GlobalGet { global_index: gas_global_idx },
				Operator::LocalGet { local_index: 0 },
				Operator::I64Sub,
				Operator::GlobalSet { global_index: gas_global_idx },
				Operator::Else,
				// sentinel val u64::MAX
				Operator::I64Const { value: -1i64 }, // non-charged instruction
				Operator::GlobalSet { global_index: gas_global_idx }, // non-charged instruction
				Operator::Unreachable,               // non-charged instruction
				Operator::End,
				Operator::End,
			];
			operators.iter().map(|op| {
				let instr = DefaultTranslator.translate_op(&op).unwrap();
				func.instruction(&instr);
			});

			// calculate gas used for the gas charging func execution itself
			let mut gas_fn_cost = operators.iter().fold(0, |cost: u64, op| {
				cost.saturating_add(rules.instruction_cost(op).unwrap_or(u32::MAX).into())
			});
			// don't charge for the instructions used to fail when out of gas
			let fail_cost = vec![
				Operator::I64Const { value: -1i64 }, // non-charged instruction
				Operator::GlobalSet { global_index: gas_global_idx }, // non-charged instruction
				Operator::Unreachable,               // non-charged instruction
			]
			.iter()
			.fold(0, |cost: u64, op| {
				cost.saturating_add(rules.instruction_cost(op).unwrap_or(u32::MAX).into())
			});

			// the fail costs are a subset of the overall costs and hence this never underflows
			gas_fn_cost -= fail_cost;

			GasMeter::Internal {
				module: self.module,
				global: self.global_name,
				func,
				cost: gas_fn_cost,
			}
		}
	}
}
