use alloc::string::String;
use anyhow::{anyhow, Error};
use wasmparser::BinaryReaderError;

// ModuleInfo errors
#[derive(Clone, Debug)]
pub enum ModuleInfoError {
	WasmParserError(BinaryReaderError),
	TranslatorError(TranslatorError),
	TypeDoesNotExist(u32),
	FunctionDoesNotExist(u32),
	NoMemorySection,
}

impl core::fmt::Display for ModuleInfoError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			ModuleInfoError::WasmParserError(err) => {
				write!(f, "WasmParserError(BinaryReaderError {{ {} }})", err)
			},
			_ => write!(f, "{:?}", self),
		}
	}
}

impl From<BinaryReaderError> for ModuleInfoError {
	fn from(err: BinaryReaderError) -> ModuleInfoError {
		ModuleInfoError::WasmParserError(err)
	}
}

impl From<TranslatorError> for ModuleInfoError {
	fn from(err: TranslatorError) -> ModuleInfoError {
		ModuleInfoError::TranslatorError(err)
	}
}

// Translator errors
#[derive(Clone, Debug)]
pub enum TranslatorError {
	WasmParserError(BinaryReaderError),
	NoMutationsApplicable,
	Error(String),
}

impl From<BinaryReaderError> for TranslatorError {
	fn from(err: BinaryReaderError) -> TranslatorError {
		TranslatorError::WasmParserError(err)
	}
}

// Auto-conversions for ModuleInfoError for anyhow::Error
impl From<ModuleInfoError> for Error {
	fn from(err: ModuleInfoError) -> Error {
		anyhow!("{:?}", err)
	}
}

// Auto-conversions for TranslatorError for anyhow::Error
impl From<TranslatorError> for Error {
	fn from(err: TranslatorError) -> Error {
		anyhow!("{:?}", err)
	}
}
