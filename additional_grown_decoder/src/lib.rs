use pyo3::prelude::*;

extern crate alloc;

pub mod uf;
pub mod dijkstra;
pub mod bounded_dijkstra;
pub mod scored; // for bounded_dijkstra
mod ufd;
mod ufd2;

use crate::ufd::UFD;
use crate::ufd2::UFD2;

#[pymodule]
fn _rust_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UFD>()?;
    m.add_class::<UFD2>()?;

    Ok(())
}
