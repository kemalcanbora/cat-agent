use std::path::Path;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

#[pyclass(module = "cat_agent._native", skip_from_py_object)]
pub struct VectorIndex {
    dimensions: usize,
    metric: String,
    index: Index,
    keys: Vec<u64>,
}

impl VectorIndex {
    fn metric_from_name(metric: &str) -> PyResult<MetricKind> {
        match metric.to_ascii_lowercase().as_str() {
            "cos" | "cosine" => Ok(MetricKind::Cos),
            "l2" | "l2sq" => Ok(MetricKind::L2sq),
            "ip" | "dot" => Ok(MetricKind::IP),
            _ => Err(PyValueError::new_err("metric must be one of: cos, l2, ip")),
        }
    }

    fn build_index(dimensions: usize, metric: &str) -> PyResult<Index> {
        let options = IndexOptions {
            dimensions,
            metric: Self::metric_from_name(metric)?,
            quantization: ScalarKind::F32,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            multi: false,
        };
        Index::new(&options)
            .map_err(|error| PyValueError::new_err(format!("failed to create index: {error}")))
    }
}

#[pymethods]
impl VectorIndex {
    #[new]
    #[pyo3(signature = (dimensions, metric = "cos"))]
    fn new(dimensions: usize, metric: &str) -> PyResult<Self> {
        if dimensions == 0 {
            return Err(PyValueError::new_err(
                "dimensions must be greater than zero",
            ));
        }
        let index = Self::build_index(dimensions, metric)?;
        Ok(Self {
            dimensions,
            metric: metric.to_string(),
            index,
            keys: Vec::new(),
        })
    }

    #[pyo3(signature = (keys, vectors))]
    fn add(&mut self, keys: Vec<u64>, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        if keys.len() != vectors.len() {
            return Err(PyValueError::new_err(
                "keys and vectors must have the same length",
            ));
        }
        let dimensions = self.dimensions;
        self.index
            .reserve(keys.len())
            .map_err(|error| PyValueError::new_err(format!("failed to reserve index: {error}")))?;
        for (key, vector) in keys.into_iter().zip(vectors) {
            if vector.len() != dimensions {
                return Err(PyValueError::new_err(format!(
                    "vector dimension mismatch: expected {dimensions}, got {}",
                    vector.len()
                )));
            }
            self.index
                .add(key, &vector)
                .map_err(|error| PyValueError::new_err(format!("failed to add vector: {error}")))?;
            self.keys.push(key);
        }
        Ok(())
    }

    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(u64, f32)>> {
        if query.len() != self.dimensions {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            )));
        }
        let matches = self
            .index
            .search(&query, k)
            .map_err(|error| PyValueError::new_err(format!("search failed: {error}")))?;
        Ok(matches.keys.into_iter().zip(matches.distances).collect())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let tmp_path = format!("{path}.tmp");
        self.index
            .save(&tmp_path)
            .map_err(|error| PyIOError::new_err(format!("failed to save vector index: {error}")))?;
        std::fs::rename(&tmp_path, path).map_err(|error| {
            PyIOError::new_err(format!("failed to finalize vector index save: {error}"))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, dimensions, metric = "cos"))]
    fn load(path: &str, dimensions: usize, metric: &str) -> PyResult<Self> {
        let mut index = Self::build_index(dimensions, metric)?;
        index
            .load(path)
            .map_err(|error| PyIOError::new_err(format!("failed to load vector index: {error}")))?;
        Ok(Self {
            dimensions,
            metric: metric.to_string(),
            index,
            keys: Vec::new(),
        })
    }

    fn __len__(&self) -> usize {
        self.keys.len()
    }
}

pub fn vector_index_exists(path: &str) -> bool {
    Path::new(path).is_file()
}
