use std::sync::atomic::{AtomicU32, Ordering};

use num_traits::Zero;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use stwo::prover::backend::simd::conversion::Pack;
use stwo::prover::backend::simd::m31::{N_LANES, PackedM31};

pub fn pack_values<T: Pack + Copy + Sync>(values: &[T]) -> Vec<T::SimdType>
where
    T::SimdType: Send,
{
    assert!(
        values.len().is_multiple_of(N_LANES),
        "pack_values: length {} is not a multiple of N_LANES ({N_LANES})",
        values.len(),
    );
    values.par_chunks_exact(N_LANES).map(|c| T::pack(c.try_into().unwrap())).collect()
}

/// Local copy of stwo-cairo's `AtomicMultiplicityColumn`, so that this crate does not depend on
/// stwo-cairo-prover (and, through it, on stwo-cairo-adapter and cairo-vm).
///
/// A column of multiplicities for lookup arguments. Allows increasing the multiplicity at a given
/// index. This version uses atomic operations to increase the multiplicity, and is `Send`.
pub struct AtomicMultiplicityColumn {
    data: Vec<PackedM31>,
}
impl AtomicMultiplicityColumn {
    /// Creates a new `AtomicMultiplicityColumn` with the given size. The elements are initialized
    /// to 0.
    pub fn new(size: usize) -> Self {
        Self { data: vec![PackedM31::zero(); size.div_ceil(N_LANES)] }
    }

    /// Atomically increments the multiplicity address by 1.
    ///
    /// # Safety
    /// Caller must ensure `address` is in bounds for the column (no bounds check is performed).
    pub fn increase_at(&self, address: u32) {
        let ptr = unsafe { (self.data.as_ptr() as *mut u32).add(address as usize) };
        unsafe { AtomicU32::from_ptr(ptr).fetch_add(1, Ordering::Relaxed) };
    }

    /// Returns the internal data as a Vec<PackedM31>. The last element of the vector is padded with
    /// zeros if needed.
    pub fn into_simd_vec(self) -> Vec<PackedM31> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use stwo::prover::backend::simd::m31::N_LANES;

    use super::AtomicMultiplicityColumn;

    #[test]
    fn test_atomic_multiplicities_column() {
        let size = N_LANES;
        let n_loops = 10;
        let col = AtomicMultiplicityColumn::new(size);
        let n_threads = 32;

        (0..n_threads).into_par_iter().for_each(|_| {
            (0..size * n_loops).for_each(|i| col.increase_at((i % size) as u32));
        });
        let result = col.into_simd_vec().into_iter().flat_map(|p| p.to_array().map(|v| v.0));

        for value in result {
            assert_eq!(value, n_threads * n_loops as u32);
        }
    }

    #[test]
    fn test_multiplicities_column_into_simd() {
        let mut rng = SmallRng::seed_from_u64(0u64);
        let expected_length = 6;
        let cpu_length = expected_length * N_LANES - 2;

        let multiplicity_column = AtomicMultiplicityColumn::new(cpu_length);
        (0..10 * N_LANES).for_each(|_| {
            let addr = rng.random_range(0..cpu_length as u32);
            multiplicity_column.increase_at(addr);
        });
        let actual_length = multiplicity_column.into_simd_vec().len();

        assert_eq!(actual_length, expected_length);
    }
}
