use circuit_common::Qm31OpsTraceGenerator;
use circuit_common::preprocessed::PreprocessedCircuit;
use circuit_verifier::circuit_claim::{CircuitInteractionElements, lookup_sum};
pub use circuit_verifier::circuit_proof::CircuitProof;
use circuit_verifier::statement::{INTERACTION_POW_BITS, all_circuit_components};
use circuit_verifier::verify::CircuitPublicData;
use circuits_stark_verifier::proof::{Proof, ProofConfig};
use circuits_stark_verifier::proof_from_stark_proof::{nonce_value, proof_from_stark_proof};
use num_traits::Zero;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::qm31::QM31;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::ExtendedStarkProof;
use stwo::core::proof_of_work::GrindOps;
use stwo::core::utils::MaybeOwned;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sMerkleHasher};
pub use stwo::prover::backend::simd::SimdBackend;
pub use stwo::prover::mempool::BaseColumnPool;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::{
    CommitmentSchemeProver, CommitmentTreeProver, ComponentProver, ProvingError, prove_ex,
};
use stwo_constraint_framework::PREPROCESSED_TRACE_IDX;

use crate::batching_grind::{BatchingGrindMerkleChannel, Fired};
use crate::circuit_air::circuit_components::CircuitComponents;
use crate::circuit_hash::compute_circuit_hash;
use crate::witness::trace::{TraceGenerator, write_interaction_trace, write_trace};

const COMPOSITION_POLYNOMIAL_LOG_DEGREE_BOUND: u32 = 1;

#[cfg(test)]
#[path = "prover_test.rs"]
pub mod test;

pub fn prove_circuit_assignment(
    values: &[QM31],
    preprocessed_circuit: &PreprocessedCircuit,
    base_column_pool: &BaseColumnPool<SimdBackend>,
    pcs_config: PcsConfig,
) -> Result<CircuitProof<Blake2sMerkleHasher>, ProvingError> {
    prove_circuit_assignment_with_channel::<Blake2sM31MerkleChannel>(
        values,
        preprocessed_circuit,
        base_column_pool,
        pcs_config,
    )
}

/// [`prove_circuit_assignment`] with a proof of work of `batching_pow_bits` between the sampled
/// values and the batching coefficient of the opened columns ([`crate::batching_grind`]); its
/// nonce is [`CircuitProof::batching_pow_nonce`]. A verifier checks it with
/// `ProofConfig::n_batching_pow_bits` set to the same bits.
pub fn prove_circuit_assignment_with_batching_pow(
    values: &[QM31],
    preprocessed_circuit: &PreprocessedCircuit,
    base_column_pool: &BaseColumnPool<SimdBackend>,
    pcs_config: PcsConfig,
    batching_pow_bits: u32,
) -> Result<CircuitProof<Blake2sMerkleHasher>, ProvingError> {
    let (twiddles, preprocessed_tree) = commit_preprocessed::<
        BatchingGrindMerkleChannel<Blake2sM31MerkleChannel>,
    >(preprocessed_circuit, base_column_pool, pcs_config);
    prove_circuit_with_precompute_and_batching_pow::<Blake2sM31MerkleChannel>(
        base_column_pool,
        &twiddles,
        preprocessed_circuit,
        MaybeOwned::Owned(preprocessed_tree),
        values,
        pcs_config,
        batching_pow_bits,
    )
}

pub fn prove_circuit_assignment_with_channel<MC>(
    values: &[QM31],
    preprocessed_circuit: &PreprocessedCircuit,
    base_column_pool: &BaseColumnPool<SimdBackend>,
    pcs_config: PcsConfig,
) -> Result<CircuitProof<MC::H>, ProvingError>
where
    MC: MerkleChannel,
    SimdBackend: stwo::prover::backend::BackendForChannel<MC>,
{
    let (twiddles, preprocessed_tree) =
        commit_preprocessed::<MC>(preprocessed_circuit, base_column_pool, pcs_config);
    prove_circuit_with_precompute::<MC>(
        base_column_pool,
        &twiddles,
        preprocessed_circuit,
        MaybeOwned::Owned(preprocessed_tree),
        values,
        pcs_config,
    )
}

/// The twiddles and the committed preprocessed tree of `preprocessed_circuit`.
fn commit_preprocessed<MC>(
    preprocessed_circuit: &PreprocessedCircuit,
    base_column_pool: &BaseColumnPool<SimdBackend>,
    pcs_config: PcsConfig,
) -> (TwiddleTree<SimdBackend>, CommitmentTreeProver<SimdBackend, MC>)
where
    MC: MerkleChannel,
    SimdBackend: stwo::prover::backend::BackendForChannel<MC>,
{
    // Precompute twiddles.
    // Account for blowup factor and for composition polynomial calculation (taking the max since
    // the composition polynomial is split prior to LDE).
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(
            preprocessed_circuit.trace_log_size
                + std::cmp::max(
                    pcs_config.fri_config.log_blowup_factor,
                    COMPOSITION_POLYNOMIAL_LOG_DEGREE_BOUND,
                ),
        )
        .circle_domain()
        .half_coset,
    );

    let preprocessed_trace = preprocessed_circuit.preprocessed_trace.get_trace::<SimdBackend>();
    let preprocessed_trace_polys = SimdBackend::interpolate_columns(preprocessed_trace, &twiddles);

    let store_polynomials_coefficients = true;
    let preprocessed_tree = CommitmentTreeProver::<SimdBackend, MC>::new(
        preprocessed_trace_polys,
        pcs_config.fri_config.log_blowup_factor,
        &twiddles,
        store_polynomials_coefficients,
        pcs_config.preprocessed_lifting_log_size,
        base_column_pool,
    );
    (twiddles, preprocessed_tree)
}

pub fn prove_circuit_with_precompute<'a, MC>(
    base_column_pool: &BaseColumnPool<SimdBackend>,
    twiddles: &TwiddleTree<SimdBackend>,
    preprocessed_circuit: &PreprocessedCircuit,
    preprocessed_tree: MaybeOwned<'a, CommitmentTreeProver<SimdBackend, MC>>,
    values: &[QM31],
    pcs_config: PcsConfig,
) -> Result<CircuitProof<MC::H>, ProvingError>
where
    MC: MerkleChannel,
    SimdBackend: stwo::prover::backend::BackendForChannel<MC>,
{
    prove_circuit_on_channel::<MC>(
        base_column_pool,
        twiddles,
        preprocessed_circuit,
        preprocessed_tree,
        values,
        pcs_config,
        |components, channel, commitment_scheme| {
            Ok((prove_ex::<SimdBackend, MC>(components, channel, commitment_scheme, true)?, None))
        },
    )
}

/// [`prove_circuit_with_precompute`] on `MC`'s channel wrapped in [`BatchingGrindChannel`], armed
/// for a proof of work of `batching_pow_bits` before the batching coefficient.
pub fn prove_circuit_with_precompute_and_batching_pow<'a, MC>(
    base_column_pool: &BaseColumnPool<SimdBackend>,
    twiddles: &TwiddleTree<SimdBackend>,
    preprocessed_circuit: &PreprocessedCircuit,
    preprocessed_tree: MaybeOwned<
        'a,
        CommitmentTreeProver<SimdBackend, BatchingGrindMerkleChannel<MC>>,
    >,
    values: &[QM31],
    pcs_config: PcsConfig,
    batching_pow_bits: u32,
) -> Result<CircuitProof<MC::H>, ProvingError>
where
    MC: MerkleChannel,
    SimdBackend: stwo::prover::backend::BackendForChannel<MC>,
{
    prove_circuit_on_channel::<BatchingGrindMerkleChannel<MC>>(
        base_column_pool,
        twiddles,
        preprocessed_circuit,
        preprocessed_tree,
        values,
        pcs_config,
        |components, channel, commitment_scheme| {
            channel.arm(batching_pow_bits, None);
            let proof = prove_ex::<SimdBackend, BatchingGrindMerkleChannel<MC>>(
                components,
                channel,
                commitment_scheme,
                true,
            )?;
            match channel.disarm() {
                Ok(Fired::Ground(nonce)) => Ok((proof, Some(nonce))),
                outcome => panic!("the batching grind did not run as the prover's: {outcome:?}"),
            }
        },
    )
}

/// The circuit prover on `MC`'s channel; `prove` runs the STARK prover on the components and
/// returns the proof with the nonce of the grind before the batching coefficient, if any.
fn prove_circuit_on_channel<'a, MC>(
    base_column_pool: &BaseColumnPool<SimdBackend>,
    twiddles: &TwiddleTree<SimdBackend>,
    preprocessed_circuit: &PreprocessedCircuit,
    preprocessed_tree: MaybeOwned<'a, CommitmentTreeProver<SimdBackend, MC>>,
    values: &[QM31],
    pcs_config: PcsConfig,
    prove: impl FnOnce(
        &[&dyn ComponentProver<SimdBackend>],
        &mut MC::C,
        CommitmentSchemeProver<'_, SimdBackend, MC>,
    ) -> Result<(ExtendedStarkProof<MC::H>, Option<u64>), ProvingError>,
) -> Result<CircuitProof<MC::H>, ProvingError>
where
    MC: MerkleChannel,
    SimdBackend: stwo::prover::backend::BackendForChannel<MC>,
{
    let PreprocessedCircuit {
        preprocessed_trace,
        first_permutation_row,
        n_outputs,
        trace_log_size: _,
    } = preprocessed_circuit;
    let trace_generator = TraceGenerator {
        qm31_ops_trace_generator: Qm31OpsTraceGenerator {
            first_permutation_row: *first_permutation_row,
        },
    };

    // Setup protocol.
    let channel = &mut MC::C::default();

    // Mix channel salt. Note that we first reduce it modulo `M31::P`, then cast it as QM31.
    let channel_salt = 0_u32;
    channel.mix_felts(&[channel_salt.into()]);
    pcs_config.mix_into(channel);
    let mut commitment_scheme = CommitmentSchemeProver::<SimdBackend, MC>::with_memory_pool(
        pcs_config,
        twiddles,
        base_column_pool,
    );

    commitment_scheme.set_store_polynomials_coefficients();

    // Grab the preprocessed root for the circuit hash before it is consumed by `commit_tree` below.
    let preprocessed_root = preprocessed_tree.commitment.root();

    // Preprocessed trace.
    commitment_scheme.commit_tree(preprocessed_tree, channel);

    // Base trace.
    let mut tree_builder = commitment_scheme.tree_builder();
    let (claim, component_log_sizes, interaction_generator) = write_trace(
        values,
        preprocessed_trace.clone(),
        *n_outputs,
        &mut tree_builder,
        &trace_generator,
        twiddles,
    );

    let circuit_hash = compute_circuit_hash::<MC::H>(
        &component_log_sizes,
        pcs_config.fri_config.log_blowup_factor,
        preprocessed_root,
    );
    MC::mix_hash(channel, circuit_hash);
    claim.mix_into(channel);
    tree_builder.commit(channel);

    // Draw interaction elements.
    let interaction_pow_nonce = SimdBackend::grind(channel, INTERACTION_POW_BITS);
    channel.mix_u64(interaction_pow_nonce);
    let interaction_elements = CircuitInteractionElements::draw(channel);

    // Interaction trace.
    let mut tree_builder = commitment_scheme.tree_builder();
    let interaction_claim = write_interaction_trace(
        &component_log_sizes,
        interaction_generator,
        &mut tree_builder,
        &interaction_elements,
        twiddles,
    );

    // Validate lookup argument.
    assert_eq!(lookup_sum(&claim, &interaction_claim, &interaction_elements), QM31::zero());

    interaction_claim.mix_into(channel);
    tree_builder.commit(channel);
    // Component provers.
    let circuit_components = CircuitComponents::new(
        &interaction_elements,
        &interaction_claim,
        &component_log_sizes,
        &preprocessed_trace.ids(),
    );
    let components = circuit_components.component_provers();

    // Prove stark.
    let (stark_proof, batching_pow_nonce) = prove(&components, channel, commitment_scheme)?;
    Ok(CircuitProof {
        pcs_config,
        claim,
        interaction_pow_nonce,
        batching_pow_nonce,
        interaction_claim,
        stark_proof,
        channel_salt,
        circuit_hash,
    })
}

pub fn prepare_circuit_proof_for_circuit_verifier(
    circuit_proof: CircuitProof<Blake2sMerkleHasher>,
) -> (Proof<QM31>, CircuitPublicData) {
    let CircuitProof {
        pcs_config,
        claim,
        interaction_pow_nonce,
        batching_pow_nonce,
        interaction_claim,
        stark_proof,
        channel_salt,
        circuit_hash: _,
    } = circuit_proof;

    let public_data = CircuitPublicData { output_values: claim.output_values.clone() };

    let proof_config = ProofConfig::new(
        &all_circuit_components::<QM31>(),
        stark_proof.proof.sampled_values[PREPROCESSED_TRACE_IDX].len(),
        &pcs_config,
        INTERACTION_POW_BITS,
    );

    let mut proof = proof_from_stark_proof(
        &stark_proof,
        &proof_config,
        interaction_claim.claimed_sums.into_array().to_vec(),
        interaction_pow_nonce,
        channel_salt,
    );
    // The nonce of the grind before the batching coefficient, when the proof has one; the bits
    // it is checked against are the verifier's own (`ProofConfig::n_batching_pow_bits`).
    proof.batching_pow_nonce = batching_pow_nonce.map(nonce_value);
    (proof, public_data)
}
