//! A proof of work before the batching coefficient of the opened columns.
//!
//! Inside `prove_ex` the channel sees, in order: the composition coefficient drawn, the
//! composition root mixed, the OODS point drawn, the sampled values mixed (`mix_felts` in
//! `CommitmentSchemeProver::prove_values`) and at once the batching coefficient drawn; the
//! verifier (`CommitmentSchemeVerifier::verify_values`) replays the same steps. A grind placed
//! before `prove_ex` would not bind that coefficient: the prover's own messages between the two
//! (the composition root, the sampled values) re-randomize it for free. Placed between the
//! sampled values and the draw it does: every fresh coefficient costs a fresh grind.
//!
//! [`BatchingGrindChannel`] wraps any channel and, once armed, notes the next `mix_felts` and at
//! the following draw grinds (prover) or checks the given nonce (verifier) on the inner channel,
//! then mixes the nonce. Unarmed it is the inner channel exactly. The in-circuit verifier
//! replays the same step (`circuits_stark_verifier::verify`, `ProofConfig::n_batching_pow_bits`).
use std::marker::PhantomData;

use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::qm31::SecureField;
use stwo::core::proof_of_work::GrindOps;
use stwo::core::vcs_lifted::hasher::Hasher;
use stwo::prover::backend::BackendForChannel;
use stwo::prover::backend::simd::SimdBackend;

/// The grind armed for the draw after the next `mix_felts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Armed {
    bits: u32,
    /// The verifier's side carries the nonce to check; the prover's grinds.
    nonce: Option<u64>,
}

/// What happened at the armed draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fired {
    /// The prover ground this nonce.
    Ground(u64),
    /// The verifier's nonce passed.
    Checked,
    /// The verifier's nonce failed; it was mixed all the same so the replay stays
    /// deterministic, and the caller refuses the proof.
    Failed,
}

/// A channel with a grind that can be armed for the draw after the next `mix_felts`.
#[derive(Clone, Debug, Default)]
pub struct BatchingGrindChannel<C: Channel> {
    inner: C,
    armed: Option<Armed>,
    /// A `mix_felts` happened since arming.
    pending: bool,
    fired: Option<Fired>,
}

impl<C: Channel> BatchingGrindChannel<C> {
    /// Wraps `inner`, unarmed.
    pub fn new(inner: C) -> Self {
        Self { inner, armed: None, pending: false, fired: None }
    }

    /// The inner channel.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Arms the grind for the draw after the next `mix_felts`: the prover (`nonce` absent)
    /// grinds `bits` there, the verifier checks `nonce`. Arming twice or over an unread outcome
    /// is a caller error.
    pub fn arm(&mut self, bits: u32, nonce: Option<u64>) {
        assert!(self.armed.is_none(), "the batching grind is already armed");
        assert!(self.fired.is_none(), "a batching grind outcome is unread");
        self.armed = Some(Armed { bits, nonce });
        self.pending = false;
    }

    /// The outcome of the armed draw; an error when the draw never came.
    pub fn disarm(&mut self) -> Result<Fired, String> {
        self.armed = None;
        self.pending = false;
        self.fired
            .take()
            .ok_or_else(|| "the grind before the batching coefficient never fired".into())
    }

    fn before_draw(&mut self)
    where
        SimdBackend: GrindOps<C>,
    {
        let Some(armed) = self.armed else {
            return;
        };
        if !self.pending {
            return;
        }
        let fired = match armed.nonce {
            None => {
                let nonce = <SimdBackend as GrindOps<C>>::grind(&self.inner, armed.bits);
                self.inner.mix_u64(nonce);
                Fired::Ground(nonce)
            }
            Some(nonce) => {
                let ok = self.inner.verify_pow_nonce(armed.bits, nonce);
                self.inner.mix_u64(nonce);
                if ok { Fired::Checked } else { Fired::Failed }
            }
        };
        self.fired = Some(fired);
        self.armed = None;
        self.pending = false;
    }
}

impl<C: Channel> Channel for BatchingGrindChannel<C>
where
    SimdBackend: GrindOps<C>,
{
    const BYTES_PER_HASH: usize = C::BYTES_PER_HASH;

    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        self.inner.verify_pow_nonce(n_bits, nonce)
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        self.inner.mix_u32s(data);
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        self.inner.mix_felts(felts);
        if self.armed.is_some() {
            self.pending = true;
        }
    }

    fn mix_u64(&mut self, value: u64) {
        self.inner.mix_u64(value);
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        self.before_draw();
        self.inner.draw_secure_felt()
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        self.before_draw();
        self.inner.draw_secure_felts(n_felts)
    }

    fn draw_u32s(&mut self) -> Vec<u32> {
        self.before_draw();
        self.inner.draw_u32s()
    }
}

/// The Merkle channel `MC` over [`BatchingGrindChannel`]: the same hasher, roots mixed as `MC`
/// mixes them.
#[derive(Default)]
pub struct BatchingGrindMerkleChannel<MC: MerkleChannel> {
    phantom: PhantomData<MC>,
}

impl<MC: MerkleChannel> MerkleChannel for BatchingGrindMerkleChannel<MC>
where
    SimdBackend: GrindOps<MC::C>,
{
    type C = BatchingGrindChannel<MC::C>;
    type H = MC::H;

    fn mix_hash(channel: &mut Self::C, hash: <Self::H as Hasher>::Hash) {
        MC::mix_hash(&mut channel.inner, hash);
    }
}

impl<C: Channel> GrindOps<BatchingGrindChannel<C>> for SimdBackend
where
    SimdBackend: GrindOps<C>,
{
    fn grind(channel: &BatchingGrindChannel<C>, pow_bits: u32) -> u64 {
        <SimdBackend as GrindOps<C>>::grind(&channel.inner, pow_bits)
    }
}

impl<MC: MerkleChannel> BackendForChannel<BatchingGrindMerkleChannel<MC>> for SimdBackend where
    SimdBackend: BackendForChannel<MC>
{
}

#[cfg(test)]
#[path = "batching_grind_test.rs"]
mod test;
