//! Proofs of work inside `prove_ex`: before the out-of-domain point, before the batching
//! coefficient of the opened columns and before every FRI folding coefficient.
//!
//! Inside `prove_ex` the channel sees, in order: the composition coefficient drawn, the
//! composition root mixed, the OODS point drawn, the sampled values mixed (`mix_felts` in
//! `CommitmentSchemeProver::prove_values`) and at once the batching coefficient drawn, then per
//! FRI layer its root mixed and its folding coefficient drawn, the last layer's coefficients
//! mixed (`mix_felts`), Stwo's own grind and the queries; the verifier
//! (`verify`, `CommitmentSchemeVerifier::verify_values`, `FriVerifier::commit`) replays the same
//! steps. A grind placed before `prove_ex` would bind none of these draws: the prover's own
//! messages between the two re-randomize them for free. Placed between the message a draw
//! answers and the draw it does: every fresh value costs a fresh grind.
//!
//! [`BatchingGrindChannel`] wraps any channel and, once armed
//! ([`BatchingGrindChannel::arm_grinds`]), grinds (prover) or checks the given nonce (verifier) on
//! the inner channel, then mixes the nonce, at each configured point ([`GrindBits`]): at the draw
//! after the first root (the composition root: the OODS point), at the draw after the first
//! `mix_felts` (the sampled values: the batching coefficient) and at the draw after every later
//! root (a FRI layer: its folding coefficient). Unarmed it is the inner channel exactly. The
//! in-circuit verifier replays the same steps (`circuits_stark_verifier::verify`,
//! `ProofConfig::n_oods_pow_bits`, `n_batching_pow_bits`, `n_fold_pow_bits`).
//! [`BatchingGrindChannel::arm`] arms the grind before the batching coefficient alone.
use std::marker::PhantomData;

use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::qm31::SecureField;
use stwo::core::proof_of_work::GrindOps;
use stwo::core::vcs_lifted::hasher::Hasher;
use stwo::prover::backend::BackendForChannel;
use stwo::prover::backend::simd::SimdBackend;

/// The bits of each proof of work inside `prove_ex`; `None` is the protocol without that one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GrindBits {
    /// Between the composition root and the OODS point.
    pub oods: Option<u32>,
    /// Between the sampled values and the batching coefficient of the opened columns.
    pub batching: Option<u32>,
    /// Between every FRI layer's root and its folding coefficient.
    pub fold: Option<u32>,
}

impl GrindBits {
    /// The grind before the batching coefficient alone.
    pub fn batching(bits: u32) -> Self {
        Self { oods: None, batching: Some(bits), fold: None }
    }
}

/// The nonces of the grinds of one proof: each present exactly when its bits are configured,
/// one fold nonce per FRI layer in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrindNonces {
    pub oods: Option<u64>,
    pub batching: Option<u64>,
    pub folds: Vec<u64>,
}

/// The point a pending grind is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Point {
    Oods,
    Batching,
    Fold,
}

/// The armed grinds.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Armed {
    bits: GrindBits,
    /// The verifier's side carries the nonces to check; the prover's grinds.
    expected: Option<GrindNonces>,
    /// A root was mixed since arming (the composition root).
    seen_root: bool,
    /// The sampled values were mixed since arming.
    seen_felts: bool,
    /// The FRI layers whose grind fired.
    folds: usize,
    /// The nonces ground or checked so far.
    nonces: GrindNonces,
    /// Whether every checked nonce passed and none was missing.
    ok: bool,
}

/// What happened at the armed draw of [`BatchingGrindChannel::arm`].
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

/// What the armed grinds did ([`BatchingGrindChannel::disarm_grinds`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// The nonces: the prover's ground ones, the verifier's as replayed.
    pub nonces: GrindNonces,
    /// The verifier's side: every nonce passed its check, none was missing and none was left
    /// over. Always true on the prover's side.
    pub ok: bool,
}

/// A channel with grinds that can be armed for the draws inside `prove_ex`.
#[derive(Clone, Debug, Default)]
pub struct BatchingGrindChannel<C: Channel> {
    inner: C,
    armed: Option<Armed>,
    /// The grind the next draw runs.
    pending: Option<Point>,
}

impl<C: Channel> BatchingGrindChannel<C> {
    /// Wraps `inner`, unarmed.
    pub fn new(inner: C) -> Self {
        Self { inner, armed: None, pending: None }
    }

    /// The inner channel.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Arms the grind before the batching coefficient alone, for the draw after the next
    /// `mix_felts`: the prover (`nonce` absent) grinds `bits` there, the verifier checks
    /// `nonce`. Arming twice is a caller error.
    pub fn arm(&mut self, bits: u32, nonce: Option<u64>) {
        self.arm_grinds(
            GrindBits::batching(bits),
            nonce.map(|nonce| GrindNonces { batching: Some(nonce), ..GrindNonces::default() }),
        );
    }

    /// Arms the grinds of `bits`, right before `prove_ex` or `verify`: the prover (`nonces`
    /// absent) grinds at each configured point, the verifier checks `nonces`. Arming twice is a
    /// caller error.
    pub fn arm_grinds(&mut self, bits: GrindBits, nonces: Option<GrindNonces>) {
        assert!(self.armed.is_none(), "the grinds are already armed");
        self.armed = Some(Armed {
            bits,
            expected: nonces,
            seen_root: false,
            seen_felts: false,
            folds: 0,
            nonces: GrindNonces::default(),
            ok: true,
        });
        self.pending = None;
    }

    /// The outcome of [`Self::arm`]'s draw; an error when the draw never came.
    pub fn disarm(&mut self) -> Result<Fired, String> {
        let prover = self.armed.as_ref().is_some_and(|armed| armed.expected.is_none());
        let outcome = self.disarm_grinds()?;
        let Some(nonce) = outcome.nonces.batching else {
            return Err("the grind before the batching coefficient never fired".into());
        };
        Ok(match (prover, outcome.ok) {
            (true, _) => Fired::Ground(nonce),
            (false, true) => Fired::Checked,
            (false, false) => Fired::Failed,
        })
    }

    /// The outcome of the armed grinds; an error when the channel was not armed, or a configured
    /// grind before the OODS point or the batching coefficient never fired (the FRI layers'
    /// count is the caller's to check against the proof's shape).
    pub fn disarm_grinds(&mut self) -> Result<Outcome, String> {
        self.pending = None;
        let armed = self.armed.take().ok_or("the grinds were not armed")?;
        if armed.bits.oods.is_some() && armed.nonces.oods.is_none() {
            return Err("the grind before the OODS point never fired".into());
        }
        if armed.bits.batching.is_some() && armed.nonces.batching.is_none() {
            return Err("the grind before the batching coefficient never fired".into());
        }
        // A verifier's nonce that no layer consumed is a proof of another shape.
        let left_over = armed.expected.as_ref().is_some_and(|expected| {
            expected.folds.len() != armed.folds
                || (expected.oods.is_some() && armed.bits.oods.is_none())
                || (expected.batching.is_some() && armed.bits.batching.is_none())
                || (!expected.folds.is_empty() && armed.bits.fold.is_none())
        });
        Ok(Outcome { nonces: armed.nonces, ok: armed.ok && !left_over })
    }

    /// A root was mixed: the composition root before the sampled values, a FRI layer's after.
    fn after_root(&mut self) {
        let Some(armed) = &mut self.armed else {
            return;
        };
        if armed.seen_felts {
            self.pending = armed.bits.fold.map(|_| Point::Fold);
        } else if !armed.seen_root {
            armed.seen_root = true;
            self.pending = armed.bits.oods.map(|_| Point::Oods);
        }
    }

    fn before_draw(&mut self)
    where
        SimdBackend: GrindOps<C>,
    {
        let Some(point) = self.pending.take() else {
            return;
        };
        let armed = self.armed.as_mut().expect("a grind is pending on an armed channel alone");
        let (bits, expected) = match point {
            Point::Oods => (armed.bits.oods, armed.expected.as_ref().map(|e| e.oods)),
            Point::Batching => (armed.bits.batching, armed.expected.as_ref().map(|e| e.batching)),
            Point::Fold => (
                armed.bits.fold,
                armed.expected.as_ref().map(|e| e.folds.get(armed.folds).copied()),
            ),
        };
        let bits = bits.expect("a pending grind is a configured one");
        let nonce = match expected {
            None => <SimdBackend as GrindOps<C>>::grind(&self.inner, bits),
            Some(nonce) => {
                // A missing nonce is replayed as zero and refused.
                let ok = nonce.is_some_and(|nonce| self.inner.verify_pow_nonce(bits, nonce));
                armed.ok &= ok;
                nonce.unwrap_or(0)
            }
        };
        self.inner.mix_u64(nonce);
        match point {
            Point::Oods => armed.nonces.oods = Some(nonce),
            Point::Batching => armed.nonces.batching = Some(nonce),
            Point::Fold => {
                armed.nonces.folds.push(nonce);
                armed.folds += 1;
            }
        }
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
        if let Some(armed) = &mut self.armed {
            // The sampled values arm the batching grind; the last FRI layer's coefficients
            // arm nothing (Stwo's own grind before the queries follows them).
            self.pending = if armed.seen_felts {
                None
            } else {
                armed.seen_felts = true;
                armed.bits.batching.map(|_| Point::Batching)
            };
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
        channel.after_root();
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
