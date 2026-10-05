use stwo::core::channel::{Blake2sM31Channel, Channel, MerkleChannel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::proof_of_work::GrindOps;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::backend::simd::SimdBackend;

use super::{BatchingGrindChannel, BatchingGrindMerkleChannel, Fired};

type GrindChannel = BatchingGrindChannel<Blake2sM31Channel>;
type GrindMerkleChannel = BatchingGrindMerkleChannel<Blake2sM31MerkleChannel>;

fn felts(n: u32) -> Vec<SecureField> {
    (0..n).map(|i| SecureField::from(M31::from_u32_unchecked(i + 1))).collect()
}

/// The sequence inside `prove_ex` after arming: a draw (the composition coefficient), a root, a
/// draw (the OODS point), the sampled values, the draw the grind binds, then the FRI draws. The
/// grind fires at the third draw alone, and the verifier's replay agrees.
#[test]
fn test_grind_fires_at_the_draw_after_the_sampled_values() {
    let bits = 6;
    let mut prover = GrindChannel::default();
    prover.mix_u32s(&[1, 2, 3]);
    let mut verifier = prover.clone();
    prover.arm(bits, None);
    let a = prover.draw_secure_felt();
    assert!(prover.fired.is_none());
    GrindMerkleChannel::mix_hash(&mut prover, Blake2sHash([7; 32]));
    let b = prover.draw_secure_felt();
    assert!(prover.fired.is_none());
    prover.mix_felts(&felts(5));
    let state_before = prover.inner().clone();
    let c = prover.draw_secure_felt();
    let Some(Fired::Ground(nonce)) = prover.fired else { panic!("the prover grinds") };
    assert!(state_before.verify_pow_nonce(bits, nonce));
    let d = prover.draw_secure_felt();
    assert_eq!(prover.disarm(), Ok(Fired::Ground(nonce)));

    verifier.arm(bits, Some(nonce));
    assert_eq!(verifier.draw_secure_felt(), a);
    GrindMerkleChannel::mix_hash(&mut verifier, Blake2sHash([7; 32]));
    assert_eq!(verifier.draw_secure_felt(), b);
    verifier.mix_felts(&felts(5));
    assert_eq!(verifier.draw_secure_felt(), c);
    assert_eq!(verifier.draw_secure_felt(), d);
    assert_eq!(verifier.disarm(), Ok(Fired::Checked));

    // A nonce that fails the check is refused and the draws diverge.
    let bad = (0..).map(|k| nonce + k).find(|&n| !state_before.verify_pow_nonce(bits, n)).unwrap();
    let mut forged = GrindChannel::default();
    forged.mix_u32s(&[1, 2, 3]);
    forged.arm(bits, Some(bad));
    forged.draw_secure_felt();
    GrindMerkleChannel::mix_hash(&mut forged, Blake2sHash([7; 32]));
    forged.draw_secure_felt();
    forged.mix_felts(&felts(5));
    assert_ne!(forged.draw_secure_felt(), c);
    assert_eq!(forged.disarm(), Ok(Fired::Failed));
}

/// Unarmed the channel is the inner one; armed but never mixed, the grind never fires and
/// disarming says so.
#[test]
fn test_unarmed_channel_is_the_inner_one_and_an_unfired_grind_is_an_error() {
    let mut ours = GrindChannel::default();
    let mut inner = Blake2sM31Channel::default();
    ours.mix_u64(9);
    inner.mix_u64(9);
    ours.mix_felts(&felts(3));
    inner.mix_felts(&felts(3));
    assert_eq!(ours.draw_secure_felts(3), inner.draw_secure_felts(3));
    assert_eq!(ours.draw_u32s(), inner.draw_u32s());
    let nonce = <SimdBackend as GrindOps<GrindChannel>>::grind(&ours, 5);
    assert!(inner.verify_pow_nonce(5, nonce));
    ours.arm(5, None);
    ours.draw_secure_felt();
    assert!(ours.disarm().is_err());
}
