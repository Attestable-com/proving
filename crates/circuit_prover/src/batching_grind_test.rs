use stwo::core::channel::{Blake2sM31Channel, Channel, MerkleChannel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::proof_of_work::GrindOps;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::backend::simd::SimdBackend;

use super::{BatchingGrindChannel, BatchingGrindMerkleChannel, Fired, GrindBits, GrindNonces};

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
    let ground = |channel: &GrindChannel| channel.armed.as_ref().unwrap().nonces.batching;
    let a = prover.draw_secure_felt();
    assert!(ground(&prover).is_none());
    GrindMerkleChannel::mix_hash(&mut prover, Blake2sHash([7; 32]));
    let b = prover.draw_secure_felt();
    assert!(ground(&prover).is_none());
    prover.mix_felts(&felts(5));
    let state_before = prover.inner().clone();
    let c = prover.draw_secure_felt();
    let Some(nonce) = ground(&prover) else { panic!("the prover grinds") };
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

/// The transcript of `prove_ex` after arming, with `layers` FRI layers: returns the draws.
fn replay(channel: &mut GrindChannel, layers: u8) -> Vec<SecureField> {
    let mut draws = vec![channel.draw_secure_felt()];
    GrindMerkleChannel::mix_hash(channel, Blake2sHash([7; 32]));
    draws.push(channel.draw_secure_felt());
    channel.mix_felts(&felts(5));
    draws.push(channel.draw_secure_felt());
    for layer in 0..layers {
        GrindMerkleChannel::mix_hash(channel, Blake2sHash([layer; 32]));
        draws.push(channel.draw_secure_felt());
    }
    // The last layer, Stwo's own nonce and the queries: no grind of ours.
    channel.mix_felts(&felts(1));
    channel.mix_u64(11);
    draws.push(SecureField::from(M31::from_u32_unchecked(channel.draw_u32s()[0] >> 1)));
    draws
}

/// Every configured grind fires at its draw: the OODS point's after the composition root, the
/// batching coefficient's after the sampled values, one per FRI layer after its root, none
/// before the queries. The sequence is Stwo's channel's with the nonces mixed at those points,
/// and the verifier's replay accepts exactly those nonces.
#[test]
fn test_grinds_fire_before_the_oods_point_the_batching_and_every_folding_coefficient() {
    let bits = GrindBits { oods: Some(3), batching: Some(6), fold: Some(4) };
    let mut prover = GrindChannel::default();
    prover.mix_u32s(&[1, 2, 3]);
    let start = prover.clone();
    prover.arm_grinds(bits, None);
    let draws = replay(&mut prover, 3);
    let outcome = prover.disarm_grinds().unwrap();
    assert!(outcome.ok);
    let nonces = outcome.nonces;
    assert!(nonces.oods.is_some() && nonces.batching.is_some());
    assert_eq!(nonces.folds.len(), 3);

    // By hand on the inner channel.
    let mut inner = start.inner().clone();
    let mut by_hand = vec![inner.draw_secure_felt()];
    Blake2sM31MerkleChannel::mix_hash(&mut inner, Blake2sHash([7; 32]));
    assert!(inner.verify_pow_nonce(3, nonces.oods.unwrap()));
    inner.mix_u64(nonces.oods.unwrap());
    by_hand.push(inner.draw_secure_felt());
    inner.mix_felts(&felts(5));
    assert!(inner.verify_pow_nonce(6, nonces.batching.unwrap()));
    inner.mix_u64(nonces.batching.unwrap());
    by_hand.push(inner.draw_secure_felt());
    for (layer, nonce) in nonces.folds.iter().enumerate() {
        Blake2sM31MerkleChannel::mix_hash(&mut inner, Blake2sHash([layer as u8; 32]));
        assert!(inner.verify_pow_nonce(4, *nonce));
        inner.mix_u64(*nonce);
        by_hand.push(inner.draw_secure_felt());
    }
    assert_eq!(draws[..draws.len() - 1], by_hand[..]);

    let verify = |nonces: GrindNonces, layers: u8| {
        let mut verifier = start.clone();
        verifier.arm_grinds(bits, Some(nonces));
        let replayed = replay(&mut verifier, layers);
        (replayed, verifier.disarm_grinds().unwrap().ok)
    };
    assert_eq!(verify(nonces.clone(), 3), (draws.clone(), true));
    // A changed nonce at each point, a missing fold nonce and one left over are refused.
    let bad = |nonce: u64, bits: u32, state: &Blake2sM31Channel| {
        (0..).map(|k| nonce + k).find(|&n| !state.verify_pow_nonce(bits, n)).unwrap()
    };
    let mut state = start.inner().clone();
    state.draw_secure_felt();
    Blake2sM31MerkleChannel::mix_hash(&mut state, Blake2sHash([7; 32]));
    let mut forged = nonces.clone();
    forged.oods = Some(bad(nonces.oods.unwrap(), 3, &state));
    assert!(!verify(forged, 3).1);
    let mut forged = nonces.clone();
    forged.folds[2] ^= 1 << 40;
    assert!(!verify(forged, 3).1);
    let mut forged = nonces.clone();
    forged.folds.pop();
    assert!(!verify(forged, 3).1);
    assert!(!verify(nonces.clone(), 2).1);
    let mut forged = nonces.clone();
    forged.oods = None;
    assert!(!verify(forged, 3).1);
}

/// Unconfigured grinds do not fire, and a verifier given a nonce for one refuses.
#[test]
fn test_unconfigured_grinds_do_not_fire() {
    let mut prover = GrindChannel::default();
    let start = prover.clone();
    prover.arm_grinds(GrindBits::default(), None);
    let draws = replay(&mut prover, 2);
    assert_eq!(prover.disarm_grinds().unwrap().nonces, GrindNonces::default());
    let mut unarmed = start.clone();
    assert_eq!(replay(&mut unarmed, 2), draws);
    let mut verifier = start;
    verifier.arm_grinds(
        GrindBits::default(),
        Some(GrindNonces { oods: Some(1), ..GrindNonces::default() }),
    );
    replay(&mut verifier, 2);
    assert!(!verifier.disarm_grinds().unwrap().ok);
}
