//! Bench des différentes façons de calculer sha256(prefix:nonce)
//! pour n'en choisir qu'une dans le mineur.
//!
//! cargo run --release --bin shabench

use sha2::compress256;
use sha2::{Digest, Sha256};
use std::time::Instant;

const PREFIX: &str = "c6a7c962c8fe044d2a6ef6e9f8a32eb7431a4cb49d2b85d0"; // 48 hex
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn write_dec(buf: &mut [u8], mut n: u64) -> usize {
    let mut tmp = [0u8; 20];
    let mut i = 0;
    if n == 0 {
        tmp[0] = b'0';
        i = 1;
    } else {
        while n > 0 {
            tmp[i] = b'0' + (n % 10) as u8;
            n /= 10;
            i += 1;
        }
    }
    buf[..i].copy_from_slice(&tmp[..i]);
    for k in 0..i {
        buf[k] = tmp[i - 1 - k];
    }
    i
}

/// Variante A : l'API digest() du crate sha2 (ce qu'utilise le mineur actuel).
fn variant_a(nonce: u64) -> [u8; 32] {
    let mut buf = [0u8; 96];
    let n = PREFIX.len();
    buf[..n].copy_from_slice(PREFIX.as_bytes());
    buf[n] = b':';
    let l = write_dec(&mut buf[n + 1..], nonce);
    let out = Sha256::digest(&buf[..n + 1 + l]);
    let mut h = [0u8; 32];
    h.copy_from_slice(&out);
    h
}

/// Construit le bloc (ou les 2 blocs) avec le padding à la main.
#[inline(always)]
fn manual(nonce: u64, out: &mut [u8; 32]) {
    let mut buf = [0u8; 96];
    let n = PREFIX.len();
    buf[..n].copy_from_slice(PREFIX.as_bytes());
    buf[n] = b':';
    let l = write_dec(&mut buf[n + 1..], nonce);
    let d = n + 1 + l; // longueur du message

    let mut state = IV;
    if d <= 55 {
        // Un seul bloc : message + 0x80 + zéros + longueur en bits.
        let mut blk = [0u8; 64];
        blk[..d].copy_from_slice(&buf[..d]);
        blk[d] = 0x80;
        blk[56..64].copy_from_slice(&((d as u64 * 8).to_be_bytes()));
        compress256(&mut state, &[blk.into()]);
    } else {
        // Le message ne tient pas avec son padding sur un bloc.
        let mut b1 = [0u8; 64];
        if d >= 64 {
            b1.copy_from_slice(&buf[..64]);
        } else {
            b1[..d].copy_from_slice(&buf[..d]);
            b1[d] = 0x80;
        }
        let mut b2 = [0u8; 64];
        if d < 64 {
            b2[56..64].copy_from_slice(&((d as u64 * 8).to_be_bytes()));
        } else {
            b2[0] = 0x80;
            b2[56..64].copy_from_slice(&((d as u64 * 8).to_be_bytes()));
        }
        compress256(&mut state, &[b1.into(), b2.into()]);
    }
    for i in 0..8 {
        out[i * 4..i * 4 + 4].copy_from_slice(&state[i].to_be_bytes());
    }
}

/// Variante B : bloc(s) construits à la main + compress256.
fn variant_b(nonce: u64) -> [u8; 32] {
    let mut h = [0u8; 32];
    manual(nonce, &mut h);
    h
}

fn bench<F: Fn(u64) -> [u8; 32]>(name: &str, f: F, secs: f64) {
    // Démarre à un nonce de 10 chiffres (le cas réel : 59 octets de message).
    let mut n = 1_000_000_000u64;
    let mut count = 0u64;
    let mut acc = [0u8; 32];
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < secs {
        for _ in 0..100_000 {
            let h = f(n);
            n += 1;
            // Empêche l'optimiseur d'éliminer le calcul.
            acc[0] ^= h[0];
            acc[1] ^= h[1];
        }
        count += 100_000;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "{name:<28} {:>12.0} H/s   ({} hashes en {dt:.2}s, {:02x}{:02x} …)",
        count as f64 / dt,
        count,
        acc[0],
        acc[1]
    );
}

fn main() {
    // 1. Vérifie que les 2 variantes calculent la même chose.
    for n in [0u64, 7, 999_999, 1_000_000_000, 4_294_967_295, 123_456_789_012] {
        let a = variant_a(n);
        let b = variant_b(n);
        assert_eq!(a, b, "divergence pour nonce={n}");
        // Et que ça correspond à la définition serveur : hex commence par "0000"
        // (non, ici on vérifie juste l'égalité des implémentations).
    }
    println!("correction des variantes: OK\n");

    bench("A: Sha256::digest (actuel)", variant_a, 2.0);
    bench("B: blocs manuels + compress256", variant_b, 2.0);
    bench_six_chars("D: nonce 6 car. (1 seule compression)", 2.0);

    // Coût théorique : une seule compression, sans construction de message.
    let mut blk = [0u8; 64];
    blk[0] = 0x80;
    let mut st = IV;
    let mut count = 0u64;
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < 2.0 {
        for _ in 0..100_000 {
            let mut s = st;
            compress256(&mut s, &[blk.into()]);
            st[0] ^= s[0];
        }
        count += 100_000;
    }
    println!(
        "{:<28} {:>12.0} H/s   (plafond: 1 compression/message)",
        "C: compress256 seule",
        count as f64 / t.elapsed().as_secs_f64()
    );
}

/// Variante D : squelette de bloc préconstruit, seuls les 6 caractères du
/// nonce changent à chaque itération -> une seule compression par nonce.
fn bench_six_chars(name: &str, secs: f64) {
    use sha2::digest::generic_array::GenericArray;
    use sha2::digest::typenum::U64;
    type Blk = GenericArray<u8, U64>;

    const L: usize = 6;
    let alpha: Vec<u8> = (0x21u8..=0x7e).filter(|c| *c != b':').collect();
    let k = alpha.len();

    // Squelette : prefix + ':' + 6 caractères + 0x80 + longueur en bits.
    let mut arr: [Blk; 1] = [GenericArray::default()];
    let total = PREFIX.len() + 1 + L;
    assert!(total <= 55, "message doit tenir sur un seul bloc");
    {
        let b = &mut arr[0];
        b[..PREFIX.len()].copy_from_slice(PREFIX.as_bytes());
        b[PREFIX.len()] = b':';
        for i in 0..L {
            b[PREFIX.len() + 1 + i] = alpha[0];
        }
        b[total] = 0x80;
        b[56..64].copy_from_slice(&((total as u64 * 8).to_be_bytes()));
    }
    let pos = PREFIX.len() + 1;

    // Vérification contre l'impl de référence (sha256(prefix:nonce)).
    let reference = |n: &[u8]| -> [u8; 32] {
        let mut msg = Vec::with_capacity(PREFIX.len() + 1 + n.len());
        msg.extend_from_slice(PREFIX.as_bytes());
        msg.push(b':');
        msg.extend_from_slice(n);
        let h = Sha256::digest(&msg);
        let mut o = [0u8; 32];
        o.copy_from_slice(&h);
        o
    };
    let hash_of = |arr: &[Blk; 1]| -> [u8; 32] {
        let mut s = IV;
        compress256(&mut s, arr);
        let mut o = [0u8; 32];
        for i in 0..8 {
            o[i * 4..i * 4 + 4].copy_from_slice(&s[i].to_be_bytes());
        }
        o
    };
    let idx_of = |c: u8| alpha.iter().position(|a| *a == c).unwrap();

    let probes: &[&[u8]] = &[b"aaaaaa", b"zzzzzz", b"000000", b"A1:b2\\"];
    for probe in probes {
        assert_eq!(probe.len(), L, "les sondes doivent faire {L} caractères");
        let mut save = arr[0];
        for (i, c) in probe.iter().enumerate() {
            save[pos + i] = *c;
        }
        let one = [save];
        assert_eq!(hash_of(&one), reference(probe), "divergence pour {probe:?}");
    }
    println!("correction variante D: OK");

    let mut count = 0u64;
    let mut acc = [0u8; 4];
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < secs {
        for _ in 0..100_000 {
            // Odomètre de base k (les ':' n'existent pas dans l'alphabet).
            let mut i = L;
            loop {
                i -= 1;
                let idx = idx_of(arr[0][pos + i]) + 1;
                if idx < k {
                    arr[0][pos + i] = alpha[idx];
                    break;
                }
                arr[0][pos + i] = alpha[0];
                if i == 0 {
                    break;
                }
            }
            let mut s = IV;
            compress256(&mut s, &arr);
            acc[0] ^= (s[0] >> 24) as u8;
            acc[1] ^= (s[0] >> 16) as u8;
        }
        count += 100_000;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "{name:<28} {:>12.0} H/s   ({} hashes en {dt:.2}s, {:02x}{:02x} …)",
        count as f64 / dt,
        count,
        acc[0],
        acc[1]
    );
}
