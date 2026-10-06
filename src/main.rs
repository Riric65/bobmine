//! Bobminer — mineur de BobCoins pour https://bob.xem.yt/bobphone/api/
//!
//! Endpoints utilisés :
//!   1. GET  /mine/token?user_id=...               -> token HMAC d'authentification
//!   2. GET  /mine/challenge?user_id=...&token=... -> challenge hashcash
//!   3. POST /mine/submit {user_id, token, proof}  -> crédite les BC
//!
//! Algo : on brute-force des nonces jusqu'à ce que
//!   sha256("{prefix}:{nonce}") commence par `difficulty` zéros hex
//!   (= difficulty * 4 bits à zéro), puis on envoie proof = "{challenge_id}:{nonce}".
//!
//! Le serveur accepte n'importe quelle chaîne comme nonce (il hashe la chaîne
//! brute), donc on utilise un nonce de 6 caractères sur un alphabet de 93
//! symboles : le message fait alors 55 octets et tient sur **un seul bloc**
//! SHA-256, soit ~2x plus rapide que le nonce décimal de 10 chiffres.

use serde::Deserialize;
use sha2::digest::generic_array::GenericArray;
use sha2::digest::typenum::U64;
use sha2::{compress256, Digest, Sha256};
use std::env;
use std::error::Error;
use std::fmt;
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(cuda)]
mod gpu;

const API_BASE: &str = "https://bob.xem.yt/bobphone/api";
const USER_AGENT: &str = "bobminer/0.2 (+rust)";

/// Longueur de nonce attendue par le kernel CUDA (cuda/miner.cu, NONCE_LEN).
/// `choose_len` produit 6 pour 32 <= bits <= 36, soit difficulty 8..=9.
const GPU_NONCE_LEN: usize = 6;

type Blk = GenericArray<u8, U64>;

// ---------------------------------------------------------------------------
// Types API
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct TokenResp {
    ok: bool,
    #[serde(default)]
    error: String,
    #[serde(default)]
    token: String,
}

#[derive(Debug, Deserialize)]
struct Challenge {
    ok: bool,
    #[serde(default)]
    error: String,
    #[serde(default)]
    prefix: String,
    #[serde(default)]
    challenge_id: String,
    #[serde(default)]
    difficulty: usize,
    #[serde(default)]
    expires_in: u64,
    #[serde(default)]
    reward_bc: f64,
}

impl Challenge {
    /// `difficulty` zéros hex = difficulty * 4 bits à zéro.
    fn target_bits(&self) -> u32 {
        (self.difficulty * 4) as u32
    }
}

#[derive(Debug, Deserialize)]
struct SubmitResp {
    ok: bool,
    #[serde(default)]
    error: String,
    #[serde(default)]
    earned: f64,
    #[serde(default)]
    balance_bc: f64,
    #[serde(default)]
    hash: String,
    #[serde(default)]
    solve_ms: f64,
}

// ---------------------------------------------------------------------------
// Erreurs
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum ApiError {
    Http(reqwest::StatusCode, String),
    Unauthorized(String),
    RateLimited(u64),
    Request(reqwest::Error),
    Api(String),
    Other(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::Http(s, b) => write!(f, "HTTP {s}: {b}"),
            ApiError::Unauthorized(m) => write!(f, "auth refusée: {m}"),
            ApiError::RateLimited(secs) => write!(f, "rate limit, réessaie dans {secs}s"),
            ApiError::Request(e) => write!(f, "requête: {e}"),
            ApiError::Api(m) => write!(f, "API: {m}"),
            ApiError::Other(m) => write!(f, "{m}"),
        }
    }
}

impl Error for ApiError {}

fn map_http(status: reqwest::StatusCode, body: String) -> ApiError {
    let lower = body.to_lowercase();
    if status.as_u16() == 429 {
        return ApiError::RateLimited(5);
    }
    if lower.contains("login_required")
        || lower.contains("unauthorized")
        || lower.contains("invalid_token")
        || status == reqwest::StatusCode::UNAUTHORIZED
    {
        return ApiError::Unauthorized(body);
    }
    if lower.contains("rate") && lower.contains("limit") {
        return ApiError::RateLimited(5);
    }
    ApiError::Http(status, body)
}

// ---------------------------------------------------------------------------
// Client API
// ---------------------------------------------------------------------------

/// Rate limit optionnel partagé entre le thread principal et le prefetch.
/// Le serveur annonce `rate_limit_sec: 0`, donc 0 par défaut.
struct Throttle {
    min: Duration,
    last: Mutex<Option<Instant>>,
}

impl Throttle {
    fn new(min: Duration) -> Self {
        Throttle {
            min,
            last: Mutex::new(None),
        }
    }

    fn wait(&self) {
        if self.min.is_zero() {
            return;
        }
        let last = { *self.last.lock().unwrap() };
        if let Some(t) = last {
            let elapsed = t.elapsed();
            if elapsed < self.min {
                thread::sleep(self.min - elapsed);
            }
        }
    }

    fn mark(&self) {
        *self.last.lock().unwrap() = Some(Instant::now());
    }
}

#[derive(Clone)]
struct Api {
    client: reqwest::blocking::Client,
    user_id: String,
    token: Arc<Mutex<String>>,
    throttle: Arc<Throttle>,
}

impl Api {
    fn new(user_id: String, token: String, interval: Duration) -> Result<Self, ApiError> {
        let client = reqwest::blocking::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(ApiError::Request)?;
        Ok(Api {
            client,
            user_id,
            token: Arc::new(Mutex::new(token)),
            throttle: Arc::new(Throttle::new(interval)),
        })
    }

    fn token(&self) -> String {
        self.token.lock().unwrap().clone()
    }

    fn get(&self, path: &str) -> Result<String, ApiError> {
        self.throttle.wait();
        let resp = self
            .client
            .get(format!("{API_BASE}{path}"))
            .send()
            .map_err(ApiError::Request)?;
        self.throttle.mark();
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(map_http(status, body));
        }
        Ok(body)
    }

    fn fetch_token(&self) -> Result<String, ApiError> {
        let body = self.get(&format!("/mine/token?user_id={}", self.user_id))?;
        let parsed: TokenResp =
            serde_json::from_str(&body).map_err(|e| ApiError::Other(format!("json: {e} ({body})")))?;
        if !parsed.ok {
            return Err(ApiError::Api(parsed.error));
        }
        *self.token.lock().unwrap() = parsed.token.clone();
        Ok(parsed.token)
    }

    fn fetch_challenge(&self) -> Result<Challenge, ApiError> {
        let body = self.get(&format!(
            "/mine/challenge?user_id={}&token={}",
            self.user_id,
            self.token()
        ))?;
        let ch: Challenge =
            serde_json::from_str(&body).map_err(|e| ApiError::Other(format!("json: {e} ({body})")))?;
        if !ch.ok {
            return Err(ApiError::Api(ch.error));
        }
        Ok(ch)
    }

    /// Challenge + horodatage de réception : `expires_in` part de l'émission
    /// côté serveur, or avec le pipeline on peut recevoir le challenge bien
    /// avant de le miner (il vieillit pendant la ronde précédente).
    fn fetch_challenge_at(&self) -> Result<(Challenge, Instant), ApiError> {
        let ch = self.fetch_challenge()?;
        Ok((ch, Instant::now()))
    }

    /// Un submit invalide **brûle** le challenge : on ne soumet que des
    /// preuves vérifiées localement.
    fn submit(&self, challenge_id: &str, nonce: &str) -> Result<SubmitResp, ApiError> {
        let payload = serde_json::json!({
            "user_id": self.user_id,
            "token": self.token(),
            "proof": format!("{challenge_id}:{nonce}"),
        });
        let resp = self
            .client
            .post(format!("{API_BASE}/mine/submit"))
            .json(&payload)
            .send()
            .map_err(ApiError::Request)?;
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(map_http(status, body));
        }
        let out: SubmitResp =
            serde_json::from_str(&body).map_err(|e| ApiError::Other(format!("json: {e} ({body})")))?;
        if !out.ok {
            return Err(ApiError::Api(out.error));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Moteur de minage : squelette de bloc SHA-256 + odomètre
// ---------------------------------------------------------------------------

const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Alphabet des nonces : imprimables ASCII sauf `:` (séparateur du proof).
/// `"` et `\` sont acceptés, serde_json les échappe correctement.
fn build_alphabet() -> Vec<u8> {
    (0x21u8..=0x7e).filter(|c| *c != b':').collect()
}

/// Le hash (serialisé big-endian à partir de l'état) commence par `bits` zéros.
#[inline(always)]
fn state_hits(st: &[u32; 8], bits: u32) -> bool {
    let full = (bits / 32) as usize;
    for w in st.iter().take(full) {
        if *w != 0 {
            return false;
        }
    }
    let rem = bits % 32;
    if rem == 0 {
        return true;
    }
    (st[full] >> (32 - rem)) == 0
}

/// Vérification de référence avec l'impl canonique de sha2.
fn reference_hits(prefix: &str, nonce: &str, bits: u32) -> bool {
    let msg = format!("{prefix}:{nonce}");
    let h = Sha256::digest(msg.as_bytes());
    let full = (bits / 8) as usize;
    if h[..full].iter().any(|b| *b != 0) {
        return false;
    }
    let rem = bits % 8;
    rem == 0 || (h[full] >> (8 - rem)) == 0
}

/// Longueur de nonce : la plus petite L telles que K^L >= 4 * 2^bits
/// (marge ×4 pour éviter d'épuiser l'espace de recherche).
fn choose_len(k: u64, bits: u32) -> usize {
    let need = 4u128 << bits.min(60);
    let mut len = 1usize;
    let mut space = k as u128;
    while space < need && len < 12 {
        len += 1;
        space = space.saturating_mul(k as u128);
    }
    len
}

struct Engine {
    alpha: Vec<u8>,
    /// Prochain caractère dans l'odomètre, 0 = débordement.
    next_char: [u8; 256],
    first: u8,
    nonce_pos: usize,
    len: usize,
    space: u64,
    /// Squelette de 1 ou 2 blocs ; seuls les octets du nonce changent.
    skeleton: Vec<Blk>,
}

impl Engine {
    /// Construit le squelette pour `sha256("{prefix}:{nonce}")` avec un nonce
    /// de `len` caractères. Le padding SHA-256 est calculé une fois pour toutes.
    fn new(prefix: &str, len: usize) -> Self {
        let alpha = build_alphabet();
        let k = alpha.len() as u64;
        let plen = prefix.len();
        let d = plen + 1 + len; // longueur du message

        // Le nonce doit tenir dans le premier bloc (2 blocs max).
        assert!(len >= 1 && plen + 1 + len <= 64, "préfixe inattendu ({plen} octets)");

        let mut next_char = [0u8; 256];
        for (i, c) in alpha.iter().enumerate() {
            next_char[*c as usize] = if i + 1 < alpha.len() {
                alpha[i + 1]
            } else {
                0
            };
        }

        let pos = plen + 1;
        let first = alpha[0];
        let mut skeleton = Vec::with_capacity(2);
        let mut b1: Blk = GenericArray::default();
        b1[..plen].copy_from_slice(prefix.as_bytes());
        b1[plen] = b':';
        for i in 0..len {
            b1[pos + i] = alpha[0];
        }

        if d <= 55 {
            // Un seul bloc : message + 0x80 + zéros + longueur en bits.
            b1[d] = 0x80;
            b1[56..64].copy_from_slice(&((d as u64 * 8).to_be_bytes()));
            skeleton.push(b1);
        } else {
            // Message trop long pour son padding sur un bloc : 2 blocs.
            if d < 64 {
                b1[d] = 0x80;
            }
            let mut b2: Blk = GenericArray::default();
            if d == 64 {
                b2[0] = 0x80;
            }
            b2[56..64].copy_from_slice(&((d as u64 * 8).to_be_bytes()));
            skeleton.push(b1);
            skeleton.push(b2);
        }

        let mut space: u64 = 1;
        for _ in 0..len {
            space = space.saturating_mul(k);
        }

        Engine {
            first,
            alpha,
            next_char,
            nonce_pos: pos,
            len,
            space,
            skeleton,
        }
    }

    fn blocks(&self) -> &[Blk] {
        &self.skeleton
    }

    /// Écrit l'indice global `idx` comme nombre en base K dans le squelette.
    fn set_index(&self, idx: u64, blocks: &mut [Blk]) {
        let k = self.alpha.len() as u64;
        let mut x = idx;
        let mut i = self.len;
        while i > 0 {
            i -= 1;
            blocks[0][self.nonce_pos + i] = self.alpha[(x % k) as usize];
            x /= k;
        }
    }

    /// Passe au nonce suivant (odomètre). `false` = espace épuisé.
    fn increment(&self, blocks: &mut [Blk]) -> bool {
        let mut i = self.len;
        loop {
            if i == 0 {
                return false;
            }
            i -= 1;
            let c = blocks[0][self.nonce_pos + i];
            let n = self.next_char[c as usize];
            if n != 0 {
                blocks[0][self.nonce_pos + i] = n;
                return true;
            }
            blocks[0][self.nonce_pos + i] = self.first;
        }
    }

    fn nonce_at(&self, idx: u64) -> String {
        let mut blocks = self.skeleton.clone();
        self.set_index(idx, &mut blocks);
        let s = &blocks[0][self.nonce_pos..self.nonce_pos + self.len];
        String::from_utf8(s.to_vec()).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Mining multi-threads
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
enum Stop {
    Running,
    Found,
    Exhausted,
}

struct MineResult {
    nonce: Option<String>,
    hashes: u64,
    elapsed: Duration,
    exhausted: bool,
    space: u64,
    nonce_len: usize,
    blocks: usize,
}

/// Worker GPU : consomme de grands lots sur le même compteur atomique que les
/// threads CPU et met à jour `found` (le minimum gagnant est conservé).
/// Retourne `None` si CUDA est indisponible (message loggé une seule fois).
#[cfg(cuda)]
#[allow(clippy::too_many_arguments)]
fn spawn_gpu_worker(
    prefix: &str,
    bits: u32,
    deadline: Instant,
    space: u64,
    counter: &Arc<AtomicU64>,
    found: &Arc<AtomicU64>,
    hashes: &Arc<AtomicU64>,
    stop: &Arc<(Mutex<Stop>, std::sync::Condvar)>,
) -> Option<thread::JoinHandle<()>> {
    match gpu::Gpu::global() {
        Err(e) => {
            static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::SeqCst) {
                eprintln!("GPU indisponible: {e} — minage CPU uniquement");
            }
            None
        }
        Ok(cell) => {
            let prefix = prefix.to_string();
            let counter = Arc::clone(counter);
            let found = Arc::clone(found);
            let hashes = Arc::clone(hashes);
            let stop = Arc::clone(stop);
            Some(thread::spawn(move || {
                // Une seule session GPU à la fois : mine() joint ce thread
                // avant le challenge suivant.
                let mut gpu = match cell.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                if let Err(e) = gpu.prepare(&prefix) {
                    eprintln!("GPU: {e} — round en CPU seulement");
                    return;
                }
                let batch = gpu.batch;
                loop {
                    if found.load(Ordering::Relaxed) != u64::MAX {
                        break;
                    }
                    if Instant::now() >= deadline {
                        break;
                    }
                    let base = counter.fetch_add(batch, Ordering::Relaxed);
                    if base >= space {
                        let mut g = stop.0.lock().unwrap();
                        if *g == Stop::Running {
                            *g = Stop::Exhausted;
                            stop.1.notify_all();
                        }
                        break;
                    }
                    let n = batch.min(space - base);
                    match gpu.run_batch(base, n, bits) {
                        Ok(Some(idx)) => {
                            // Arrêt anticipé du kernel : on ne compte que
                            // les indices réellement parcourus.
                            let done = (idx.saturating_sub(base) + 1).min(n);
                            hashes.fetch_add(done, Ordering::Relaxed);
                            let mut cur = found.load(Ordering::Relaxed);
                            while idx < cur {
                                match found.compare_exchange_weak(
                                    cur,
                                    idx,
                                    Ordering::SeqCst,
                                    Ordering::Relaxed,
                                ) {
                                    Ok(_) => break,
                                    Err(actual) => cur = actual,
                                }
                            }
                            let mut g = stop.0.lock().unwrap();
                            *g = Stop::Found;
                            stop.1.notify_all();
                            break;
                        }
                        Ok(None) => {
                            // Pas de solution dans le lot (ou trouvée par un
                            // thread CPU : dans ce cas on sort à l'itération
                            // suivante sans surcompter).
                            if found.load(Ordering::Relaxed) == u64::MAX {
                                hashes.fetch_add(n, Ordering::Relaxed);
                            }
                        }
                        Err(e) => {
                            eprintln!("GPU: {e} — on continue en CPU");
                            break;
                        }
                    }
                }
            }))
        }
    }
}

#[cfg(not(cuda))]
#[allow(clippy::too_many_arguments)]
fn spawn_gpu_worker(
    _prefix: &str,
    _bits: u32,
    _deadline: Instant,
    _space: u64,
    _counter: &Arc<AtomicU64>,
    _found: &Arc<AtomicU64>,
    _hashes: &Arc<AtomicU64>,
    _stop: &Arc<(Mutex<Stop>, std::sync::Condvar)>,
) -> Option<thread::JoinHandle<()>> {
    None
}

/// Brute-force multi-threads : compteur atomique partagé, arrêt immédiat dès
/// qu'un nonce gagnant est trouvé ou que le challenge expire.
fn mine(prefix: &str, difficulty: usize, deadline: Instant, threads: usize, gpu: bool) -> MineResult {
    let bits = (difficulty * 4) as u32;
    let alpha_len = build_alphabet().len() as u64;
    let len = choose_len(alpha_len, bits);
    let engine = Arc::new(Engine::new(prefix, len));
    let blocks_count = engine.blocks().len();

    let found = Arc::new(AtomicU64::new(u64::MAX));
    let counter = Arc::new(AtomicU64::new(0));
    let hashes = Arc::new(AtomicU64::new(0));
    let stop: Arc<(Mutex<Stop>, std::sync::Condvar)> =
        Arc::new((Mutex::new(Stop::Running), std::sync::Condvar::new()));
    let start = Instant::now();

    // GPU en parallèle des threads CPU : même compteur, même signaux d'arrêt.
    // Le kernel CUDA est spécialisé pour le cas standard du serveur
    // (prefix de 48 caractères + nonce de 6) ; sinon on reste en CPU.
    let gpu_handle = if gpu && engine.len == GPU_NONCE_LEN {
        spawn_gpu_worker(
            prefix,
            bits,
            deadline,
            engine.space,
            &counter,
            &found,
            &hashes,
            &stop,
        )
    } else {
        None
    };

    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let engine = Arc::clone(&engine);
        let found = Arc::clone(&found);
        let counter = Arc::clone(&counter);
        let hashes = Arc::clone(&hashes);
        let stop = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            const BATCH: u64 = 8192;
            let mut blocks = engine.skeleton.clone();
            loop {
                if found.load(Ordering::Relaxed) != u64::MAX {
                    break;
                }
                if Instant::now() >= deadline {
                    break;
                }
                let base = counter.fetch_add(BATCH, Ordering::Relaxed);
                if base >= engine.space {
                    let mut g = stop.0.lock().unwrap();
                    if *g == Stop::Running {
                        *g = Stop::Exhausted;
                        stop.1.notify_all();
                    }
                    break;
                }
                let end = base.saturating_add(BATCH).min(engine.space);
                engine.set_index(base, &mut blocks);
                let mut done = 0u64;
                let mut hit = u64::MAX;
                for idx in base..end {
                    let mut st = IV;
                    compress256(&mut st, &blocks);
                    done += 1;
                    if state_hits(&st, bits) {
                        hit = idx;
                        break;
                    }
                    if !engine.increment(&mut blocks) {
                        break;
                    }
                }
                hashes.fetch_add(done, Ordering::Relaxed);
                if hit != u64::MAX {
                    // On garde le plus petit indice trouvé.
                    let mut cur = found.load(Ordering::Relaxed);
                    while hit < cur {
                        match found.compare_exchange_weak(
                            cur,
                            hit,
                            Ordering::SeqCst,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(actual) => cur = actual,
                        }
                    }
                    let mut g = stop.0.lock().unwrap();
                    *g = Stop::Found;
                    stop.1.notify_all();
                    break;
                }
            }
        }));
    }

    // Boucle d'attente : hashrate en direct, réveil immédiat à la solution.
    let mut last_report = start;
    {
        let mut guard = stop.0.lock().unwrap();
        while *guard == Stop::Running && Instant::now() < deadline {
            let (g, timeout) = stop
                .1
                .wait_timeout(guard, Duration::from_millis(250))
                .unwrap();
            guard = g;
            if timeout.timed_out() && last_report.elapsed() >= Duration::from_secs(2) {
                last_report = Instant::now();
                let done = hashes.load(Ordering::Relaxed);
                let dt = start.elapsed().as_secs_f64().max(0.001);
                eprintln!(
                    "   … {:>9.0} H/s  ({} hashes, {:.1}s)",
                    done as f64 / dt,
                    done,
                    dt
                );
            }
        }
    }

    for h in handles {
        let _ = h.join();
    }
    if let Some(h) = gpu_handle {
        let _ = h.join();
    }

    let n = found.load(Ordering::Relaxed);
    let exhausted = {
        let g = stop.0.lock().unwrap();
        *g == Stop::Exhausted
    };
    MineResult {
        nonce: (n != u64::MAX).then(|| engine.nonce_at(n)),
        hashes: hashes.load(Ordering::Relaxed),
        elapsed: start.elapsed(),
        exhausted,
        space: engine.space,
        nonce_len: len,
        blocks: blocks_count,
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Config {
    user_id: String,
    token: Option<String>,
    threads: usize,
    interval_secs: u64,
    max_rounds: Option<u64>,
    timeout_secs: u64,
    dry_run: bool,
    pipeline: bool,
    gpu: bool,
}

fn usage() -> ! {
    eprintln!(
        "bobminer — mineur de BobCoins (bobphone API)

USAGE:
    bobminer --user-id <ID> [OPTIONS]

OPTIONS:
    --user-id <ID>     ID utilisateur (ou env BOBMINE_USER_ID)   [requis]
    --token <HEX>      token HMAC pré-fait (sinon: GET /mine/token)
    --threads <N>      nombre de threads de minage (défaut: tous les cores)
    --interval <SEC>   délai min entre 2 GET challenge (défaut: 0 = aucun)
    --max <N>          arrêter après N challenges (défaut: illimité)
    --timeout <SEC>    arrêter le minage d'un challenge après SEC s (bench)
    --no-pipeline      ne pas pré-récupérer le challenge suivant
    --no-gpu           ne pas utiliser le GPU (CUDA) même s'il est dispo
    --dry-run          miner sans soumettre
    -h, --help         cette aide"
    );
    process::exit(2);
}

fn parse_args() -> Config {
    let mut user_id = env::var("BOBMINE_USER_ID").unwrap_or_default();
    let mut token = env::var("BOBMINE_TOKEN").ok().filter(|s| !s.is_empty());
    let mut threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut interval_secs = 0u64;
    let mut max_rounds = None;
    let mut timeout_secs = 0u64;
    let mut dry_run = false;
    let mut pipeline = true;
    let mut gpu = true;

    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| -> String {
            match args.next() {
                Some(v) => v,
                None => {
                    eprintln!("valeur manquante pour {name}");
                    usage();
                }
            }
        };
        match a.as_str() {
            "--user-id" => user_id = val("--user-id"),
            "--token" => token = Some(val("--token")),
            "--threads" => threads = val("--threads").parse().unwrap_or(threads).max(1),
            "--interval" => interval_secs = val("--interval").parse().unwrap_or(interval_secs),
            "--max" => max_rounds = Some(val("--max").parse().unwrap_or(1)),
            "--timeout" => timeout_secs = val("--timeout").parse().unwrap_or(0),
            "--no-pipeline" => pipeline = false,
            "--no-gpu" => gpu = false,
            "--dry-run" => dry_run = true,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("argument inconnu: {other}");
                usage();
            }
        }
    }

    if user_id.is_empty() {
        eprintln!("--user-id obligatoire (ou BOBMINE_USER_ID)");
        usage();
    }

    Config {
        user_id,
        token,
        threads,
        interval_secs,
        max_rounds,
        timeout_secs,
        dry_run,
        pipeline,
        gpu,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cfg = parse_args();
    let interval = Duration::from_secs(cfg.interval_secs);

    let api = match Api::new(
        cfg.user_id.clone(),
        cfg.token.clone().unwrap_or_default(),
        interval,
    ) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("erreur: {e}");
            process::exit(1);
        }
    };

    println!(
        "bobminer — user_id={}  threads={}  interval={}s  pipeline={}  gpu={}",
        cfg.user_id, cfg.threads, cfg.interval_secs, cfg.pipeline, cfg.gpu
    );

    // Init CUDA en amont : échec rapide, JIT du PTX payé une seule fois.
    #[cfg(cuda)]
    if cfg.gpu {
        match gpu::Gpu::global() {
            Ok(cell) => {
                if let Ok(g) = cell.lock() {
                    println!(
                        "GPU: {}  (lots de {:>9} indices par lancement)",
                        g.device_name, g.batch
                    );
                }
            }
            Err(e) => eprintln!("GPU indisponible: {e} — minage CPU uniquement"),
        }
    }
    #[cfg(not(cuda))]
    if cfg.gpu {
        eprintln!("build sans CUDA (nvcc absent) : minage CPU uniquement");
    }

    // Token HMAC : fourni ou récupéré via GET /mine/token.
    if api.token().is_empty() {
        match api.fetch_token() {
            Ok(t) => println!("token HMAC: {}…", &t[..t.len().min(16)]),
            Err(e) => {
                eprintln!("échec /mine/token: {e}");
                process::exit(1);
            }
        }
    }

    let mut round: u64 = 0;
    let mut total_earned = 0.0f64;
    let mut total_hashes: u64 = 0;
    let session_start = Instant::now();
    // Challenge récupéré en avance pendant que l'on mine le précédent.
    let mut pending: Option<thread::JoinHandle<Result<(Challenge, Instant), ApiError>>> = None;

    // Marge de sécurité sur le TTL avant de lancer un minage.
    let ttl_margin = Duration::from_secs(5);
    let fresh_enough = |c: &Challenge, at: Instant| -> bool {
        at + Duration::from_secs(c.expires_in.saturating_sub(2)) > Instant::now() + ttl_margin
    };

    loop {
        if let Some(max) = cfg.max_rounds {
            if round >= max {
                break;
            }
        }

        // ---- 1. Challenge (pré-récupéré pendant la ronde précédente) ----
        let mut got: Option<(Challenge, Instant)> = None;
        if let Some(h) = pending.take() {
            match h.join() {
                Ok(Ok(pair)) => got = Some(pair),
                Ok(Err(e)) => eprintln!("prefetch refusé: {e}"),
                Err(_) => eprintln!("prefetch interrompu"),
            }
        }
        // Un challenge pré-récupéré peut avoir trop vieilli pendant la ronde
        // précédente : dans ce cas on en demande un frais.
        while got.as_ref().map(|(c, at)| !fresh_enough(c, *at)).unwrap_or(true) {
            got = Some(match api.fetch_challenge_at() {
                Ok(pair) => pair,
                Err(e) => {
                    eprintln!("challenge impossible: {e}");
                    match e {
                        ApiError::Unauthorized(_) => {
                            if api.fetch_token().is_ok() {
                                thread::sleep(Duration::from_secs(2));
                                continue;
                            }
                            process::exit(1);
                        }
                        ApiError::RateLimited(s) => {
                            thread::sleep(Duration::from_secs(s.max(1)));
                            continue;
                        }
                        _ => {
                            thread::sleep(Duration::from_secs(5));
                            continue;
                        }
                    }
                }
            });
            break;
        }
        let (ch, received) = match got {
            Some(pair) => pair,
            None => continue, // erreur gérée plus haut : on retente
        };
        round += 1;

        // On lance déjà la récupération du challenge suivant : elle se fait
        // pendant le minage, donc zéro temps mort entre deux solutions.
        if cfg.pipeline {
            let api = api.clone();
            pending = Some(thread::spawn(move || api.fetch_challenge_at()));
        }

        let target = ch.target_bits();
        let k = build_alphabet().len() as u64;
        let nlen = choose_len(k, target);
        let mut space: u64 = 1;
        for _ in 0..nlen {
            space = space.saturating_mul(k);
        }
        // Deadline basé sur la réception, pas sur "maintenant" : le challenge
        // a pu vieillir pendant la ronde précédente.
        let mut deadline = received + Duration::from_secs(ch.expires_in.saturating_sub(2).max(5));
        if cfg.timeout_secs > 0 {
            deadline = deadline.min(Instant::now() + Duration::from_secs(cfg.timeout_secs));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        println!(
            "[{round}] challenge {}  prefix={}  diff={} ({} bits)  nonce={} car.  espace={:.2e}  TTL restant={:.0}s  reward={} BC",
            &ch.challenge_id,
            &ch.prefix,
            ch.difficulty,
            target,
            nlen,
            space as f64,
            remaining.as_secs_f64(),
            ch.reward_bc
        );

        // ---- 2. Mining -------------------------------------------------
        let res = mine(&ch.prefix, ch.difficulty, deadline, cfg.threads, cfg.gpu);
        total_hashes += res.hashes;
        let dt = res.elapsed.as_secs_f64().max(0.0001);
        let hps = res.hashes as f64 / dt;

        let nonce = match res.nonce {
            Some(n) => n,
            None => {
                if res.exhausted {
                    eprintln!(
                        "[{round}] espace épuisé ({} nonces de {} car.) sans solution — retry",
                        res.space, res.nonce_len
                    );
                } else {
                    eprintln!(
                        "[{round}] timeout {dt:.1}s : {:>9.0} H/s ({} hashes) — retry",
                        hps, res.hashes
                    );
                }
                continue;
            }
        };

        // Vérification avec l'impl de référence avant soumission :
        // un submit raté brûle définitivement le challenge.
        if !reference_hits(&ch.prefix, &nonce, target) {
            eprintln!("[{round}] incohérence locale (nonce={nonce}) — abandon");
            continue;
        }

        println!(
            "[{round}] trouvé nonce={nonce} en {dt:.3}s ({hps:.0} H/s, {} compression{})",
            res.blocks,
            if res.blocks == 1 { "" } else { "s" }
        );

        if cfg.dry_run {
            println!("[{round}] dry-run: soumission ignorée");
            continue;
        }

        // ---- 3. Submit -------------------------------------------------
        match api.submit(&ch.challenge_id, &nonce) {
            Ok(ok) => {
                total_earned += ok.earned;
                let elapsed = session_start.elapsed().as_secs_f64().max(0.001);
                let rate = total_earned / elapsed * 3600.0;
                println!(
                    "[{round}] SUBMIT OK  +{:.2} BC  solde={:.2} BC  (solve {:.0} ms, hash {})",
                    ok.earned, ok.balance_bc, ok.solve_ms, ok.hash
                );
                println!(
                    "[{round}] cadence: {:.1} BC/h  ({} BC en {:.0} s, {} h/s)",
                    rate,
                    total_earned,
                    elapsed,
                    (total_hashes as f64 / elapsed / 1000.0).round() as u64
                );
            }
            Err(e) => eprintln!("[{round}] soumission refusée: {e}"),
        }
    }

    let secs = session_start.elapsed().as_secs_f64().max(0.001);
    println!(
        "terminé: {} challenge(s) en {:.1}s  |  +{:.2} BC  ({:.1} BC/h)  |  {} hashes ({:.0} H/s moy.)",
        round,
        secs,
        total_earned,
        total_earned / secs * 3600.0,
        total_hashes,
        total_hashes as f64 / secs
    );
}
