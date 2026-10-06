# bobminer

Mineur de BobCoins (monnaie fictive du BobPhone) écrit en Rust.
**CPU + GPU (CUDA)** : ~3 GH/s, ~4 000 BC/h — [licence MIT](LICENSE).

Il n'utilise que les endpoints de l'API `https://bob.xem.yt/bobphone/api/` :

| Étape | Endpoint | Rôle |
|---|---|---|
| 1 | `GET /mine/token?user_id=...` | obtient un token HMAC d'authentification |
| 2 | `GET /mine/challenge?user_id=...&token=...` | récupère un challenge hashcash |
| 3 | `POST /mine/submit` `{user_id, token, proof}` | soumet la preuve, crédite les BC |

## Démarrage rapide

**1. Installe Rust** (une seule fois)

| Système | Comment |
|---|---|
| Windows | télécharge `rustup-init.exe` sur [rustup.rs](https://rustup.rs), lance-le, `Entrée` partout |
| Linux / macOS | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |

**2. Compile** — le script te dit si la version GPU est active :

* Windows : double-clique sur **`build.bat`**
* Linux / macOS : **`./build.sh`**

**3. Lance**

```bash
# Windows
target\release\bobminer.exe --user-id TON_ID

# Linux / macOS
./target/release/bobminer --user-id TON_ID
```

`TON_ID` = identifiant de ton compte BobPhone. Tape `--help` pour toutes les
options, `--dry-run` pour tester sans soumettre.

### Encore plus vite : CUDA (optionnel)

Sans CUDA, le mineur tourne en CPU (~230 M H/s). Avec un GPU NVIDIA et le
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installé, le
kernel GPU se compile **tout seul** au moment du build (le script le
détecte et te le dit) : **~3 GH/s, soit ~14× plus rapide**. Rien d'autre à
faire : installe le Toolkit puis relance `build.bat` / `build.sh`.

## Algo client

Le challenge renvoie un `prefix`, un `challenge_id` et une `difficulty`
(zéros hex à obtenir, `difficulty * 4` bits à zéro) :

```
sha256("{prefix}:{nonce}").hexdigest() commence par `difficulty` zéros hex
```

et on envoie `proof = "{challenge_id}:{nonce}"`.

### Optimisations

1. **Nonce alphanumérique de 6 caractères.** Le serveur hashe la chaîne brute
   (vérifié : `sha256(prefix:abc)` correspond au hash qu'il renvoie), il ne
   parse pas le nonce en entier. Avec 6 caractères sur un alphabet de 93
   symboles, le message fait 55 octets → il tient sur **un seul bloc**
   SHA-256, alors que le nonce décimal (10 chiffres) en demandait 2.
   Espace de recherche : 93⁶ = 6.5×10¹¹ (largement suffisant jusqu'à 36 bits,
   la longueur passe à 7 automatiquement au-delà).

2. **Squelette de bloc préconstruit.** Le padding SHA-256 (longueur du
   message, `0x80`) ne dépend pas du nonce : on le calcule une fois, on ne
   réécrit que les 6 octets du nonce à chaque essai, puis on appelle
   directement `sha2::compress256` (boucle interne zéro copie).

3. **SHA-NI.** Le crate `sha2` détecte les instructions SHA-NI de la CPU à
   l'exécution (actives sur ce machine : i5-12400F).

4. **Pipeline.** Le challenge suivant est récupéré *pendant* le minage du
   courant (un challenge émis n'invalide pas les précédents, vérifié) : zéro
   temps mort réseau entre deux solutions. Le TTL est calculé depuis la date
   de réception réelle, donc un challenge pré-récupéré trop vieux est écarté.

5. **Multi-threads** : compteur atomique partagé (lots de 8192), arrêt
   immédiat via condvar dès qu'un nonce gagnant est trouvé.

Résultat CPU : **~230 M H/s** sur 12 threads (contre 91 M H/s avec l'impl
initiale `Sha256::digest` + nonce décimal).

### Minage GPU (CUDA)

Au-delà du CPU, le mineur embarque un kernel CUDA (`cuda/miner.cu`) :

- le message fixe de 55 octets (prefix 48 + `:` + nonce 6) tient sur **un
  seul bloc** SHA-256 ; le préfixe complet est en mémoire constante, seuls
  les 6 octets du nonce changent ;
- chaque thread parcourt un index global **contigu** : conversion de l'indice
  de départ en base 93 (6 divisions, une seule fois) puis odomètre +1 par
  essai — aucune division dans la boucle ;
- vérification des `bits` premiers bits sur H0/H1 puis `atomicMin` sur
  l'indice gagnant, arrêt anticipé du kernel dès qu'une solution existe ;
- lancements par lots de ~2,3×10⁸ indices (≈ 75 ms), piochés sur le **même
  compteur atomique** que les threads CPU : le plus petit indice gagne.

`build.rs` compile le PTX au build time (`nvcc`, chemin auto-détecté ou
variable `NVCC`) puis le mineur le charge via l'API Driver (`libcuda.so`,
livrée par le driver NVIDIA, pas de cudart à installer) et le JITe une seule
fois au démarrage. Sans `nvcc`, le build reste 100 % CPU (cfg `cuda` off) ;
`--no-gpu` désactive le GPU à l'exécution.

Le kernel est spécialisé pour le cas standard du serveur (prefix de 48
caractères + nonce de 6, soit difficulty 8..=9) : si `choose_len` demande 5
ou 7 caractères (difficulty 7 ou 10), il reste de côté et le CPU mine seul.
Les solutions GPU passent par la même vérification de référence (`Sha256`
canonique) que les solutions CPU avant soumission.

Résultat sur **RTX 3060 12 GB** : ~3,1 GH/s (GPU) + 0,23 GH/s (CPU)
**≈ 3,3 GH/s**, soit ×14 sur le CPU seul.

### Bench interne

```bash
cargo run --release --bin shabench
```

Compare les variantes : `Sha256::digest` (~9.5 M H/s), blocs manuels
(~10.8 M H/s), squelette 6 caractères (~25 M H/s), et le plafond théorique
d'une compression.

Le kernel CUDA se bench de la même façon (autonome, sans Rust) :

```bash
nvcc -O3 -DBENCH cuda/miner.cu -o /tmp/bench_cuda
/tmp/bench_cuda <prefix48> <bits> <nb_hashes>   # ex: 32 8000000000
/tmp/bench_cuda <prefix48> --hash <index>       # hash GPU de contrôle
```

## Points d'attention API

- **Un submit invalide brûle définitivement le challenge**
  (`unknown_or_expired_challenge` ensuite) : le mineur re-vérifie la solution
  avec l'impl de référence `Sha256` avant d'envoyer.
- `rate_limit_sec: 0` : pas de limite, `--interval` est à 0 par défaut
  (reste réglable si le serveur en remet une).
- Difficulté ajustée par le serveur entre 7 et 10 (28 → 40 bits),
  TTL 600 s, `target_solve_ms: 25000`.
- À 230 M H/s (CPU) : diff 8 ≈ 19 s en moyenne, diff 9 ≈ 300 s (TTL serré).
  À ~3,3 GH/s (CPU+GPU) : diff 8 ≈ 1,3 s, diff 9 ≈ 21 s, diff 10 ≈ 5,5 min
  (le serveur reste alors autour de 8-9, cible 25 s).

## Commandes

La compilation se fait avec `build.bat` / `build.sh` (voir *Démarrage
rapide* ci-dessus) ; équivalent manuel : `cargo build --release`.

```bash
# boucle infinie
./target/release/bobminer --user-id TON_ID

# un seul challenge (test)
./target/release/bobminer --user-id TON_ID --max 1

# mesurer le hashrate sans soumettre
./target/release/bobminer --user-id TON_ID --dry-run --timeout 12
```

Options :

```
--user-id <ID>     ID utilisateur (ou env BOBMINE_USER_ID)   [requis]
--token <HEX>      token HMAC pré-fait (sinon: GET /mine/token)
--threads <N>      nombre de threads (défaut: tous les cores)
--interval <SEC>   délai min entre 2 GET challenge (défaut: 0 = aucun)
--max <N>          arrêter après N challenges (défaut: illimité)
--timeout <SEC>    arrêter le minage d'un challenge après SEC s (bench)
--no-pipeline      ne pas pré-récupérer le challenge suivant
--no-gpu           ne pas utiliser le GPU (CUDA) même s'il est dispo
--dry-run          miner sans soumettre
```

Sortie typée :

```
bobminer — user_id=TON_ID  threads=12  interval=0s  pipeline=true  gpu=true
GPU: NVIDIA GeForce RTX 3060  (lots de 234881024 indices par lancement)
token HMAC: d2ba09f90c622421…
[27] challenge 93f48f9e578dd04da0168d5f4c557f44  prefix=27cd83e4…  diff=9 (36 bits)  nonce=6 car.  espace=6.47e11  TTL restant=598s  reward=24 BC
   … 3212430244 H/s  (68005402112 hashes, 20.5s)
[27] trouvé nonce=*NDKqE en 20.528s (3212430244 H/s, 1 compression)
[27] SUBMIT OK  +24.00 BC  solde=297.13 BC  (solve 20793 ms, hash 000000000bb790e5…)
[27] cadence: 4113.3 BC/h  (63 BC en 55 s, 3110616 h/s)
```
