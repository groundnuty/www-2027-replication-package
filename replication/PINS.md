# Integrity pins changed by anonymization

The analysis refuses to run unless the code and configuration it loads match pinned digests. Anonymization changed
the bytes of some pinned files, so each affected pin was recomputed on the anonymized files with the code's own
digest functions. The logic of the code did not change (see the README), except for the tests adapted to the synthetic store-copy tables. Digests recorded inside the run data
(step 2) refer to the original values below.

| Pin | Original | Package |
|---|---|---|
| prereg-config grouping_digest (function-source digest) | `a654dc070d7176a9526a5d4e44c2521c391a66553d58dd53545e1510eca9badd` | `82b738c9ea7202da070d7f63fd67a741dda46c44556fabe80b1d15251413cec3` |
| prereg-config multiplicity_digest (function-source digest) | `8faa431b41291c48f68b50d4f549d2be4973c9e8c4ff6401bc121168ebe1e325` | `699ee5cf8fe614a836a78c7d63c0ffb6a6caf9d9b84623ec2ef02fe535cf02b7` |
| sha256 of data/analysis/weighted-holm-step3-alpha.json | `b8ac7a27434ba0b1894d087d54cdb59e7d095c9fef200c6cd3b9e5c36cfc6603` | `003d47ae65ebde262448392c1444f3570964d89c42951f5c9746fd94e38d4789` |
| sha256 of data/analysis/weighted-holm-step3.json | `f57c01f09e3e8c6879b697131e1fd61125c2757c7de95e58244b6429ff321b53` | `4a76d3718cc0f87128aea0840fe66d3d122b74d5ca94b744a0bcc489714e5707` |
| sha256 of data/analysis/prereg-config.json | `e46fd50494c689943cac508c9b2c2bd46f4aa65b233c0514096d5d4eae1313c0` | `ddca19da589e268ed5b88c8275c4d51106ab2cd40bf25920840848e63f5d97dd` |
| sha256 of data/analysis/kat-hosts.json | `9c733b19e8ce27d22c1fedd411e38648dc5d49365d1fe48c8e30ad1518634213` | `3e7be2e5a97a4122444052758cd77260ce56d57be2ef5282f1f785d4d1dc9e20` |
| sha256 of data/analysis/weighted-holm-step2.json | `42cd54d815614e4851f143796b82137807fafc6c0c63112bd78e13f5ea94350d` | `87dd9abaa9fe04181b95e968bf088a273309cd6c6fc332db2083c8ba14a9bc22` |
| sha256 of results/e189-exploratory-20261022/store-proofs.json | `8f71fc75d8045a7af380b9554c94eb141408e18c00ddd33e24b5195ccb412208` | `42f5bcaffb960ec85e85ad9d3970a23a310cbad43b3d38a8df3ca704da8f92fa` |
| sha256 of results/e189-exploratory-20261022/endpoint-copies.json | `c00bda611889a694acb6f48714806c39837e064bd592d99c21bbaac24fdfcae5` | `90bb0d3e33a421af9150220f3a3aaddd3a98ebc724c8b4087773b69e9da303c1` |
| sha256 of results/e189-exploratory-20261022/class-overrides.json | `5e43df1e4cbe228ca4287883ce36a53e391e07acde3fbeacf407169fd70c2c80` | `9aa84f10a22b89b25d76aef51ff7969de5574c7fec78d0e30288b36f4de4e382` |
| sha256 of results/e189-exploratory-20261022/attribution.py | `4eaf5a00405ab250db560d408ad5eca4d75e9e58c61f9b1816154768340dac53` | `9299922566fd8f0a3878663f813386f8eaee4f6a785db5f4bba5732ba527f875` |
| sha256 of results/e189-exploratory-20261022/extra-copies.json | `968e7621b239ca817ccf62c77a301cb142c2cbacc2cfecf81ac07a0bdd44c331` | `affd31ef2ba166a604b677d1cd827beeabacc6b04b886f31e82c652dceeabe66` |
| sha256 of results/e189-exploratory-20261022/copy-hosts.json | `34d1c7cb51d9bcf5f9915ac1a7ef9f9f2edf6fc67bd9010fd9546d247f161bce` | `31f4087fb4006b0900a64a34f9a5f4c224c7115ecd2b7865a71fd808646d17aa` |
| exploratory.py FROZEN_CODE_DIGEST (every file under harness/ and scripts/) | `917d47e000456a8023a3c3873f504413faff502b841a5d8c35da85c968d9f67d` | `6b50a93c0c458fe6c37d485ed5fe4bab4abcf96c1654e06b7cd9ebf45a907969` |
| exploratory.py UNTOUCHED_LIST_SHA256 (33 files) | `cdcf2db2d7f32224349d308503067415ec826bb48586b6e8d5117d6fc41633a3` | `e74b844c42b639cfba874078ee3ced6fc721b7d5c4ae1da121ef596844deba81` |

`FROZEN_CODE_DIGEST`'s original value is the digest of every file under `harness/` and `scripts/` of the frozen
analysis commit `847cc275`, including the scripts this package does not include (see the README); the package value
is the digest of this package's `harness/` and `scripts/`. It also moved because the comments and copy ids in `harness/agent/store_fault.py` and the ports in its `COPY_ENDPOINTS` were neutralized (syntax tree unchanged apart from string and integer constants). The scorer digest and `calibrated_inference_digest` are
unchanged: the comment neutralization was not applied to the files they cover, and no identity anonymization fell
inside the functions they hash. `grouping_digest` and `multiplicity_digest` hash the source text of their registered
functions, comments included, and move because anonymization rewrote comments inside some of them; the logic is unchanged.

Hashes of store endpoint addresses (in the store-copy tables, `exploratory.py` and its tests) are the hashes of the
anonymized addresses, so that they match anonymized run data; their original values are not listed.

The package value of the five store-copy table rows above (`store-proofs.json`, `endpoint-copies.json`,
`class-overrides.json`, `extra-copies.json`, `copy-hosts.json`) is the digest of the synthetic example table that replaced
the authors' table (the original is the digest of the authors' table); `admitted-fingerprints.json` is replaced the same
way and no code pins it. `copy-classes.json` is replaced the same way (invented copy names and classes, consistent with the synthetic tables), and its
`sources` record the digests of the synthetic tables; no code pins it.
