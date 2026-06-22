# TODO / state of play

Branch: `full-regularisation` (CCD-Sophie). Talk: `~/Git/nagoya-talk` on `main`.

## Headline result: a competitive ONE-parameter "per-new-split" CCD

We tested Alexei's idea of unifying RegCCD's `alpha` (split expansion) and KRegCCD's `mu`
(escape) into a single per-new-split rate. It works.

RSV2 held-out log-prob / tree (train 1000 / test 1000, n=129; `GRegCCDRSV2`):

| model | params | held-out logP/tree | notes |
|-------|:------:|-------------------:|-------|
| GRegCCD (global Z = prod counts / Z) | 1 | -71.6 | wrong normalisation |
| conditional 1-mu, uniform escape      | 1 | -72.7 | normalisation fixed, escape terrible |
| **conditional 1-mu, eps-reserve escape** | **1** | **-58.7** | the real model (mu ~ 0.0077) |
| KRegCCD                                | 2 | -56.2 | the target |
| CCD1 (common trees only)               | - | -51.3 | |

The 1-parameter model is within **2.5 nats** of two-parameter KRegCCD, and BEATS it on
common trees (-51.3 vs -52.3; raw CCPs, no alpha). The residual gap is reserve DEPTH
(our order-2 M2/M3 truncation vs KRegCCD's full depth + tail), not the idea.

### The model (conditional per-new-split, per-clade normalised, no global Z)
At each observed clade C:
- observed split: `(1-mu) * count / total_C`  (= CCD1 scaled, so matches CCD1 on common trees up to (n-1)log(1-mu))
- escape (novel split, region top): `eps_C^(m-1)` where m = boundary size of the escape region
  (eps per new split, recombinations included)
- `eps_C` solved per clade from `M2*eps + M3*eps^2 = mu`:
  - M2(C) = # recombinations (split C into two observed subclades, split unseen)
  - M3(C) = # one-novel-clade boundary-3 escape resolutions

Currently lives as static methods in `src/ccd/experiments/regularisation/GRegCCDRSV2.java`
(`scoreReserve`, `epsFor`, `boundarySize`, `M2/M3`/`splitPairs`).

## Next steps (ranked)

1. **Close the 2.5-nat gap**: add deeper reserve orders (M4+) and a geometric tail correction
   to `epsFor`, mirroring KRegCCD's reserve. Check how often eps falls back (M2=M3=0):
   `fallbackClades` is tracked but not printed -- print it.
2. **Promote to a model class** `MRegCCD` (or similar) implementing the distribution interface,
   so the conditional-reserve model sits alongside CCD1/regCCD/KRegCCD (currently driver-only).
3. **Fair fit**: fit mu by leave-k-out CV (not the test-held-out grid) for an apples-to-apples
   comparison with KRegCCD's CV-fitted mu.
4. **Try alpha smoothing** on the observed CCPs (optional 2nd knob) to see if it helps the novel
   trees without hurting common -- but the point was ONE parameter, so keep this a sanity check.
5. **PIT calibration** of the 1-mu model (add to `KRegPITExperiment` as a 4th model).

## Other artifacts from this session (all committed)

- `GRegCCD` (exact small-n, src/ccd/model/GRegCCD.java) + `GRegZApprox` (large-n Z) + tests.
  Note: the GLOBAL GRegCCD underperforms (normalisation); kept as the worked-through negative result.
- KReg small-example analysis: 4/5/6-taxon enumeration, RF-decay geometric kernel
  (`P(T) ~ eps^(RF/2)`), MDS of RF distances. Drivers: `KRegFiveTaxaTest`, `KRegSixTaxaTest`,
  `KRegSingleTreeDecayTest`; plots `doc/plot_rfprob.py`, `doc/plot_five_taxa_mds.py`.
- musweep 3-curve calibration figure (`doc/plot_musweep.py`, cvLogP column in `RSV2ThreeModelTable`).
- Nagoya talk slides: four-taxon worked example, 3-curve calibration, geometric-kernel slide.

## In-flight PRs / loose ends

- PR #13 (yangsoph): `rsv2-talk-experiments` (3-model PIT).
- PR #14 (yangsoph): `doc/kreg-four-taxa` (15 four-taxon probabilities, symbolic doc).
- Local branch `doc/kreg-four-taxa` is redundant (already on the fork / PR #14) -- safe to delete.
- Stale untracked, NOT ours: `doc/sa-ccd1.*`, `src/ccd/algorithms/RankedTreeCounting*`,
  `doc/pit-yule50-fixedalpha-noccd0.pdf`.

## Build & run (non-obvious classpath)

Needs sibling repos `../beast2`, `../BeastFX`, `../CCD` built.
```
CP="build:../beast2/build:../BeastFX/build:$(ls ../beast2/lib/*.jar | tr '\n' ':')../beast2/lib/junit/junit-platform-console-standalone-1.8.2.jar"
javac -d build -cp "$CP" $(find src/ccd -name "*.java" -not -path "*/experiments/*")
javac -d build -cp "build:$CP" src/test/ccd/model/GRegCCDTest.java
java -cp "$CP" org.junit.platform.console.ConsoleLauncher -c test.ccd.model.GRegCCDTest --disable-banner
# RSV2 run (RSV2.trees under ../bayesianPhylogeneticInstability/data/zenodo/...):
java -Xmx6g -cp "$CP" ccd.experiments.regularisation.GRegCCDRSV2 "$RSV2" 0.1 1000
```
