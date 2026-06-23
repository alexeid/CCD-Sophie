package ccd.experiments.regularisation;

import beast.base.evolution.tree.Tree;
import ccd.model.AbstractCCD;
import ccd.model.ITreeDistribution;
import ccd.model.KRegCCD;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Sweeps the {@link KRegPITExperiment} PIT calibration test over several models, sample sizes and
 * replicates, pooling the randomized PIT across replicates into one histogram per (model, size).
 *
 * <p>Pooling is statistically sound: each test tree's PIT is computed under <em>its own</em>
 * replicate's fitted model, and if every replicate is calibrated then each PIT is Uniform(0,1), so
 * their mixture is Uniform(0,1) too. Pooling simply gives a smoother, higher-power calibration
 * check than any single replicate (the worst single-replicate noise is at the smallest sample
 * size). The chi-square and KS tests are run on the pooled values.</p>
 *
 * <p>Expects the figshare Yule-style layout: {@code <dataRoot>/rep<r>/run1} (training) and
 * {@code .../run2} (testing), each containing one {@code <prefix>*.trees} posterior file. The trees
 * file is found by globbing, not by reconstructing {@code <prefix>-<r>.trees}, so a mislabelled rep
 * number does not silently drop the replicate (e.g. rep100's run trees are named {@code yule-n50-0.trees}).
 * The single combined tidy TSV it writes feeds straight into {@code doc/plot_pit.py}.</p>
 *
 * <h3>Parallelism and reproducibility</h3>
 * Replicates are processed in parallel (one task per rep, {@code threads} at a time). Within a rep
 * each (train, test) file is loaded <em>once per sample size</em> and shared across the three models,
 * instead of re-parsing it per model. Each (model, size, rep) cell gets its own {@link Random},
 * deterministically seeded from {@code (seed, rep, sizeIndex, modelIndex)} via {@link #mix}, rather
 * than relying on the shared global BEAST {@code Randomizer}; model fitting and the pooled
 * {@code (alpha, mu)} are therefore bit-identical across runs and core counts.
 *
 * <p>The PIT <em>histogram</em> is reproducible bit-for-bit only single-threaded. Under parallelism
 * the Monte Carlo sample is not bit-identical run-to-run: the sampler consumes its (seeded) RNG while
 * walking clade/partition structures whose iteration order follows object-allocation order, and
 * concurrent allocation across threads varies that order. Each draw is still a valid sample from the
 * same fitted distribution -- the effect is exactly that of a different RNG seed, i.e. Monte Carlo
 * noise that the pooling over replicates averages out (the L1 noise floor scales like 1/sqrt(m)) --
 * so the calibration verdict is unchanged; the histogram counts simply differ within that noise.
 *
 * <p>Usage: {@code KRegPITSweep <dataRoot> [prefix=yule-n50] [startRep=1] [endRep=10]
 * [sizes=300,1000,3000] [models=kreg,ccd1,ccd0] [numBins=20] [numSamples=100000] [seed=42]
 * [out=pit_sweep_hist.tsv] [threads=#cores]}</p>
 */
public class KRegPITSweep {

    /** PIT results for one (model, size, rep) cell. */
    private record Cell(List<Double> v, int zeroed, int scored) {
    }

    /** All cells for one replicate, indexed {@code [modelIndex][sizeIndex]}, plus the fitted params
     *  of the models that have them: KRegCCD {@code (alpha, mu)} and MRegCCD {@code mu}. */
    private record RepOut(int rep, Cell[][] cells, double[] kregAlpha, double[] kregMu, double[] mregMu) {
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: KRegPITSweep <dataRoot> [prefix=yule-n50] [startRep=1] [endRep=10] "
                    + "[sizes=300,1000,3000] [models=kreg,ccd1,ccd0] [numBins=20] [numSamples=100000] "
                    + "[seed=42] [out=pit_sweep_hist.tsv] [threads=#cores]");
            return;
        }

        String dataRoot = args[0];
        String prefix = args.length > 1 ? args[1] : "yule-n50";
        int startRep = args.length > 2 ? Integer.parseInt(args[2]) : 1;
        int endRep = args.length > 3 ? Integer.parseInt(args[3]) : 10;
        int[] sizes = parseInts(args.length > 4 ? args[4] : "300,1000,3000");
        String[] models = (args.length > 5 ? args[5] : "kreg,ccd1,ccd0").split(",");
        int numBins = args.length > 6 ? Integer.parseInt(args[6]) : KRegPITExperiment.DEFAULT_NUM_BINS;
        int numSamples = args.length > 7 ? Integer.parseInt(args[7]) : KRegPITExperiment.DEFAULT_NUM_SAMPLES;
        long seed = args.length > 8 ? Long.parseLong(args[8]) : KRegPITExperiment.DEFAULT_SEED;
        String out = args.length > 9 ? args[9] : "pit_sweep_hist.tsv";
        int threads = args.length > 10 ? Integer.parseInt(args[10]) : Runtime.getRuntime().availableProcessors();
        String paramsOut = (out.endsWith(".tsv") ? out.substring(0, out.length() - 4) : out) + ".params.tsv";

        long t0 = System.currentTimeMillis();
        int numReps = Math.max(0, endRep - startRep + 1);
        System.out.printf(Locale.US, "PIT sweep: reps %d..%d, sizes %s, models %s, %d samples, seed %d, %d threads%n",
                startRep, endRep, Arrays.toString(sizes), Arrays.toString(models), numSamples, seed, threads);

        // One task per replicate; each loads its files once per size and scores every model.
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        AtomicInteger done = new AtomicInteger();
        List<Future<RepOut>> futures = new ArrayList<>(numReps);
        for (int rep = startRep; rep <= endRep; rep++) {
            final int r = rep;
            futures.add(pool.submit(repTask(dataRoot, prefix, r, sizes, models, numSamples, seed,
                    done, numReps, t0)));
        }
        pool.shutdown();

        // Pool the PIT across reps, in rep order (order does not affect the pooled histogram).
        List<List<Double>>[] vPool = pools(models.length, sizes.length);
        int[] zeroedPool = new int[models.length * sizes.length];
        List<RepOut> repOuts = new ArrayList<>(numReps);
        try {
            for (Future<RepOut> f : futures) {
                RepOut ro = f.get();
                repOuts.add(ro);
                for (int mi = 0; mi < models.length; mi++) {
                    for (int si = 0; si < sizes.length; si++) {
                        Cell c = ro.cells()[mi][si];
                        if (c == null) continue; // rep had no trees file
                        vPool[mi * sizes.length + si].get(0).addAll(c.v());
                        zeroedPool[mi * sizes.length + si] += c.zeroed();
                    }
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            pool.shutdownNow();
            throw new IOException("sweep task failed", e);
        }

        try (PrintWriter w = new PrintWriter(new File(out));
             PrintWriter pw = new PrintWriter(new File(paramsOut))) {
            KRegPITExperiment.writeTidyHeader(w);
            // long format: one row per fitted model (kreg has alpha+mu, mreg has mu; alpha = NaN).
            pw.println("model\tsampleSize\trep\talpha\tmu");
            for (int si = 0; si < sizes.length; si++) { // (size, rep) order within each model block
                for (RepOut ro : repOuts) {
                    if (!Double.isNaN(ro.kregMu()[si])) {
                        pw.printf(Locale.US, "kreg\t%d\t%d\t%.6f\t%.6f%n",
                                sizes[si], ro.rep(), ro.kregAlpha()[si], ro.kregMu()[si]);
                    }
                }
            }
            for (int si = 0; si < sizes.length; si++) {
                for (RepOut ro : repOuts) {
                    if (!Double.isNaN(ro.mregMu()[si])) {
                        pw.printf(Locale.US, "mreg\t%d\t%d\tNaN\t%.6f%n",
                                sizes[si], ro.rep(), ro.mregMu()[si]);
                    }
                }
            }

            for (int mi = 0; mi < models.length; mi++) {
                for (int si = 0; si < sizes.length; si++) {
                    List<Double> pooled = vPool[mi * sizes.length + si].get(0);
                    int zeroed = zeroedPool[mi * sizes.length + si];
                    if (pooled.isEmpty()) {
                        System.err.printf(Locale.US, "no scored trees for %s size=%d%n", models[mi], sizes[si]);
                        continue;
                    }
                    long[] observed = KRegPITExperiment.histogram(pooled, numBins);
                    double[] stats = KRegPITExperiment.uniformityTests(pooled, observed);
                    KRegPITExperiment.writeTidyGroup(w, models[mi], sizes[si], observed,
                            stats[0], stats[1], stats[2], stats[3], stats[4], pooled.size(), zeroed);

                    System.out.printf(Locale.US, "=> POOLED %s size=%d: m=%d L1=%.3f chiP=%.3g ksP=%.3g zeroed=%d%n",
                            models[mi], sizes[si], pooled.size(), stats[4], stats[1], stats[3], zeroed);
                }
            }
        }
        System.out.printf(Locale.US, "wrote %s and %s  (%.0fs total)%n",
                out, paramsOut, (System.currentTimeMillis() - t0) / 1000.0);
    }

    private static Callable<RepOut> repTask(String dataRoot, String prefix, int rep, int[] sizes,
                                            String[] models, int numSamples, long seed,
                                            AtomicInteger done, int numReps, long t0) {
        return () -> {
            Cell[][] cells = new Cell[models.length][sizes.length];
            double[] kregAlpha = new double[sizes.length];
            double[] kregMu = new double[sizes.length];
            double[] mregMu = new double[sizes.length];
            Arrays.fill(kregAlpha, Double.NaN);
            Arrays.fill(kregMu, Double.NaN);
            Arrays.fill(mregMu, Double.NaN);

            File trainDir = new File(dataRoot + "/rep" + rep + "/run1");
            File testDir = new File(dataRoot + "/rep" + rep + "/run2");
            File train = findTreesFile(trainDir, prefix);
            File test = findTreesFile(testDir, prefix);
            if (train == null || test == null) {
                System.err.printf(Locale.US, "skip rep %d: no %s*.trees in %s%n", rep, prefix,
                        train == null ? trainDir : testDir);
                return new RepOut(rep, cells, kregAlpha, kregMu, mregMu);
            }

            for (int si = 0; si < sizes.length; si++) {
                int size = sizes[si];
                // Load once per size; the three models all train and test on the same trees.
                List<Tree> trainTrees = KRegPITExperiment.loadTrees(train, size);
                List<Tree> testTrees = KRegPITExperiment.loadTrees(test, size);
                for (int mi = 0; mi < models.length; mi++) {
                    ITreeDistribution m = KRegPITExperiment.buildModel(models[mi], trainTrees);
                    // Own, deterministically-seeded RNG per cell (see class doc): makes fitting and the
                    // single-threaded sample reproducible; parallel sampling still varies at MC-noise level.
                    ((AbstractCCD) m).setRandom(new Random(mix(seed, rep, si, mi, 1)));
                    if (m instanceof KRegCCD kreg) { // KRegCCD has fitted (alpha, mu)
                        kregAlpha[si] = kreg.getAlpha();
                        kregMu[si] = kreg.getMu();
                    } else if (m instanceof ccd.model.MRegCCD mreg) { // MRegCCD has fitted mu
                        mregMu[si] = mreg.getMu();
                    }
                    double[] sortedLogP = KRegPITExperiment.sortedSampleLogProbs(m, numSamples);
                    List<Double> uOut = new ArrayList<>();
                    List<Double> vOut = new ArrayList<>();
                    Random jitter = new Random(mix(seed, rep, si, mi, 2));
                    int z = KRegPITExperiment.accumulatePIT(m, testTrees, sortedLogP, jitter, uOut, vOut);
                    cells[mi][si] = new Cell(vOut, z, testTrees.size() - z);
                }
            }
            System.out.printf(Locale.US, "rep %d done (%d/%d, %.0fs)%n",
                    rep, done.incrementAndGet(), numReps, (System.currentTimeMillis() - t0) / 1000.0);
            return new RepOut(rep, cells, kregAlpha, kregMu, mregMu);
        };
    }

    /** Stable mix of the seed and cell indices into a per-cell RNG seed. */
    private static long mix(long... xs) {
        long h = 1125899906842597L;
        for (long x : xs) {
            h = h * 1000003L + x;
        }
        return h;
    }

    /** One single-element holder list per (model, size) pool (avoids a generic-array declaration). */
    @SuppressWarnings("unchecked")
    private static List<List<Double>>[] pools(int numModels, int numSizes) {
        List<List<Double>>[] pools = new List[numModels * numSizes];
        for (int i = 0; i < pools.length; i++) {
            pools[i] = new ArrayList<>(1);
            pools[i].add(new ArrayList<>());
        }
        return pools;
    }

    private static int indexOf(String[] arr, String value) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(value)) return i;
        }
        return -1;
    }

    /**
     * Finds the single {@code <prefix>*.trees} posterior file in a run directory. We glob rather than
     * reconstruct {@code <prefix>-<rep>.trees} because the data pipeline mislabels some reps (e.g.
     * rep100's run trees are named yule-n50-0.trees, not yule-n50-100.trees); each run dir holds
     * exactly one matching trees file. Returns null if the directory has none.
     */
    private static File findTreesFile(File runDir, String prefix) {
        File[] hits = runDir.listFiles((dir, name) -> name.startsWith(prefix) && name.endsWith(".trees"));
        if (hits == null || hits.length == 0) return null;
        if (hits.length > 1) {
            Arrays.sort(hits); // deterministic pick, but flag the ambiguity
            System.err.printf(Locale.US, "warning: %d %s*.trees files in %s; using %s%n",
                    hits.length, prefix, runDir, hits[0].getName());
        }
        return hits[0];
    }

    private static int[] parseInts(String csv) {
        String[] parts = csv.split(",");
        int[] r = new int[parts.length];
        for (int i = 0; i < parts.length; i++) r[i] = Integer.parseInt(parts[i].trim());
        return r;
    }
}
