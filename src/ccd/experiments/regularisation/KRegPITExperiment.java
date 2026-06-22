package ccd.experiments.regularisation;

import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import ccd.algorithms.LoadOrStoreTrees;
import ccd.model.CCD0;
import ccd.model.CCD1;
import ccd.model.ITreeDistribution;
import ccd.model.KRegCCD;
import ccd.model.RegCCD;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Probability-Integral-Transform (PIT) calibration test for a tree distribution.
 *
 * <p>For a model {@code q} fit to training trees, every held-out test tree {@code T} is given a
 * PIT statistic
 * <pre>
 *     u(T) = P_{S~q}[ q(S) &gt;= q(T) ]
 * </pre>
 * the probability that a tree sampled from the model is at least as probable as {@code T}. We
 * estimate it by Monte Carlo: the fraction of trees drawn from {@code q} whose probability is
 * &ge; {@code q(T)}. This is exactly the credible level at which {@code T} first enters the
 * model's highest-probability credible set &mdash; the same statistic that underlies the
 * credible-set coverage curve. If the model is calibrated and {@code T ~ q}, then
 * {@code u(T) ~ Uniform(0,1)}.</p>
 *
 * <p>The <em>cumulative</em> view of {@code u(T)} over the test set is the coverage curve (empirical
 * CDF versus the diagonal); the <em>density</em> view is the PIT histogram, which localizes where
 * miscalibration lives:</p>
 * <ul>
 *   <li>U-shaped &rarr; model over-confident / under-dispersed (real trees pile up in the tails);</li>
 *   <li>&cap;-shaped &rarr; model under-confident / over-dispersed;</li>
 *   <li>sloped &rarr; systematic probability bias.</li>
 * </ul>
 *
 * <p>Because the tree distribution is discrete, the non-randomized {@code u(T)} is itself not exactly
 * uniform &mdash; it clumps at the achievable mass levels. We therefore also compute the
 * <em>randomized</em> PIT
 * <pre>
 *     V(T) ~ Uniform( u_lo(T), u_hi(T) ),   u_lo = P[q(S) &gt; q(T)],  u_hi = P[q(S) &gt;= q(T)],
 * </pre>
 * which is exactly {@code Uniform(0,1)} under {@code T ~ q} (Dunn &amp; Smyth 1996), and run the
 * formal uniformity tests on it: Pearson's chi-square on the histogram and a one-sample
 * Kolmogorov-Smirnov test against {@code Uniform(0,1)} on the raw values.</p>
 *
 * <p>For a full-support model (KRegCCD) every test tree has positive probability, so the PIT is
 * defined for all of them. For CCD0/CCD1 any test tree containing a novel clade has
 * {@code q(T) = 0}; such trees are reported separately and excluded from the PIT (excluding them
 * makes the reported PIT optimistic for those models &mdash; that excluded fraction is itself a
 * blunt measure of miscalibration).</p>
 *
 * <p>This class runs a single (model, test-set) pair; {@link KRegPITSweep} reuses the static
 * helpers here to pool the PIT over many replicates and sample sizes.</p>
 *
 * <p>Usage: {@code KRegPITExperiment <trainingTrees> <testingTrees>
 * [ccdType=kreg|ccd1|ccd0] [subsample=all] [numBins=20] [numSamples=100000] [seed=42]
 * [outPrefix=pit]}</p>
 */
public class KRegPITExperiment {

    /** Trees sampled from the model to estimate the PIT statistic u(T). */
    static final int DEFAULT_NUM_SAMPLES = 100_000;
    /** Histogram bins over [0,1]. */
    static final int DEFAULT_NUM_BINS = 20;
    /** Seed for the randomized-PIT jitter, so runs are reproducible. */
    static final long DEFAULT_SEED = 42L;

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: KRegPITExperiment <trainingTrees> <testingTrees> "
                    + "[ccdType=kreg|ccd1|ccd0] [subsample=all] [numBins=20] [numSamples=100000] "
                    + "[seed=42] [outPrefix=pit]");
            return;
        }

        File trainingFile = new File(args[0]);
        File testingFile = new File(args[1]);
        String ccdType = args.length > 2 ? args[2].toLowerCase() : "kreg";
        int subsample = args.length > 3 ? Integer.parseInt(args[3]) : -1;
        int numBins = args.length > 4 ? Integer.parseInt(args[4]) : DEFAULT_NUM_BINS;
        int numSamples = args.length > 5 ? Integer.parseInt(args[5]) : DEFAULT_NUM_SAMPLES;
        long seed = args.length > 6 ? Long.parseLong(args[6]) : DEFAULT_SEED;
        String outPrefix = args.length > 7 ? args[7] : "pit";
        Randomizer.setSeed(seed); // make the model sampling reproducible

        // --- load trees (optionally thinned to a fixed subsample, matching the credible-set runs) ---
        List<Tree> trainingTrees = loadTrees(trainingFile, subsample);
        List<Tree> testingTrees = loadTrees(testingFile, subsample);
        System.out.printf(Locale.US, "Loaded %d training trees, %d testing trees; model = %s%n",
                trainingTrees.size(), testingTrees.size(), ccdType);
        int sizeLabel = (subsample > 0) ? subsample : trainingTrees.size();

        // --- fit the model and compute the PIT statistics over the test set ---
        ITreeDistribution model = buildModel(ccdType, trainingTrees);
        double[] sortedLogP = sortedSampleLogProbs(model, numSamples);
        List<Double> uValues = new ArrayList<>();   // non-randomized PIT: u = P[q(S) >= q(T)]
        List<Double> vValues = new ArrayList<>();   // randomized PIT:     V ~ U(u_lo, u_hi)
        int zeroed = accumulatePIT(model, testingTrees, sortedLogP, new Random(seed), uValues, vValues);

        int m = vValues.size();
        if (m == 0) {
            System.out.println("No test trees with positive probability under the model; PIT undefined.");
            return;
        }

        long[] observed = histogram(vValues, numBins);
        double[] stats = uniformityTests(vValues, observed); // {chiStat, chiP, ksStat, ksP}

        printReport(ccdType, m, zeroed, testingTrees.size(), numBins, observed,
                stats[0], stats[1], stats[2], stats[3], stats[4]);
        try (PrintWriter w = new PrintWriter(new File(outPrefix + "_" + ccdType + "_pit_hist.tsv"))) {
            writeTidyHeader(w);
            writeTidyGroup(w, ccdType, sizeLabel, observed,
                    stats[0], stats[1], stats[2], stats[3], stats[4], m, zeroed);
        }
        writeValues(outPrefix, ccdType, uValues, vValues);
        System.out.printf(Locale.US, "Wrote %s_%s_pit_hist.tsv and %s_%s_pit_values.tsv%n",
                outPrefix, ccdType, outPrefix, ccdType);
    }

    // ----------------------------------------------------------------------------------------
    // Shared helpers (also used by KRegPITSweep)
    // ----------------------------------------------------------------------------------------

    /** Loads trees with no burn-in, thinned to {@code subsample} if positive, else all of them. */
    static List<Tree> loadTrees(File file, int subsample) throws IOException {
        return (subsample > 0)
                ? LoadOrStoreTrees.loadTrees(file, 0, subsample)
                : LoadOrStoreTrees.loadTrees(file, 0);
    }

    static ITreeDistribution buildModel(String ccdType, List<Tree> trainingTrees) {
        switch (ccdType) {
            case "ccd0":
                return new CCD0(trainingTrees, 0);
            case "ccd1":
                return new CCD1(trainingTrees, 0);
            case "regccd":
                return new RegCCD(trainingTrees, 0, KRegCCD.DEFAULT_ALPHA);
            case "kreg":
                return KRegCCD.withOptimisedParameters(trainingTrees);
            case "mreg": {
                // one-parameter per-new-split model; tail OFF so the (self-consistent) sampler used by
                // the PIT and the scorer are the same distribution (see MRegCCD sampling).
                double mu = ccd.algorithms.regularisation.MRegCCDParameterOptimiser
                        .optimiseMu(trainingTrees).mu();
                return new ccd.model.MRegCCD(trainingTrees, 0.0, mu,
                        ccd.model.MRegCCD.DEFAULT_RESERVE_DEPTH, false);
            }
            default:
                throw new IllegalArgumentException("Unknown ccdType '" + ccdType + "' (expected kreg, mreg, regccd, ccd1 or ccd0)");
        }
    }

    /** Log-probabilities of {@code numSamples} trees drawn from the model, sorted ascending. */
    static double[] sortedSampleLogProbs(ITreeDistribution model, int numSamples) {
        double[] sampleLogP = new double[numSamples];
        for (int i = 0; i < numSamples; i++) {
            sampleLogP[i] = model.sampleTreeLogProbability();
        }
        Arrays.sort(sampleLogP);
        return sampleLogP;
    }

    /**
     * Appends the PIT statistics of each test tree to {@code uOut} (non-randomized) and
     * {@code vOut} (randomized), against the sorted model-sample log-probabilities. Test trees
     * with zero probability (outside the model's support) are skipped; the count of those is
     * returned.
     */
    static int accumulatePIT(ITreeDistribution model, List<Tree> testTrees, double[] sortedLogP,
                             Random rng, List<Double> uOut, List<Double> vOut) {
        int zeroed = 0;
        for (Tree tree : testTrees) {
            double logp = model.getLogProbabilityOfTree(tree);
            if (!Double.isFinite(logp)) { // q(T) == 0: tree lies outside the model's support
                zeroed++;
                continue;
            }
            double uHi = fractionAtLeast(sortedLogP, logp); // P[q(S) >= q(T)]
            double uLo = fractionGreater(sortedLogP, logp); // P[q(S) >  q(T)]
            uOut.add(uHi);
            vOut.add(uLo + rng.nextDouble() * (uHi - uLo));
        }
        return zeroed;
    }

    /** Bins values in [0,1) into {@code numBins} equal-width bins. */
    static long[] histogram(List<Double> values, int numBins) {
        long[] observed = new long[numBins];
        for (double v : values) {
            int b = (int) Math.floor(v * numBins);
            if (b < 0) b = 0;
            if (b >= numBins) b = numBins - 1;
            observed[b]++;
        }
        return observed;
    }

    /**
     * Tests the randomized PIT for uniformity and measures its deviation from uniform: Pearson
     * chi-square on the histogram, a one-sample Kolmogorov-Smirnov test against Uniform(0,1) on the
     * raw values, and the L1 calibration error.
     *
     * <p>The L1 error {@code sum_i |observed_i/m - 1/B|} is the L1 distance of the empirical
     * bin-probability vector from uniform (the density analogue of the coverage curve's "L1 norm").
     * Unlike the p-values it is an effect size: its H0 noise floor shrinks like 1/sqrt(m), so it
     * stays meaningful when pooling many replicates makes the p-values reject trivially.</p>
     *
     * @return {@code {chiSquareStat, chiSquareP, ksStat, ksP, l1}}
     */
    static double[] uniformityTests(List<Double> values, long[] observed) {
        int numBins = observed.length;
        int m = values.size();
        double[] expected = new double[numBins];
        Arrays.fill(expected, (double) m / numBins);

        ChiSquareTest chi = new ChiSquareTest();
        double chiStat = chi.chiSquare(expected, observed);
        double chiP = chi.chiSquareTest(expected, observed);

        double[] v = values.stream().mapToDouble(Double::doubleValue).toArray();
        KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
        UniformRealDistribution unif = new UniformRealDistribution(0, 1);
        double ksStat = ks.kolmogorovSmirnovStatistic(unif, v);
        double ksP = ks.kolmogorovSmirnovTest(unif, v, false); // asymptotic p-value (fast for large m)

        double l1 = 0.0;
        for (long c : observed) l1 += Math.abs(c / (double) m - 1.0 / numBins);

        return new double[]{chiStat, chiP, ksStat, ksP, l1};
    }

    /** Fraction of the ascending-sorted values that are &ge; x. */
    static double fractionAtLeast(double[] sortedAsc, double x) {
        return (sortedAsc.length - lowerBound(sortedAsc, x)) / (double) sortedAsc.length;
    }

    /** Fraction of the ascending-sorted values that are &gt; x. */
    static double fractionGreater(double[] sortedAsc, double x) {
        return (sortedAsc.length - upperBound(sortedAsc, x)) / (double) sortedAsc.length;
    }

    /** First index i with a[i] &ge; x (= number of elements strictly less than x). */
    private static int lowerBound(double[] a, double x) {
        int lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] < x) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /** First index i with a[i] &gt; x (= number of elements &le; x). */
    private static int upperBound(double[] a, double x) {
        int lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] <= x) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    // ----------------------------------------------------------------------------------------
    // Output
    // ----------------------------------------------------------------------------------------

    /** Header for the tidy ("long") histogram TSV consumed by doc/plot_pit.py. */
    static void writeTidyHeader(PrintWriter w) {
        w.println("model\tsampleSize\tbinLeft\tbinRight\tcount\texpected\tchiSq\tchiP\tksD\tksP\tpitL1\tnScored\tnZeroed");
    }

    /**
     * Writes one (model, sampleSize) histogram group as tidy rows &mdash; one row per bin, with the
     * model label, sample size and uniformity results repeated &mdash; so several groups can be
     * concatenated into one grid figure without a manifest.
     */
    static void writeTidyGroup(PrintWriter w, String model, int sizeLabel, long[] observed,
                               double chiStat, double chiP, double ksStat, double ksP, double pitL1,
                               int nScored, int nZeroed) {
        int numBins = observed.length;
        double expected = (double) nScored / numBins;
        for (int i = 0; i < numBins; i++) {
            w.printf(Locale.US, "%s\t%d\t%.6f\t%.6f\t%d\t%.6f\t%.6f\t%.6g\t%.6f\t%.6g\t%.6f\t%d\t%d%n",
                    model, sizeLabel, (double) i / numBins, (double) (i + 1) / numBins,
                    observed[i], expected, chiStat, chiP, ksStat, ksP, pitL1, nScored, nZeroed);
        }
    }

    private static void writeValues(String outPrefix, String ccdType,
                                    List<Double> uValues, List<Double> vValues) throws IOException {
        try (PrintWriter w = new PrintWriter(new File(outPrefix + "_" + ccdType + "_pit_values.tsv"))) {
            w.println("u_nonrandomized\tv_randomized");
            for (int i = 0; i < vValues.size(); i++) {
                w.printf(Locale.US, "%.6f\t%.6f%n", uValues.get(i), vValues.get(i));
            }
        }
    }

    private static void printReport(String ccdType, int m, int zeroed, int total, int numBins,
                                    long[] observed, double chiStat, double chiP,
                                    double ksStat, double ksP, double pitL1) {
        System.out.println();
        System.out.println("=== PIT calibration test: " + ccdType + " ===");
        System.out.printf(Locale.US, "test trees scored: %d of %d", m, total);
        if (zeroed > 0) {
            System.out.printf(Locale.US, "  (%d = %.1f%% had zero probability and were excluded)",
                    zeroed, 100.0 * zeroed / total);
        }
        System.out.println();

        long maxCount = 0;
        for (long c : observed) maxCount = Math.max(maxCount, c);
        System.out.printf(Locale.US, "%nPIT histogram (%d bins, expected %.1f per bin):%n", numBins, (double) m / numBins);
        for (int i = 0; i < numBins; i++) {
            int barLen = maxCount == 0 ? 0 : (int) Math.round(40.0 * observed[i] / maxCount);
            StringBuilder bar = new StringBuilder();
            for (int b = 0; b < barLen; b++) bar.append('#');
            System.out.printf(Locale.US, "[%.2f,%.2f) %6d %s%n",
                    (double) i / numBins, (double) (i + 1) / numBins, observed[i], bar);
        }

        System.out.printf(Locale.US, "%nPIT L1 calibration error (vs uniform): %.4f%n", pitL1);
        System.out.printf(Locale.US, "Pearson chi-square (uniformity): X^2 = %.3f, df = %d, p = %.4g%n",
                chiStat, numBins - 1, chiP);
        System.out.printf(Locale.US, "Kolmogorov-Smirnov (vs Uniform):  D = %.4f, p = %.4g%n", ksStat, ksP);
        System.out.println(chiP < 0.05 || ksP < 0.05
                ? "=> reject uniformity at 0.05: the PIT is not flat, model is miscalibrated."
                : "=> no significant departure from uniformity: PIT consistent with calibration.");
    }
}
