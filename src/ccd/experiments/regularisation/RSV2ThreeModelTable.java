package ccd.experiments.regularisation;

import beast.base.evolution.tree.Tree;
import ccd.algorithms.LoadOrStoreTrees;
import ccd.algorithms.regularisation.KRegCCDParameterOptimiser;
import ccd.algorithms.regularisation.NNIHeldOutComparison.FoldAssignment;
import ccd.model.CCD1;
import ccd.model.KRegCCD;
import ccd.model.RegCCD;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Held-out comparison of three topology distributions on one posterior tree set, for the talk
 * table: plain CCD1 (unsmoothed split frequencies), regCCD (split-expanded + additive-alpha
 * smoothed), and KRegCCD (full support, escape mass mu).
 *
 * <p>Protocol: train all three models on {@code sampleSize} trees from the FIRST half of the run,
 * score them on {@code sampleSize} trees from the SECOND half (independent of the training half, so
 * the held-out split is clean). KRegCCD's escape parameter mu is fitted by leave-k-out
 * cross-validation on the FIRST half only ({@link KRegCCD#withOptimisedMu}, alpha pinned at 0.4), so
 * the test half never influences mu. CCD1 has no parameter; regCCD uses the same alpha = 0.4 backbone.
 *
 * <p>Coverage is nested CCD1 subset regCCD subset KRegCCD (KRegCCD is always full support), so the
 * common set on which the three log-probabilities are comparable is exactly the CCD1-covered set.
 *
 * <p>Usage: RSV2ThreeModelTable &lt;trees-file&gt; [burnin=0.1] [sampleSize=1000]
 */
public class RSV2ThreeModelTable {

    private static final double ALPHA = 0.4;

    /** Even-thin a list down to {@code k} entries (keep the spread, not a contiguous block). */
    private static List<Tree> thin(List<Tree> src, int k) {
        if (src.size() <= k) return new ArrayList<>(src);
        List<Tree> out = new ArrayList<>(k);
        double step = (double) src.size() / k;
        for (int i = 0; i < k; i++) out.add(src.get((int) (i * step)));
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: RSV2ThreeModelTable <trees-file> [burnin=0.1] [sampleSize=1000]");
            System.exit(1);
        }
        File treeFile = new File(args[0]);
        double burnin = args.length > 1 ? Double.parseDouble(args[1]) : 0.1;
        int sampleSize = args.length > 2 ? Integer.parseInt(args[2]) : 1000;

        List<Tree> all = LoadOrStoreTrees.loadTrees(treeFile, burnin);
        int half = all.size() / 2;
        List<Tree> train = thin(all.subList(0, half), sampleSize);                 // first half
        List<Tree> test = thin(all.subList(half, all.size()), sampleSize);          // second half
        System.out.printf("Loaded %d post-burn-in trees; train=%d (1st half), test=%d (2nd half), alpha=%.2f%n%n",
                all.size(), train.size(), test.size(), ALPHA);

        CCD1 ccd1 = new CCD1(train, 0.0);                 // plain, unsmoothed
        RegCCD reg = new RegCCD(train, 0.0, ALPHA);       // split-expanded + alpha
        KRegCCD kreg = KRegCCD.withOptimisedMu(train);    // mu by leave-k-out CV on the train half
        System.out.printf("KRegCCD fitted mu = %.5f (alpha = %.2f)%n%n", kreg.getMu(), kreg.getAlpha());

        int nTest = test.size(), ccd1Cov = 0, regCov = 0;   // KRegCCD covers everything
        int common = 0;                                     // CCD1-covered trees (the comparable common set)
        double sCcd1 = 0, sReg = 0, sKreg = 0;              // sum log-prob over the common set
        int regSet = 0;                                     // regCCD-covered trees
        double rReg = 0, rKreg = 0;                         // sum log-prob over the regCCD-covered set

        for (Tree t : test) {
            double lc = ccd1.getLogProbabilityOfTree(t);
            double lr = reg.getLogProbabilityOfTree(t);
            double lk = kreg.getLogProbabilityOfTree(t);
            boolean cCov = Double.isFinite(lc);   // covered = finite log-prob (not log 0)
            boolean rCov = Double.isFinite(lr);
            if (cCov) ccd1Cov++;
            if (rCov) regCov++;
            if (cCov) { common++; sCcd1 += lc; sReg += lr; sKreg += lk; }
            if (rCov) { regSet++; rReg += lr; rKreg += lk; }
        }

        System.out.printf("=== RSV2 held-out (train 1st half, test 2nd half; %d test trees) ===%n", nTest);
        System.out.printf("%-9s %9s %26s %22s%n",
                "model", "coverage", "mean logP (common/CCD1)", "mean logP (regCCD set)");
        System.out.printf("%-9s %8.1f%% %26.2f %22s%n",
                "CCD1", 100.0 * ccd1Cov / nTest, sCcd1 / common, "n/a");
        System.out.printf("%-9s %8.1f%% %26.2f %22.2f%n",
                "regCCD", 100.0 * regCov / nTest, sReg / common, rReg / regSet);
        System.out.printf("%-9s %8.1f%% %26.2f %22.2f%n",
                "KRegCCD", 100.0, sKreg / common, rKreg / regSet);
        System.out.printf("%ncommon set (CCD1-covered) = %d trees; regCCD-covered = %d trees%n",
                common, regSet);

        // --- mean logP / tree over ALL held-out trees (the basis the mu-calibration slide uses) ---
        double allKreg = 0, allCcd1 = 0, allReg = 0;
        boolean ccd1Fin = true, regFin = true;
        for (Tree t : test) {
            double lc = ccd1.getLogProbabilityOfTree(t);
            double lr = reg.getLogProbabilityOfTree(t);
            allKreg += kreg.getLogProbabilityOfTree(t);
            if (Double.isFinite(lc)) allCcd1 += lc; else ccd1Fin = false;
            if (Double.isFinite(lr)) allReg += lr; else regFin = false;
        }
        System.out.printf("%nmean logP/tree over ALL %d held-out trees:%n", nTest);
        System.out.printf("  CCD1    = %s%n", ccd1Fin ? String.format("%.2f", allCcd1 / nTest) : "-inf (assigns 0 to some held-out trees)");
        System.out.printf("  regCCD  = %s%n", regFin ? String.format("%.2f", allReg / nTest) : "-inf (assigns 0 to some held-out trees)");
        System.out.printf("  KRegCCD = %.2f%n", allKreg / nTest);

        // --- mu-calibration sweep on the same split, all held-out trees ---
        // One backbone (mu-independent); P(novel) and held-out logP are both re-evaluated per mu on it
        // (getNovelCladeProbability(mu) / getLogProbabilityOfTree(t, mu)), so no per-mu rebuild.
        int novel = 0;
        for (Tree t : test) if (kreg.containsNovelClade(t)) novel++;
        double emp = (double) novel / nTest;
        KRegCCD sweep = new KRegCCD(train, 0.0, 0.005, ALPHA, 2, KRegCCD.TailMode.NONE);

        int gridN = 40;
        double muLo = 0.001, muHi = 0.01;
        File tsv = new File("doc/rsv2-musweep.tsv");
        try (PrintWriter w = new PrintWriter(new FileWriter(tsv))) {
            // cvLogP = leave-k-out CV held-out mean logP/tree on the TRAINING half only (the criterion
            // the optimiser uses, profiled here over the same mu grid); meanLogP is the second-half
            // held-out density. Both peak near the mu where the model's P(novel) matches the empirical rate.
            w.println("mu\tpnovel\tempirical\tmeanLogP\tcvLogP");
            System.out.printf("%nempirical novel-clade rate among held-out trees = %.3f%n", emp);
            System.out.printf("%-9s %12s %16s %16s%n", "mu", "P(novel)", "meanLogP/tree", "cvLogP/tree");
            double bestMu = 0, bestLogP = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < gridN; i++) {
                double mu = muLo * Math.pow(muHi / muLo, i / (double) (gridN - 1));   // log-spaced
                double pnov = sweep.getNovelCladeProbability(mu);
                double s = 0;
                for (Tree t : test) s += sweep.getLogProbabilityOfTree(t, mu);
                double meanLogP = s / nTest;
                double cvLogP = KRegCCDParameterOptimiser.crossValidatedLogProb(
                        train, KRegCCDParameterOptimiser.DEFAULT_FOLDS, FoldAssignment.CONTIGUOUS, ALPHA, mu)
                        / train.size();
                if (meanLogP > bestLogP) { bestLogP = meanLogP; bestMu = mu; }
                w.printf("%.6f\t%.6f\t%.6f\t%.6f\t%.6f%n", mu, pnov, emp, meanLogP, cvLogP);
                System.out.printf("%-9.5f %12.3f %16.3f %16.3f%n", mu, pnov, meanLogP, cvLogP);
            }
            System.out.printf("%nheld-out logP optimum at mu = %.5f (mean logP = %.3f); CV-fitted mu = %.5f%n",
                    bestMu, bestLogP, kreg.getMu());
            System.out.printf("wrote %s%n", tsv.getPath());
        }

        // --- normalisation: held-out inflation from scoring all orders off a depth-k epsilon estimate ---
        // TailMode.NONE over-reserves (omits the reserve tail) and so inflates held-out logP; per held-out
        // tree the inflation is logP(NONE) - logP(corrected). BOUND uses the geometric tail bound (upper
        // bound on the inflation); SAMPLED uses a Knuth tail estimate (the near-exact "true" inflation).
        double[] muInfl = {0.005, 0.01, 0.05, 0.10};
        File tsv2 = new File("doc/rsv2-inflation.tsv");
        try (PrintWriter w = new PrintWriter(new FileWriter(tsv2))) {
            w.println("mu\ttrueInflation\tupperBound");
            System.out.printf("%n=== normalisation: mean held-out inflation (nat/tree) ===%n");
            System.out.printf("%-8s %18s %14s%n", "mu", "true (sampled)", "upper bound");
            for (double mu : muInfl) {
                KRegCCD none = new KRegCCD(train, 0.0, mu, ALPHA, 2, KRegCCD.TailMode.NONE);
                KRegCCD bnd = new KRegCCD(train, 0.0, mu, ALPHA, 2, KRegCCD.TailMode.BOUND);
                KRegCCD smp = new KRegCCD(train, 0.0, mu, ALPHA, 2, KRegCCD.TailMode.SAMPLED);
                double dB = 0, dS = 0;
                for (Tree t : test) {
                    double ln = none.getLogProbabilityOfTree(t);
                    dB += ln - bnd.getLogProbabilityOfTree(t);
                    dS += ln - smp.getLogProbabilityOfTree(t);
                }
                double trueInfl = dS / nTest, bound = dB / nTest;
                w.printf("%.4f\t%.6f\t%.6f%n", mu, trueInfl, bound);
                System.out.printf("%-8.4f %18.6f %14.6f%n", mu, trueInfl, bound);
            }
            System.out.printf("wrote %s%n", tsv2.getPath());
        }
    }
}
