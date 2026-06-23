package ccd.experiments.regularisation;

import beast.base.evolution.tree.Tree;
import ccd.algorithms.regularisation.KRegCCDParameterOptimiser;
import ccd.algorithms.regularisation.MRegCCDParameterOptimiser;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Fits the regularised-CCD hyperparameters over a sweep of replicates and sample sizes and writes them
 * to a long-format params TSV ({@code model  sampleSize  rep  alpha  mu}) -- one row per (model, size,
 * rep) fit. Both the two-parameter {@link ccd.model.KRegCCD} ({@code alpha} pinned at 0.4, {@code mu}
 * CV-fit) and the one-parameter {@link ccd.model.MRegCCD} ({@code mu} CV-fit, {@code alpha} = NaN) are
 * captured, so the fitted escape rates of the two models can be compared in one figure
 * ({@code doc/plot_params.py}).
 *
 * <p>This is the cheap, parameters-only counterpart of {@link KRegPITSweep}: it does the same
 * deterministic CV fits (no Monte-Carlo PIT sampling), so it reproduces the sweep's fitted parameters
 * exactly while running in minutes rather than hours. Loads each replicate's {@code run1} (training)
 * file once per size; one task per replicate.
 *
 * <p>Usage: {@code RegParamSweep <dataRoot> [prefix=yule-n50] [startRep=1] [endRep=100]
 * [sizes=300,1000,3000] [models=kreg,mreg] [out=params.tsv] [threads=#cores]}
 */
public class RegParamSweep {

    private record Fit(String model, int size, int rep, double alpha, double mu) {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: RegParamSweep <dataRoot> [prefix=yule-n50] [startRep=1] "
                    + "[endRep=100] [sizes=300,1000,3000] [models=kreg,mreg] [out=params.tsv] [threads]");
            return;
        }
        String dataRoot = args[0];
        String prefix = args.length > 1 ? args[1] : "yule-n50";
        int startRep = args.length > 2 ? Integer.parseInt(args[2]) : 1;
        int endRep = args.length > 3 ? Integer.parseInt(args[3]) : 100;
        int[] sizes = parseInts(args.length > 4 ? args[4] : "300,1000,3000");
        String[] models = (args.length > 5 ? args[5] : "kreg,mreg").split(",");
        String out = args.length > 6 ? args[6] : "params.tsv";
        int threads = args.length > 7 ? Integer.parseInt(args[7]) : Runtime.getRuntime().availableProcessors();

        long t0 = System.currentTimeMillis();
        System.out.printf(Locale.US, "param sweep: reps %d..%d, sizes %s, models %s, %d threads%n",
                startRep, endRep, java.util.Arrays.toString(sizes), String.join(",", models), threads);

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<List<Fit>>> futures = new ArrayList<>();
        for (int rep = startRep; rep <= endRep; rep++) {
            final int r = rep;
            futures.add(pool.submit(repTask(dataRoot, prefix, r, sizes, models, t0)));
        }
        pool.shutdown();

        List<Fit> all = new ArrayList<>();
        for (Future<List<Fit>> f : futures) {
            all.addAll(f.get());
        }

        try (PrintWriter w = new PrintWriter(new File(out))) {
            w.println("model\tsampleSize\trep\talpha\tmu");
            for (String model : models) {
                for (int size : sizes) {
                    for (Fit fit : all) {
                        if (fit.model().equals(model) && fit.size() == size) {
                            w.printf(Locale.US, "%s\t%d\t%d\t%.6f\t%.6f%n",
                                    fit.model(), fit.size(), fit.rep(), fit.alpha(), fit.mu());
                        }
                    }
                }
            }
        }
        System.out.printf(Locale.US, "wrote %s (%d fits, %.0fs)%n",
                out, all.size(), (System.currentTimeMillis() - t0) / 1000.0);
    }

    private static Callable<List<Fit>> repTask(String dataRoot, String prefix, int rep, int[] sizes,
                                               String[] models, long t0) {
        return () -> {
            List<Fit> out = new ArrayList<>();
            File train = findTreesFile(new File(dataRoot + "/rep" + rep + "/run1"), prefix);
            if (train == null) {
                System.err.printf(Locale.US, "skip rep %d: no %s*.trees%n", rep, prefix);
                return out;
            }
            for (int size : sizes) {
                List<Tree> trees = KRegPITExperiment.loadTrees(train, size);
                for (String model : models) {
                    switch (model) {
                        case "kreg" -> {
                            KRegCCDParameterOptimiser.Params p = KRegCCDParameterOptimiser.optimise(trees);
                            out.add(new Fit("kreg", size, rep, p.alpha(), p.mu()));
                        }
                        case "mreg" -> {
                            double mu = MRegCCDParameterOptimiser.optimiseMu(trees).mu();
                            out.add(new Fit("mreg", size, rep, Double.NaN, mu));
                        }
                        default -> throw new IllegalArgumentException("unknown model " + model);
                    }
                }
            }
            System.out.printf(Locale.US, "rep %d fit (%.0fs)%n", rep, (System.currentTimeMillis() - t0) / 1000.0);
            return out;
        };
    }

    /** Globs the single {@code <prefix>*.trees} in a run dir (handles rep100's misnamed file). */
    private static File findTreesFile(File dir, String prefix) {
        File[] matches = dir.listFiles((d, name) -> name.startsWith(prefix) && name.endsWith(".trees"));
        return (matches != null && matches.length > 0) ? matches[0] : null;
    }

    private static int[] parseInts(String csv) {
        String[] parts = csv.split(",");
        int[] out = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            out[i] = Integer.parseInt(parts[i].trim());
        }
        return out;
    }
}
