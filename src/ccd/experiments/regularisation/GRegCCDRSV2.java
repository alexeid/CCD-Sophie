package ccd.experiments.regularisation;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import ccd.algorithms.LoadOrStoreTrees;
import ccd.model.GRegZApprox;
import ccd.model.KRegCCD;
import ccd.model.bitsets.BitSet;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Run GRegCCD on the RSV2 posterior: held-out log-probability of the one-parameter "per-new-split"
 * model, alongside the two-parameter KRegCCD (both full support). For a tree,
 * logweight(T;eps) = sumLogCount(T) + novel(T)*log(eps), where sumLogCount sums log of the observed
 * split counts and novel counts the splits never seen in training; held-out
 * mean logP = mean_test[logweight(T;eps)] - logZ(eps), over an eps grid.
 *
 * <p>The exact partition function Z = sum_T weight(T) is #P-hard at n=129. Naive importance
 * sampling from KRegCCD fails (effective sample size ~3/20000: GRegCCD's count-weighting gives
 * weight(T) an enormous dynamic range). Instead Z is computed by the deterministic observed-clade-DAG
 * approximation {@link GRegZApprox} (fresh remainders priced by g(m), recombinations into two
 * observed clades kept exact), whose error is O(eps^2)-per-clade and so very small at the operating
 * eps.
 */
public class GRegCCDRSV2 {

    static int nTaxa;
    /** observed splits: parent clade -> (canonical child clade -> training count). */
    static final Map<BitSet, Map<BitSet, Integer>> obs = new HashMap<>();

    public static void main(String[] args) throws Exception {
        File treeFile = new File(args[0]);
        double burnin = args.length > 1 ? Double.parseDouble(args[1]) : 0.1;
        int sampleSize = args.length > 2 ? Integer.parseInt(args[2]) : 1000;
        int nIS = args.length > 3 ? Integer.parseInt(args[3]) : 40000;

        List<Tree> all = LoadOrStoreTrees.loadTrees(treeFile, burnin);
        int half = all.size() / 2;
        List<Tree> train = thin(all.subList(0, half), sampleSize);
        List<Tree> test = thin(all.subList(half, all.size()), sampleSize);
        nTaxa = train.get(0).getLeafNodeCount();
        System.out.printf("Loaded %d trees; train=%d, test=%d, n=%d taxa%n", all.size(), train.size(), test.size(), nTaxa);

        for (Tree t : train) recordObserved(t);
        System.out.printf("observed clades=%d, observed splits=%d%n",
                obs.size(), obs.values().stream().mapToInt(Map::size).sum());

        // KRegCCD: comparison model (both are full support)
        KRegCCD kreg = KRegCCD.withOptimisedParameters(train);
        double kregMeanLogP = 0;
        for (Tree t : test) kregMeanLogP += kreg.getLogProbabilityOfTree(t);
        kregMeanLogP /= test.size();

        // pre-compute per-test-tree (novel, sumLogCount)
        double[] testNovel = new double[test.size()], testSLC = new double[test.size()];
        for (int i = 0; i < test.size(); i++) { double[] s = stats(test.get(i)); testNovel[i] = s[0]; testSLC[i] = s[1]; }

        // GRegCCD partition function via the tractable observed-DAG approximation
        GRegZApprox z = GRegZApprox.fromTrees(train);

        System.out.printf("%n%-9s %14s %12s %14s%n", "eps", "GReg logP/tree", "logZ", "(KReg logP/tree)");
        double bestEps = 0, bestLogP = Double.NEGATIVE_INFINITY;
        int grid = 30;
        double lo = 1e-4, hi = 0.5;
        for (int g = 0; g < grid; g++) {
            double eps = lo * Math.pow(hi / lo, g / (double) (grid - 1));
            double logEps = Math.log(eps);
            double logZ = z.logZ(eps);
            double meanLogP = 0;
            for (int i = 0; i < test.size(); i++) meanLogP += testSLC[i] + testNovel[i] * logEps;
            meanLogP = meanLogP / test.size() - logZ;
            if (meanLogP > bestLogP) { bestLogP = meanLogP; bestEps = eps; }
            System.out.printf("%-9.5f %14.3f %12.3f %14.3f%n", eps, meanLogP, logZ, kregMeanLogP);
        }
        System.out.printf("%nGRegCCD best held-out logP/tree = %.3f at eps = %.5f (1 parameter)%n", bestLogP, bestEps);
        System.out.printf("KRegCCD   held-out logP/tree = %.3f (2 parameters, CV-fitted)%n", kregMeanLogP);
    }

    /** (novel split count, sum of log observed-split counts) for a tree. */
    static double[] stats(Tree t) {
        Map<Node, BitSet> bits = new HashMap<>();
        computeBits(t.getRoot(), bits);
        double novel = 0, slc = 0;
        for (Node v : t.getNodesAsArray()) {
            if (v.isLeaf()) continue;
            BitSet pb = bits.get(v);
            BitSet canon = canonChild(pb, bits.get(v.getChildren().get(0)), bits.get(v.getChildren().get(1)));
            Map<BitSet, Integer> m = obs.get(pb);
            Integer c = (m == null) ? null : m.get(canon);
            if (c == null) novel++;
            else slc += Math.log(c);
        }
        return new double[]{novel, slc};
    }

    static void recordObserved(Tree t) {
        Map<Node, BitSet> bits = new HashMap<>();
        computeBits(t.getRoot(), bits);
        for (Node v : t.getNodesAsArray()) {
            if (v.isLeaf()) continue;
            BitSet pb = bits.get(v);
            BitSet canon = canonChild(pb, bits.get(v.getChildren().get(0)), bits.get(v.getChildren().get(1)));
            obs.computeIfAbsent(pb, k -> new HashMap<>()).merge(canon, 1, Integer::sum);
        }
    }

    /** Order-independent split key: the child clade containing the parent's lowest taxon bit. */
    static BitSet canonChild(BitSet parent, BitSet c0, BitSet c1) {
        int lb = parent.nextSetBit(0);
        return c0.get(lb) ? c0 : c1;
    }

    static BitSet computeBits(Node v, Map<Node, BitSet> bits) {
        BitSet b = BitSet.newBitSet(nTaxa);
        if (v.isLeaf()) {
            b.set(v.getNr());
        } else {
            b.or(computeBits(v.getChildren().get(0), bits));
            b.or(computeBits(v.getChildren().get(1), bits));
        }
        bits.put(v, b);
        return b;
    }

    static List<Tree> thin(List<Tree> trees, int k) {
        if (trees.size() <= k) return new ArrayList<>(trees);
        List<Tree> out = new ArrayList<>(k);
        for (int i = 0; i < k; i++) out.add(trees.get((int) ((long) i * trees.size() / k)));
        return out;
    }
}
