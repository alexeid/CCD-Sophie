package ccd.experiments.regularisation;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import ccd.algorithms.LoadOrStoreTrees;
import ccd.model.AbstractCCD;
import ccd.model.HeightSettingStrategy;
import ccd.model.KRegCCD;
import ccd.model.bitsets.BitSet;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Run GRegCCD on the RSV2 posterior: held-out log-probability of the one-parameter
 * "per-new-split" model, alongside KRegCCD. The exact partition function Z = sum_T weight(T)
 * (weight(T) = product over splits of count-or-eps) is #P-hard at n=129, so we estimate it by
 * importance sampling from a fitted KRegCCD (full support, samplable, close to GRegCCD):
 *   Z(eps) = E_{T~KReg}[ weight(T;eps) / P_KReg(T) ],
 * using the sampler's stamped exact log-probability as the proposal density and reusing the same
 * samples across the eps grid (only weight(T;eps) depends on eps).
 *
 * For a tree, logweight(T;eps) = sumLogCount(T) + novel(T)*log(eps), where sumLogCount sums log of
 * the observed split counts and novel counts the splits never seen in training. Held-out
 * mean logP = mean_test[logweight(T;eps)] - logZ(eps).
 *
 * <p>RESULT (RSV2, 1000/1000, 20k samples): this naive IS estimator FAILS -- effective sample size
 * ~3 / 20000. GRegCCD's count-weighting (observed counts up to ~1000) gives weight(T) an enormous
 * dynamic range, and KRegCCD is too different a proposal, so a handful of near-MAP samples carry all
 * the weight; logZ is then untrustworthy and barely depends on eps. An accurate large-n Z needs
 * either an eps-expansion around the exact observed-core partition function (CCD sum-product over
 * observed clades, deterministic) or annealed importance sampling bridging CCD1 -> GRegCCD; both are
 * substantial. Kept as the infrastructure (observed-split extraction, per-tree stats, eps grid) and
 * a record of why the cheap estimator is insufficient.
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

        // KRegCCD: comparison model + importance-sampling proposal
        KRegCCD kreg = KRegCCD.withOptimisedParameters(train);

        // pre-compute per-test-tree (novel, sumLogCount)
        double[] testNovel = new double[test.size()], testSLC = new double[test.size()];
        for (int i = 0; i < test.size(); i++) { double[] s = stats(test.get(i)); testNovel[i] = s[0]; testSLC[i] = s[1]; }
        double kregMeanLogP = 0;
        for (Tree t : test) kregMeanLogP += kreg.getLogProbabilityOfTree(t);
        kregMeanLogP /= test.size();

        // importance samples from KRegCCD: store (novel, sumLogCount, logQ)
        System.out.printf("drawing %d importance samples from KRegCCD ...%n", nIS);
        double[] isNovel = new double[nIS], isSLC = new double[nIS], isLogQ = new double[nIS];
        for (int s = 0; s < nIS; s++) {
            Tree t = kreg.sampleTree(HeightSettingStrategy.None);
            double[] st = stats(t);
            isNovel[s] = st[0]; isSLC[s] = st[1];
            isLogQ[s] = (Double) t.getRoot().getMetaData(AbstractCCD.LOG_PROB_SUBTREE_KEY);
        }

        System.out.printf("%n%-9s %14s %12s %8s %14s%n", "eps", "GReg logP/tree", "logZ", "ESS", "(KReg logP/tree)");
        double bestEps = 0, bestLogP = Double.NEGATIVE_INFINITY;
        int grid = 25;
        double lo = 1e-4, hi = 0.05;
        for (int g = 0; g < grid; g++) {
            double eps = lo * Math.pow(hi / lo, g / (double) (grid - 1));
            double logEps = Math.log(eps);
            // logZ via log-sum-exp of logweight - logQ
            double[] lr = new double[nIS];
            double max = Double.NEGATIVE_INFINITY;
            for (int s = 0; s < nIS; s++) { lr[s] = isSLC[s] + isNovel[s] * logEps - isLogQ[s]; if (lr[s] > max) max = lr[s]; }
            double sum = 0, sum2 = 0;
            for (int s = 0; s < nIS; s++) { double w = Math.exp(lr[s] - max); sum += w; sum2 += w * w; }
            double logZ = max + Math.log(sum) - Math.log(nIS);
            double ess = sum * sum / sum2; // effective sample size

            double meanLogP = 0;
            for (int i = 0; i < test.size(); i++) meanLogP += testSLC[i] + testNovel[i] * logEps;
            meanLogP = meanLogP / test.size() - logZ;
            if (meanLogP > bestLogP) { bestLogP = meanLogP; bestEps = eps; }
            System.out.printf("%-9.5f %14.3f %12.3f %8.0f %14.3f%n", eps, meanLogP, logZ, ess, kregMeanLogP);
        }
        System.out.printf("%nGRegCCD best held-out logP/tree = %.3f at eps = %.5f%n", bestLogP, bestEps);
        System.out.printf("KRegCCD   held-out logP/tree = %.3f%n", kregMeanLogP);
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
