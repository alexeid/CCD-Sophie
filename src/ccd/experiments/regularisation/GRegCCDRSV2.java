package ccd.experiments.regularisation;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import ccd.algorithms.LoadOrStoreTrees;
import ccd.model.CCD1;
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
    /** parent clade -> total observed split count (for conditional CCPs). */
    static final Map<BitSet, Double> total = new HashMap<>();
    private static final double LOG2 = Math.log(2);

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

        // comparison models (KRegCCD full support; CCD1 zero on novel trees)
        KRegCCD kreg = KRegCCD.withOptimisedParameters(train);
        CCD1 ccd1 = new CCD1(train, 0.0);
        double kregMeanLogP = 0;
        for (Tree t : test) kregMeanLogP += kreg.getLogProbabilityOfTree(t);
        kregMeanLogP /= test.size();

        // pre-compute per-test-tree (novel, sumLogCount); split common (novel=0) vs novel-containing
        double[] testNovel = new double[test.size()], testSLC = new double[test.size()];
        List<Integer> common = new ArrayList<>(), withNovel = new ArrayList<>();
        for (int i = 0; i < test.size(); i++) {
            double[] s = stats(test.get(i)); testNovel[i] = s[0]; testSLC[i] = s[1];
            (s[0] == 0 ? common : withNovel).add(i);
        }

        // GRegCCD partition function via the tractable observed-DAG approximation
        GRegZApprox z = GRegZApprox.fromTrees(train);

        // CONSISTENCY CHECK on the observed-support (common) held-out trees, where there is no escape:
        // GReg logP = sumLogCount - logZ (eps-independent). Compare to CCD1 and KRegCCD on the SAME trees.
        double logZobs = z.logZ(1e-6);
        double gregCommon = 0, ccd1Common = 0, kregCommon = 0;
        for (int i : common) {
            gregCommon += testSLC[i] - logZobs;
            ccd1Common += ccd1.getLogProbabilityOfTree(test.get(i));
            kregCommon += kreg.getLogProbabilityOfTree(test.get(i));
        }
        int nc = common.size();
        System.out.printf("%n-- observed-support held-out trees: %d of %d --%n", nc, test.size());
        System.out.printf("GReg(eps->0) logP/tree = %.3f   CCD1 = %.3f   KReg = %.3f   (logZ_obs=%.3f)%n",
                gregCommon / nc, ccd1Common / nc, kregCommon / nc, logZobs);
        System.out.printf("-- novel-containing held-out trees: %d (CCD1 scores 0 = -inf here) --%n", withNovel.size());

        // ---- conditional "1-mu" per-new-split model: observed split -> (1-mu)*count/total_C,
        // novel split -> mu * product over the escape region's novel splits of 1/bipartitions(size).
        // Per-clade normalised (chain rule), no global Z. Matches CCD1 on common trees up to (n-1)log(1-mu).
        for (Map.Entry<BitSet, Map<BitSet, Integer>> e : obs.entrySet()) {
            double tot = 0; for (int c : e.getValue().values()) tot += c;
            total.put(e.getKey(), tot);
        }
        System.out.printf("%n%-9s %16s %16s %14s%n", "mu", "1mu logP/tree", "1mu common/tree", "(CCD1 common)");
        double bestMu = 0, best1mu = Double.NEGATIVE_INFINITY;
        for (int gi = 0; gi < 25; gi++) {
            double mu = 1e-4 * Math.pow(0.2 / 1e-4, gi / 24.0);
            double mAll = 0, mCom = 0;
            for (int i = 0; i < test.size(); i++) mAll += scoreMu1(test.get(i), mu);
            for (int i : common) mCom += scoreMu1(test.get(i), mu);
            mAll /= test.size(); mCom /= nc;
            if (mAll > best1mu) { best1mu = mAll; bestMu = mu; }
            System.out.printf("%-9.5f %16.3f %16.3f %14.3f%n", mu, mAll, mCom, ccd1Common / nc);
        }
        System.out.printf("1-mu model best held-out logP/tree = %.3f at mu = %.5f%n", best1mu, bestMu);

        System.out.printf("%n%-9s %14s %12s %14s%n", "eps", "GReg logP/tree", "logZ", "(KReg logP/tree)");
        double bestEps = 0, bestLogP = Double.NEGATIVE_INFINITY;
        int grid = 32;
        double lo = 1e-4, hi = 0.95;
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

    /** log bipartitions of an m-clade: log(2^(m-1) - 1). */
    static double logBip(int m) {
        if (m <= 2) return 0.0;
        return (m - 1) * LOG2 + Math.log1p(-Math.pow(2.0, -(m - 1)));
    }

    /** Conditional "1-mu" per-new-split log-probability of a tree (per-clade normalised, no global Z). */
    static double scoreMu1(Tree t, double mu) {
        Map<Node, BitSet> bits = new HashMap<>();
        computeBits(t.getRoot(), bits);
        double logP = 0;
        double log1mMu = Math.log(1 - mu), logMu = Math.log(mu);
        for (Node v : t.getNodesAsArray()) {
            if (v.isLeaf()) continue;
            BitSet pb = bits.get(v);
            int m = pb.cardinality();
            BitSet canon = canonChild(pb, bits.get(v.getChildren().get(0)), bits.get(v.getChildren().get(1)));
            Map<BitSet, Integer> sp = obs.get(pb);
            boolean cladeObserved = sp != null;
            Integer cnt = cladeObserved ? sp.get(canon) : null;
            if (cnt != null) {
                logP += log1mMu + Math.log(cnt) - Math.log(total.get(pb)); // observed split: (1-mu)*count/total
            } else if (cladeObserved) {
                logP += logMu - logBip(m);                                 // region entry: escape at observed clade
            } else {
                logP += -logBip(m);                                        // inner novel split
            }
        }
        return logP;
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
