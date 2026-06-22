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

        // ---- conditional 1-mu with the PROPER eps-reserve escape (per-new-split) ----
        allObs = new ArrayList<>(obs.keySet());
        for (int i = 0; i < nTaxa; i++) { BitSet s = BitSet.newBitSet(nTaxa); s.set(i); allObs.add(s); }
        int nNov = withNovel.size();
        System.out.printf("%n%-9s %14s %12s %12s %10s%n",
                "mu", "reserve all", "reserve com", "reserve nov", "fbRegHit");
        double bestRMu = 0, bestR = Double.NEGATIVE_INFINITY, bestRNov = 0, bestRCom = 0;
        int[] bestHist = null; int bestFbClades = 0, bestFbHits = 0, bestTops = 0;
        for (int gi = 0; gi < 22; gi++) {
            double mu = 1e-4 * Math.pow(0.2 / 1e-4, gi / 21.0);
            epsCache.clear(); fallbackClades = 0; fallbackSet.clear();
            java.util.Arrays.fill(regionHist, 0); fallbackRegionHits = 0; totalRegionTops = 0;
            double rAll = 0, rCom = 0, rNov = 0;
            for (int i = 0; i < test.size(); i++) rAll += scoreReserve(test.get(i), mu);
            for (int i : common) rCom += scoreReserve(test.get(i), mu);
            for (int i : withNovel) rNov += scoreReserve(test.get(i), mu);
            rAll /= test.size(); rCom /= nc; rNov /= Math.max(1, nNov);
            if (rAll > bestR) {
                bestR = rAll; bestRMu = mu; bestRNov = rNov; bestRCom = rCom;
                bestHist = regionHist.clone(); bestFbClades = fallbackClades;
                bestFbHits = fallbackRegionHits; bestTops = totalRegionTops;
            }
            System.out.printf("%-9.5f %14.3f %12.3f %12.3f %7d/%-5d%n",
                    mu, rAll, rCom, rNov, fallbackRegionHits, totalRegionTops);
        }
        // KRegCCD common/novel breakdown for an apples-to-apples gap decomposition
        double kregCom = 0, kregNov = 0;
        for (int i : common) kregCom += kreg.getLogProbabilityOfTree(test.get(i));
        for (int i : withNovel) kregNov += kreg.getLogProbabilityOfTree(test.get(i));
        kregCom /= nc; kregNov /= Math.max(1, nNov);
        System.out.printf("reserve 1-mu model best held-out logP/tree = %.3f at mu = %.5f%n", bestR, bestRMu);
        System.out.printf("  reserve  : all %.3f  common %.3f  novel %.3f  (%d common, %d novel)%n",
                bestR, bestRCom, bestRNov, nc, nNov);
        System.out.printf("  KRegCCD  : all %.3f  common %.3f  novel %.3f%n", kregMeanLogP, kregCom, kregNov);
        System.out.printf("  CCD1-com %.3f%n", ccd1Common / nc);
        System.out.printf("  at best mu: %d fallback clades (M2=M3=0); region tops on held-out: %d total, "
                + "%d hit a fallback clade (%.1f%%)%n",
                bestFbClades, bestTops, bestFbHits, 100.0 * bestFbHits / Math.max(1, bestTops));
        System.out.print("  region boundary-size histogram (m -> count): ");
        for (int m = 2; m < bestHist.length; m++) if (bestHist[m] > 0) System.out.printf("%d:%d  ", m, bestHist[m]);
        System.out.println();

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

    // ---- proper eps-reserve escape (per-new-split, conditional) ----
    static List<BitSet> allObs;                                   // observed clades + singletons
    static final Map<BitSet, List<BitSet>> subCache = new HashMap<>();
    static final Map<BitSet, Double> epsCache = new HashMap<>();  // per current mu
    static int fallbackClades = 0;
    // ---- diagnostics (reset per mu) ----
    static final java.util.Set<BitSet> fallbackSet = new java.util.HashSet<>(); // clades that used the eps fallback
    static int[] regionHist = new int[64];   // boundary-size histogram of region tops scored on held-out trees
    static int fallbackRegionHits = 0;       // region tops whose clade used the eps fallback
    static int totalRegionTops = 0;          // region tops scored (over the all-test pass)

    static boolean isObs(BitSet x) {
        return x.cardinality() == 1 || obs.containsKey(x);
    }

    /** Observed subclades (incl. singletons) strictly inside C. */
    static List<BitSet> subclades(BitSet C) {
        return subCache.computeIfAbsent(C, c -> {
            List<BitSet> out = new ArrayList<>();
            for (BitSet x : allObs) {
                if (x.cardinality() < c.cardinality()) {
                    BitSet t = BitSet.newBitSet(x); t.andNot(c);
                    if (t.isEmpty()) out.add(x); // x subset of c
                }
            }
            return out;
        });
    }

    /* ---- generalized per-clade reserve (mirrors KRegCCD.computeReg) ----
     * M_m(C) = number of all-novel resolutions of C with a boundary of m observed subclades (FLAT:
     * each distinct resolution counted once). The escape mass at C is
     *   R(C; eps) = sum_{m>=2} M_m eps^(m-1)   (m boundary parts -> m-1 new splits; one eps each),
     * and eps(C) solves R = mu over the computed orders m = 2..RESERVE_DEPTH; orders beyond that are
     * a geometric tail. A clade is reservable iff some M_m > 0 (cherries / no-escape clades are not). */
    static int RESERVE_DEPTH =
            Integer.parseInt(System.getProperty("greg.reserveDepth", "5")); // max boundary size enumerated
    static final long OPS_BUDGET = Long.parseLong(System.getProperty("greg.enumOps", "20000000"));
    static long enumOps;
    static final class Budget extends RuntimeException { Budget() { super(null, null, false, false); } }
    static final Budget BUDGET = new Budget();
    static final Map<BitSet, int[]> countsCache = new HashMap<>(); // clade -> M_m counts (mu-independent)

    /** M_m counts (index m = boundary size, 2..min(|C|,RESERVE_DEPTH)); cached, mu-independent. */
    static int[] countsFor(BitSet C) {
        int[] cached = countsCache.get(C);
        if (cached != null) return cached;
        int card = C.cardinality();
        int[] n = new int[Math.min(card, RESERVE_DEPTH) + 1];
        if (card >= 2) {
            List<BitSet> subs = subclades(C);
            enumOps = 0;
            for (int m = 2; m < n.length; m++) {
                try { n[m] = countBoundaries(C, subs, m); }
                catch (Budget b) { break; }   // deeper orders omitted (negligible, like the tail)
            }
        }
        countsCache.put(C, n);
        return n;
    }

    static boolean reservable(BitSet C) {
        for (int v : countsFor(C)) if (v > 0) return true;
        return false;
    }

    /** Escape root eps solving sum_{m>=2} M_m eps^(m-1) = mu (monotone bisection); crude fallback if
     *  C has no escape route up to RESERVE_DEPTH (should not happen for an actually-escaped region top). */
    static double epsFor(BitSet C, double mu) {
        Double cached = epsCache.get(C);
        if (cached != null) return cached;
        double eps;
        if (!reservable(C)) { eps = mu; fallbackClades++; fallbackSet.add(C); }
        else eps = solveEps(countsFor(C), mu);
        epsCache.put(C, eps);
        return eps;
    }

    /** Omitted-tail escape mass beyond the computed orders: geometric bound from the top two orders
     *  (mirrors KRegCCD's TailMode.BOUND). Clamped to [0, mu]. */
    static double tailFor(BitSet C, double mu) {
        int[] n = countsFor(C);
        int last = n.length - 1;
        if (last < 3) return 0.0;
        int nLast = n[last], nPrev = n[last - 1];
        if (nLast <= 0 || nPrev <= 0) return 0.0;
        double eps = epsFor(C, mu);
        double rho = ((double) nLast / nPrev) * eps;   // ratio of successive order masses
        if (rho <= 0 || rho >= 1) return 0.0;
        return Math.min(nLast * Math.pow(eps, last - 1) * rho / (1 - rho), mu);
    }

    static double solveEps(int[] n, double mu) {
        double lo = 0, hi = 1;
        while (evalReserve(n, hi) < mu) hi *= 2;
        for (int it = 0; it < 100; it++) {
            double mid = 0.5 * (lo + hi);
            if (evalReserve(n, mid) < mu) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    /** sum_{m>=2} n[m] x^(m-1). */
    static double evalReserve(int[] n, double x) {
        double s = 0;
        for (int m = 2; m < n.length; m++) if (n[m] > 0) s += n[m] * Math.pow(x, m - 1);
        return s;
    }

    /** Count m-part boundaries of C into observed subclades, each weighted by its all-novel pathcount (FLAT). */
    static int countBoundaries(BitSet C, List<BitSet> subs, int m) {
        return enumerateBoundaries(C, subs, m, 0, BitSet.newBitSet(nTaxa), new ArrayList<>(m));
    }

    static int enumerateBoundaries(BitSet C, List<BitSet> subs, int m, int startIdx,
                                   BitSet used, List<BitSet> chosen) {
        if (++enumOps > OPS_BUDGET) throw BUDGET;
        if (chosen.size() == m - 1) {
            BitSet last = BitSet.newBitSet(C); last.andNot(used);
            if (last.isEmpty() || !isObs(last)) return 0;
            if (compareBitSets(chosen.get(chosen.size() - 1), last) >= 0) return 0; // canonical: last is largest
            BitSet[] parts = new BitSet[m];
            for (int i = 0; i < m - 1; i++) parts[i] = chosen.get(i);
            parts[m - 1] = last;
            return countAllNovelResolutions(C, parts);
        }
        int count = 0;
        for (int i = startIdx; i < subs.size(); i++) {
            BitSet pb = subs.get(i);
            if (pb.intersects(used)) continue;
            chosen.add(pb);
            BitSet newUsed = BitSet.newBitSet(used); newUsed.or(pb);
            count += enumerateBoundaries(C, subs, m, i + 1, newUsed, chosen);
            chosen.remove(chosen.size() - 1);
        }
        return count;
    }

    /** Number of all-novel binary resolutions of C into the given observed parts (subset DP over parts). */
    static int countAllNovelResolutions(BitSet C, BitSet[] parts) {
        int k = parts.length;
        if (k == 1) return 1;
        int full = (1 << k) - 1;
        BitSet[] unionOf = new BitSet[1 << k];
        unionOf[0] = BitSet.newBitSet(nTaxa);
        for (int mask = 1; mask <= full; mask++) {
            int low = Integer.numberOfTrailingZeros(mask);
            BitSet u = BitSet.newBitSet(unionOf[mask & (mask - 1)]); u.or(parts[low]);
            unionOf[mask] = u;
        }
        int[] f = new int[1 << k];
        for (int mask = 1; mask <= full; mask++) {
            if (Integer.bitCount(mask) == 1) { f[mask] = 1; continue; }
            int low = mask & (-mask), rest = mask ^ low, count = 0;
            for (int sub = rest; ; sub = (sub - 1) & rest) {
                int s1 = sub | low, s2 = mask ^ s1;
                if (s2 != 0 && splitAllowed(mask == full, unionOf[mask], unionOf[s1], unionOf[s2]))
                    count += f[s1] * f[s2];
                if (sub == 0) break;
            }
            f[mask] = count;
        }
        return f[full];
    }

    /** A split is allowed in a maximal region iff: at the region root C (observed) the split is
     *  unobserved (a real escape); at an intermediate node the clade itself is novel (a maximal
     *  region stops at observed clades, matching {@link #boundarySize}). */
    static boolean splitAllowed(boolean top, BitSet union, BitSet a, BitSet b) {
        return top ? !isSplitObserved(union, a, b) : !isObs(union);
    }

    static boolean isSplitObserved(BitSet parent, BitSet a, BitSet b) {
        Map<BitSet, Integer> sp = obs.get(parent);
        if (sp == null) return false;
        return sp.containsKey(a.get(parent.nextSetBit(0)) ? a : b);
    }

    /** Canonical total order on clade bitsets (lexicographic by set-bit indices). */
    static int compareBitSets(BitSet a, BitSet b) {
        int ia = a.nextSetBit(0), ib = b.nextSetBit(0);
        while (ia >= 0 && ib >= 0) {
            if (ia != ib) return Integer.compare(ia, ib);
            ia = a.nextSetBit(ia + 1); ib = b.nextSetBit(ib + 1);
        }
        return Integer.compare(ia, ib);
    }

    /** Boundary size m of the escape region rooted at v (maximal observed/leaf subclades below). */
    static int boundarySize(Node v, Map<Node, BitSet> bits) {
        int m = 0;
        for (Node c : v.getChildren()) {
            if (c.isLeaf() || obs.containsKey(bits.get(c))) m++;
            else m += boundarySize(c, bits);
        }
        return m;
    }

    /** Conditional 1-mu model with the generalized eps-reserve escape and a reservability-gated
     *  (1 - mu - tail) discount: only clades that can actually escape reserve mass; cherries and
     *  no-escape clades keep the raw CCD1 CCP undiscounted (mirrors KRegCCD's reservable() gate). */
    static double scoreReserve(Tree t, double mu) {
        Map<Node, BitSet> bits = new HashMap<>();
        computeBits(t.getRoot(), bits);
        double logP = 0;
        for (Node v : t.getNodesAsArray()) {
            if (v.isLeaf()) continue;
            BitSet pb = bits.get(v);
            if (!obs.containsKey(pb)) continue;       // novel clade: counted at its region top
            BitSet canon = canonChild(pb, bits.get(v.getChildren().get(0)), bits.get(v.getChildren().get(1)));
            Integer cnt = obs.get(pb).get(canon);
            if (cnt != null) {
                if (reservable(pb)) {                                       // discount only escaping clades
                    double resv = Math.min(mu + tailFor(pb, mu), 1 - 1e-12);
                    logP += Math.log(1 - resv);
                }
                logP += Math.log(cnt) - Math.log(total.get(pb));           // raw CCD1 CCP
            } else {
                int m = boundarySize(v, bits);                              // region top: eps^(new splits)
                double eps = epsFor(pb, mu);                                 // (populates fallbackSet)
                logP += (m - 1) * Math.log(eps);
                totalRegionTops++;
                regionHist[Math.min(m, regionHist.length - 1)]++;
                if (fallbackSet.contains(pb)) fallbackRegionHits++;
            }
        }
        return logP;
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
