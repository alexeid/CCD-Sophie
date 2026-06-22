package ccd.model;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beastfx.app.treeannotator.TreeAnnotator.TreeSet;
import ccd.model.bitsets.BitSet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * MRegCCD -- the one-parameter "per-new-split" regularised CCD. It unifies RegCCD's split-expansion
 * {@code alpha} and KRegCCD's escape {@code mu} into a single per-clade escape rate, giving a
 * full-support tree distribution with <em>one</em> hyperparameter {@code mu} (and no {@code alpha}).
 *
 * <p>The model is a plain {@link CCD1} backbone (raw conditional clade probabilities, no smoothing)
 * extended with a per-clade escape reserve. The distribution is defined conditionally, clade by clade
 * (chain rule over the observed-clade DAG; no global partition function):
 * <ul>
 *   <li>An <em>observed split</em> {@code C -> {L, R}} (seen in training) is priced
 *       {@code (1 - mu - tail(C)) * ccp(L|R)} when {@code C} can escape, else just {@code ccp(L|R)}.</li>
 *   <li>An <em>escape</em> at {@code C} resolves it through novel intermediate clades down to a
 *       boundary of {@code m} observed subclades (a maximal "blue" region). Such a region has
 *       {@code m - 1} new splits and is priced {@code eps(C)^(m-1)} -- one factor of {@code eps} per
 *       new split (so a recombination of two observed subclades, {@code m = 2}, costs one
 *       {@code eps}). This is the difference from KRegCCD, which makes recombinations representable in
 *       its {@code alpha}-expanded backbone and charges {@code eps} only per novel <em>clade</em>
 *       ({@code eps^(m-2)}).</li>
 * </ul>
 *
 * <p>The per-clade escape rate {@code eps(C)} is the root of {@code sum_{m>=2} M_m(C) eps^(m-1) = mu},
 * where {@code M_m(C)} counts the all-novel resolutions of {@code C} with an {@code m}-part boundary
 * (the FLAT weighting: each distinct novel resolution counted once). Computing the full sum is
 * #P-hard, so -- mirroring KRegCCD -- the orders {@code m = 2..reserveDepth} are enumerated exactly
 * (bounded by an op-budget) and the omitted higher orders are a geometric tail correction added to
 * {@code mu} (so observed splits are discounted by {@code 1 - mu - tail}, keeping the conditional
 * properly normalised; truncating without the tail super-normalises). A clade with no escape route
 * ({@code M_m = 0} for all computed {@code m}) is not reservable and keeps its raw CCP undiscounted.
 *
 * <p>Every tree on the taxon set has positive probability (full support), so {@link #containsTree}
 * is always true and {@link #getLogProbabilityOfTree} is finite for all trees.
 *
 * @author Claude (CCD-Sophie)
 */
public class MRegCCD extends CCD1 {

    /** Default per-clade escape probability (the RSV2 operating point of the conditional model). */
    public static final double DEFAULT_MU = 0.0159;

    /** Default reserve depth: enumerate boundary sizes {@code m = 2..DEFAULT_RESERVE_DEPTH} exactly. */
    public static final int DEFAULT_RESERVE_DEPTH = 5;

    /** Per-clade enumeration-op budget (mirrors KRegCCD's; bounds the boundary enumeration). */
    private static final long OPS_BUDGET = Long.parseLong(System.getProperty("mreg.enumOps", "20000000"));

    private static final class BudgetExceeded extends RuntimeException {
        BudgetExceeded() {
            super(null, null, false, false);
        }
    }

    private static final BudgetExceeded BUDGET_EXCEEDED = new BudgetExceeded();

    /** Per-clade escape probability (the single hyperparameter). */
    private final double mu;

    /** Max boundary size enumerated when solving eps; deeper orders are a geometric tail. */
    private final int reserveDepth;

    /** Whether the {@code (1 - mu - tail)} discount carries the geometric tail correction. */
    private final boolean useTail;

    /** Observed-clade bitsets (incl. leaves), sorted canonically; built lazily. */
    private List<BitSet> sortedCladeBits;
    private final Map<BitSet, List<BitSet>> subCache = new HashMap<>();
    private final Map<BitSet, int[]> countsCache = new HashMap<>();
    private long enumOps;

    public MRegCCD(List<Tree> trees, double burnin, double mu) {
        this(trees, burnin, mu, DEFAULT_RESERVE_DEPTH, true);
    }

    public MRegCCD(List<Tree> trees, double burnin, double mu, int reserveDepth, boolean useTail) {
        super(trees, burnin);
        validate(mu, reserveDepth);
        this.mu = mu;
        this.reserveDepth = reserveDepth;
        this.useTail = useTail;
    }

    public MRegCCD(TreeSet treeSet, double mu) {
        this(treeSet, mu, DEFAULT_RESERVE_DEPTH, true);
    }

    public MRegCCD(TreeSet treeSet, double mu, int reserveDepth, boolean useTail) {
        super(treeSet);
        validate(mu, reserveDepth);
        this.mu = mu;
        this.reserveDepth = reserveDepth;
        this.useTail = useTail;
    }

    private static void validate(double mu, int reserveDepth) {
        if (mu <= 0 || mu >= 1) {
            throw new IllegalArgumentException("mu must be in (0, 1), got " + mu);
        }
        if (reserveDepth < 2) {
            throw new IllegalArgumentException("reserveDepth must be >= 2, got " + reserveDepth);
        }
    }

    /** The per-clade escape probability this model was built with. */
    public double getMu() {
        return mu;
    }

    public int getReserveDepth() {
        return reserveDepth;
    }

    /**
     * Reserve counts {@code M_m(C)} by boundary size {@code m} (array index {@code m}, valid for
     * {@code m = 2..min(|C|, reserveDepth)}); {@code M_m} is the number of all-novel resolutions of
     * {@code C} with an {@code m}-part boundary. The first coefficient {@code M_2} (the {@code eps^1}
     * term) is exactly the number of CCD0-expanded splits of {@code C} -- recombinations of two
     * observed subclades whose split was never observed -- since those are the only escapes with no
     * other novel (blue) clade. Exposed for inspection and cross-checks.
     */
    public int[] reserveCounts(BitSet cladeInBits) {
        return countsFor(cladeInBits).clone();
    }

    @Override
    public String toString() {
        return "MRegCCD [mu = " + mu + ", reserveDepth = " + reserveDepth + ", tail = " + useTail
                + ", per-new-split, full support]";
    }

    /* ----------------------------------------------------------------------
     * Scoring
     * ------------------------------------------------------------------- */

    @Override
    public double getLogProbabilityOfTree(Tree tree) {
        return scoreTree(tree, mu);
    }

    /**
     * Full-support log-probability at an arbitrary escape probability {@code scoreMu}, reusing this
     * model's ({@code mu}-independent) backbone and cached reserve counts. Lets a parameter search /
     * cross-validation evaluate many {@code mu} on one trained model without rebuilding. For
     * {@code scoreMu == mu} it equals {@link #getLogProbabilityOfTree(Tree)}.
     */
    public double getLogProbabilityOfTree(Tree tree, double scoreMu) {
        if (scoreMu <= 0 || scoreMu >= 1) {
            throw new IllegalArgumentException("scoreMu must be in (0, 1), got " + scoreMu);
        }
        return scoreTree(tree, scoreMu);
    }

    @Override
    public double getProbabilityOfTree(Tree tree) {
        return Math.exp(getLogProbabilityOfTree(tree));
    }

    /** Always true: MRegCCD is full support, so every tree on this taxon set has positive probability. */
    @Override
    public boolean containsTree(Tree tree) {
        return true;
    }

    private double scoreTree(Tree tree, double scoreMu) {
        Map<Node, BitSet> bits = new HashMap<>();
        computeBits(tree.getRoot(), bits);
        double logp = 0.0;
        for (Node v : tree.getNodesAsArray()) {
            if (v.isLeaf()) {
                continue;
            }
            BitSet vb = bits.get(v);
            Clade c = getClade(vb);
            if (c == null) {
                continue; // novel clade: scored once at its maximal region's top
            }
            BitSet b1 = bits.get(v.getChildren().get(0));
            BitSet b2 = bits.get(v.getChildren().get(1));
            if (isSplitObserved(vb, b1, b2)) {
                if (reservable(vb)) { // discount only clades that can actually escape
                    double resv = Math.min(scoreMu + (useTail ? tailFor(vb, scoreMu) : 0.0), 1 - 1e-12);
                    logp += Math.log(1.0 - resv);
                }
                logp += rawLogCCP(c, b1, b2); // raw CCD1 CCP
            } else {
                // region top: an observed clade resolved through a novel split. m-1 new splits.
                int m = boundarySize(v, bits);
                logp += (m - 1) * Math.log(epsFor(vb, scoreMu));
            }
        }
        return logp;
    }

    /* ----------------------------------------------------------------------
     * Per-clade reserve  (M_m counts -> eps, tail; mirrors KRegCCD.computeReg)
     * ------------------------------------------------------------------- */

    /** Whether clade {@code C} (given in bits) reserves any escape mass up to {@code reserveDepth}. */
    boolean reservable(BitSet C) {
        for (int v : countsFor(C)) {
            if (v > 0) {
                return true;
            }
        }
        return false;
    }

    /** Escape root {@code eps} solving {@code sum_{m>=2} M_m eps^(m-1) = scoreMu} (monotone bisection). */
    double epsFor(BitSet C, double scoreMu) {
        int[] n = countsFor(C);
        if (!reservable(C)) {
            return scoreMu; // crude fallback (no escape route within reserveDepth); should not be hit
        }
        return solveEps(n, scoreMu);
    }

    /** Omitted-tail escape mass beyond the computed orders: geometric bound from the top two orders. */
    double tailFor(BitSet C, double scoreMu) {
        int[] n = countsFor(C);
        int last = n.length - 1;
        if (last < 3) {
            return 0.0;
        }
        int nLast = n[last], nPrev = n[last - 1];
        if (nLast <= 0 || nPrev <= 0) {
            return 0.0;
        }
        double eps = epsFor(C, scoreMu);
        double rho = ((double) nLast / nPrev) * eps;
        if (rho <= 0 || rho >= 1) {
            return 0.0;
        }
        return Math.min(nLast * Math.pow(eps, last - 1) * rho / (1 - rho), scoreMu);
    }

    /** M_m counts (index m = boundary size, 2..min(|C|, reserveDepth)); cached, mu-independent. */
    int[] countsFor(BitSet C) {
        int[] cached = countsCache.get(C);
        if (cached != null) {
            return cached;
        }
        int card = C.cardinality();
        int[] n = new int[Math.min(card, reserveDepth) + 1];
        if (card >= 2) {
            List<BitSet> subs = subclades(C);
            enumOps = 0;
            for (int m = 2; m < n.length; m++) {
                try {
                    n[m] = countBoundaries(C, subs, m);
                } catch (BudgetExceeded e) {
                    break; // deeper orders omitted (negligible, like the tail)
                }
            }
        }
        countsCache.put(C, n);
        return n;
    }

    private static double solveEps(int[] n, double mu) {
        double lo = 0.0, hi = 1.0;
        while (evalReserve(n, hi) < mu) {
            hi *= 2.0;
        }
        for (int it = 0; it < 100; it++) {
            double mid = 0.5 * (lo + hi);
            if (evalReserve(n, mid) < mu) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    }

    /** {@code sum_{m>=2} n[m] x^(m-1)}. */
    private static double evalReserve(int[] n, double x) {
        double s = 0.0;
        for (int m = 2; m < n.length; m++) {
            if (n[m] > 0) {
                s += n[m] * Math.pow(x, m - 1);
            }
        }
        return s;
    }

    /** Count m-part boundaries of C into observed subclades, weighted by their all-novel pathcount. */
    private int countBoundaries(BitSet C, List<BitSet> subs, int m) {
        return enumerateBoundaries(C, subs, m, 0, BitSet.newBitSet(leafArraySize), new ArrayList<>(m));
    }

    private int enumerateBoundaries(BitSet C, List<BitSet> subs, int m, int startIdx,
                                    BitSet used, List<BitSet> chosen) {
        if (++enumOps > OPS_BUDGET) {
            throw BUDGET_EXCEEDED;
        }
        if (chosen.size() == m - 1) {
            BitSet last = BitSet.newBitSet(C);
            last.andNot(used);
            if (last.isEmpty() || !isObs(last)) {
                return 0;
            }
            if (compareBitSets(chosen.get(chosen.size() - 1), last) >= 0) {
                return 0; // canonical: the derived last part must be the largest
            }
            BitSet[] parts = new BitSet[m];
            for (int i = 0; i < m - 1; i++) {
                parts[i] = chosen.get(i);
            }
            parts[m - 1] = last;
            return countAllNovelResolutions(C, parts);
        }
        int count = 0;
        for (int i = startIdx; i < subs.size(); i++) {
            BitSet pb = subs.get(i);
            if (pb.intersects(used)) {
                continue;
            }
            chosen.add(pb);
            BitSet newUsed = BitSet.newBitSet(used);
            newUsed.or(pb);
            count += enumerateBoundaries(C, subs, m, i + 1, newUsed, chosen);
            chosen.remove(chosen.size() - 1);
        }
        return count;
    }

    /**
     * Number of all-novel binary resolutions of C into the given observed parts (subset DP over the
     * parts). A split is allowed iff: at the region root (full mask = C, an observed clade) the split
     * is unobserved (a real escape); at an intermediate node the clade itself is novel (a maximal
     * region stops at observed clades, matching {@link #boundarySize}).
     */
    private int countAllNovelResolutions(BitSet C, BitSet[] parts) {
        int k = parts.length;
        if (k == 1) {
            return 1;
        }
        int full = (1 << k) - 1;
        BitSet[] unionOf = new BitSet[1 << k];
        unionOf[0] = BitSet.newBitSet(leafArraySize);
        for (int mask = 1; mask <= full; mask++) {
            int low = Integer.numberOfTrailingZeros(mask);
            BitSet u = BitSet.newBitSet(unionOf[mask & (mask - 1)]);
            u.or(parts[low]);
            unionOf[mask] = u;
        }
        int[] f = new int[1 << k];
        for (int mask = 1; mask <= full; mask++) {
            if (Integer.bitCount(mask) == 1) {
                f[mask] = 1;
                continue;
            }
            int low = mask & (-mask), rest = mask ^ low, count = 0;
            for (int sub = rest; ; sub = (sub - 1) & rest) {
                int s1 = sub | low, s2 = mask ^ s1;
                if (s2 != 0 && splitAllowed(mask == full, unionOf[mask], unionOf[s1], unionOf[s2])) {
                    count += f[s1] * f[s2];
                }
                if (sub == 0) {
                    break;
                }
            }
            f[mask] = count;
        }
        return f[full];
    }

    private boolean splitAllowed(boolean top, BitSet union, BitSet a, BitSet b) {
        return top ? !isSplitObserved(union, a, b) : !isObs(union);
    }

    /* ----------------------------------------------------------------------
     * Observed-backbone queries (over the inherited CCD1 clade DAG)
     * ------------------------------------------------------------------- */

    private boolean isObs(BitSet x) {
        return getClade(x) != null; // leaves are clades too
    }

    private boolean isSplitObserved(BitSet parentBits, BitSet aBits, BitSet bBits) {
        Clade parent = getClade(parentBits);
        if (parent == null) {
            return false;
        }
        Clade a = getClade(aBits);
        Clade b = getClade(bBits);
        if (a == null || b == null) {
            return false;
        }
        return parent.getCladePartition(a, b) != null;
    }

    private double rawLogCCP(Clade parent, BitSet aBits, BitSet bBits) {
        CladePartition p = parent.getCladePartition(getClade(aBits), getClade(bBits));
        return p.getLogCCP();
    }

    /** Observed clades (incl. leaves) strictly contained in C, in canonical order; cached. */
    private List<BitSet> subclades(BitSet C) {
        return subCache.computeIfAbsent(C, c -> {
            List<BitSet> out = new ArrayList<>();
            int card = c.cardinality();
            for (BitSet x : sortedCladeBits()) {
                if (x.cardinality() < card && subset(x, c)) {
                    out.add(x);
                }
            }
            return out;
        });
    }

    private List<BitSet> sortedCladeBits() {
        if (sortedCladeBits == null) {
            List<BitSet> all = new ArrayList<>();
            for (Clade c : getClades()) {
                all.add(c.getCladeInBits());
            }
            all.sort(MRegCCD::compareBitSets);
            sortedCladeBits = all;
        }
        return sortedCladeBits;
    }

    /** Boundary size of the maximal region rooted at v: count of maximal observed/leaf subclades below. */
    private int boundarySize(Node v, Map<Node, BitSet> bits) {
        int m = 0;
        for (Node child : v.getChildren()) {
            if (child.isLeaf() || getClade(bits.get(child)) != null) {
                m++;
            } else {
                m += boundarySize(child, bits);
            }
        }
        return m;
    }

    private BitSet computeBits(Node v, Map<Node, BitSet> bits) {
        BitSet b = BitSet.newBitSet(leafArraySize);
        if (v.isLeaf()) {
            b.set(v.getNr());
        } else {
            b.or(computeBits(v.getChildren().get(0), bits));
            b.or(computeBits(v.getChildren().get(1), bits));
        }
        bits.put(v, b);
        return b;
    }

    private static boolean subset(BitSet a, BitSet c) {
        BitSet tmp = BitSet.newBitSet(a);
        tmp.andNot(c);
        return tmp.isEmpty();
    }

    /** Canonical total order on clade bitsets (lexicographic by set-bit indices). */
    private static int compareBitSets(BitSet a, BitSet b) {
        int ia = a.nextSetBit(0), ib = b.nextSetBit(0);
        while (ia >= 0 && ib >= 0) {
            if (ia != ib) {
                return Integer.compare(ia, ib);
            }
            ia = a.nextSetBit(ia + 1);
            ib = b.nextSetBit(ib + 1);
        }
        return Integer.compare(ia, ib);
    }

    /* ----------------------------------------------------------------------
     * Sampling -- needed by ITreeDistribution (PIT). Implemented in a later step.
     * ------------------------------------------------------------------- */

    @Override
    public Tree sampleTree(HeightSettingStrategy heightStrategy) {
        throw new UnsupportedOperationException(
                "MRegCCD escape sampling is not yet implemented; use getLogProbabilityOfTree for scoring.");
    }
}
