package test.ccd.model;

import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import ccd.model.KRegCCD;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Five-taxon analogue of {@link KRegFourTaxaTest}: add taxon E as an outgroup to the two
 * sampled four-taxon trees and enumerate all (2*5-3)!! = 105 rooted topologies.
 *
 * Sample (each once): T1 = ((((A,B),C),D),E), T2 = ((((D,C),B),A),E).
 * Scored with TailMode.NONE (exact, sums to 1) at alpha=0.4, mu=0.05.
 */
public class KRegFiveTaxaTest {

    private static final List<String> TAXA = Arrays.asList("A", "B", "C", "D", "E");
    private static final double ALPHA = 0.4, MU = 0.05;

    // ---- functional tree for topology enumeration ----
    private sealed interface T permits Leaf, Node {}
    private record Leaf(String name) implements T {}
    private record Node(T l, T r) implements T {}

    private static String nwk(T t) {
        if (t instanceof Leaf lf) return lf.name() + ":1";
        Node n = (Node) t;
        return "(" + nwk(n.l()) + "," + nwk(n.r()) + "):1";
    }

    /** Topology-only newick (names, no branch lengths) for the MDS dump. */
    private static String topo(T t) {
        if (t instanceof Leaf lf) return lf.name();
        Node n = (Node) t;
        return "(" + topo(n.l()) + "," + topo(n.r()) + ")";
    }

    /** All trees formed by inserting leaf x at every edge of t (2k-1 of them for k leaves). */
    private static List<T> insertAll(T t, String x) {
        List<T> out = new ArrayList<>();
        out.add(new Node(new Leaf(x), t));            // x as sibling of the whole subtree
        if (t instanceof Node n) {
            for (T l2 : insertAll(n.l(), x)) out.add(new Node(l2, n.r()));
            for (T r2 : insertAll(n.r(), x)) out.add(new Node(n.l(), r2));
        }
        return out;
    }

    private static List<T> allTopologies(List<String> taxa) {
        List<T> trees = new ArrayList<>();
        trees.add(new Leaf(taxa.get(0)));
        for (int i = 1; i < taxa.size(); i++) {
            List<T> next = new ArrayList<>();
            for (T t : trees) next.addAll(insertAll(t, taxa.get(i)));
            trees = next;
        }
        return trees;
    }

    @Test
    public void enumerate105() {
        List<T> topos = allTopologies(TAXA);
        System.out.println("topologies enumerated = " + topos.size());

        List<Tree> sample = new ArrayList<>();
        sample.add(new TreeParser(TAXA, "((((A:1,B:1):1,C:1):1,D:1):1,E:1):0;", 1, false));
        sample.add(new TreeParser(TAXA, "((((D:1,C:1):1,B:1):1,A:1):1,E:1):0;", 1, false));

        // Does any (TailMode, k) make the 105 probs sum to 1? NONE/BOUND/SAMPLED at k=2,3.
        for (KRegCCD.TailMode tm : new KRegCCD.TailMode[]{
                KRegCCD.TailMode.NONE, KRegCCD.TailMode.BOUND, KRegCCD.TailMode.SAMPLED}) {
            for (int k : new int[]{2, 3}) {
                KRegCCD c = new KRegCCD(sample, 0, MU, ALPHA, k, tm);
                double s = 0;
                for (T t : topos) {
                    String b = nwk(t);
                    s += c.getProbabilityOfTree(new TreeParser(TAXA, b.substring(0, b.lastIndexOf(':')) + ":0;", 1, false));
                }
                System.out.printf("%-7s k=%d  sum(105) = %.12f%n", tm, k, s);
            }
        }

        KRegCCD ccd = new KRegCCD(sample, 0, MU, ALPHA, 3, KRegCCD.TailMode.NONE);
        double pT1 = ccd.getProbabilityOfTree(sample.get(0));
        double pT2 = ccd.getProbabilityOfTree(sample.get(1));

        // group by probability (rounded), keep an example newick + novel-clade count
        Map<String, int[]> tiers = new LinkedHashMap<>();
        Map<String, String> example = new LinkedHashMap<>();
        Map<String, Integer> exNovel = new LinkedHashMap<>();
        double sum = 0, eOut = 0, eBuried = 0;
        int nOut = 0, nBuried = 0;
        for (T t : topos) {
            String body = nwk(t);
            String nk = body.substring(0, body.lastIndexOf(':'));
            Tree tree = new TreeParser(TAXA, nk + ":0;", 1, false);
            double p = ccd.getProbabilityOfTree(tree);
            int nov = ccd.novelCladeCount(tree);
            sum += p;
            // E-outgroup iff a root child is the leaf E (root split = {A,B,C,D}|{E})
            boolean eOutgroup = t instanceof Node n
                    && (n.l() instanceof Leaf lf && lf.name().equals("E")
                        || n.r() instanceof Leaf rf && rf.name().equals("E"));
            if (eOutgroup) { eOut += p; nOut++; } else { eBuried += p; nBuried++; }
            String key = String.format("%.12f", p);
            tiers.computeIfAbsent(key, k -> new int[1])[0]++;
            example.putIfAbsent(key, nk);
            exNovel.putIfAbsent(key, nov);
        }
        System.out.printf("%nE-outgroup: %d trees, mass = %.12f  (expect (1-mu) = %.4f)%n", nOut, eOut, 1 - MU);
        System.out.printf("E-buried : %d trees, mass = %.12f  (expect mu = %.4f)%n", nBuried, eBuried, MU);
        System.out.printf("root escape shortfall = mu - eBuried = %.12f%n", MU - eBuried);

        // dump newick + probability + novel-clade count for the MDS visualisation
        try (java.io.PrintWriter w = new java.io.PrintWriter(new java.io.FileWriter("doc/kreg-five-taxa.tsv"))) {
            w.println("newick\tprob\tnovel");
            for (T t : topos) {
                Tree tree = new TreeParser(TAXA, topo(t) + ";", 1, false);
                w.printf("%s\t%.12g\t%d%n", topo(t), ccd.getProbabilityOfTree(tree), ccd.novelCladeCount(tree));
            }
        } catch (java.io.IOException e) {
            throw new RuntimeException(e);
        }

        System.out.printf("%nP(T1) = %.12f   P(T2) = %.12f%n", pT1, pT2);
        System.out.printf("distinct probability values = %d%n", tiers.size());
        System.out.printf("%-16s %5s %7s   %s%n", "P(tree)", "count", "#novel", "example");
        tiers.entrySet().stream()
                .sorted((a, b) -> Double.compare(Double.parseDouble(b.getKey()), Double.parseDouble(a.getKey())))
                .forEach(e -> System.out.printf("%-16s %5d %7d   %s%n",
                        e.getKey(), e.getValue()[0], exNovel.get(e.getKey()), example.get(e.getKey())));
        System.out.printf("%nSUM over 105 trees = %.12f%n", sum);
    }
}
