package test.ccd.model;

import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import ccd.model.KRegCCD;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Decay of KRegCCD probability with rooted-RF distance from a SINGLE observed tree.
 * Builds the model on one caterpillar T0 and enumerates all rooted topologies, grouping by RF
 * distance to T0. Tests the prediction P(T) ~ eps^(d/2): rooted RF = 2 * (novel-clade count), and
 * each novel clade costs a factor eps, so successive even-distance tiers fall by ~eps.
 */
public class KRegSingleTreeDecayTest {

    private sealed interface T permits Leaf, Node {}
    private record Leaf(String name) implements T {}
    private record Node(T l, T r) implements T {}

    private static String topo(T t) {
        if (t instanceof Leaf lf) return lf.name();
        Node n = (Node) t;
        return "(" + topo(n.l()) + "," + topo(n.r()) + ")";
    }

    private static List<T> insertAll(T t, String x) {
        List<T> out = new ArrayList<>();
        out.add(new Node(new Leaf(x), t));
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

    /** Non-trivial clades (size 2..n-1) of a names-only newick. */
    private static Set<Set<String>> clades(String nwk, int n) {
        Set<Set<String>> all = new HashSet<>();
        collect(nwk, all);
        Set<Set<String>> nt = new HashSet<>();
        for (Set<String> c : all) if (c.size() >= 2 && c.size() <= n - 1) nt.add(c);
        return nt;
    }

    private static Set<String> collect(String s, Set<Set<String>> all) {
        if (!s.contains("(")) { Set<String> leaf = new HashSet<>(List.of(s)); return leaf; }
        String inner = s.substring(1, s.length() - 1);
        int depth = 0, start = 0;
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < inner.length(); i++) {
            char ch = inner.charAt(i);
            if (ch == '(') depth++;
            else if (ch == ')') depth--;
            else if (ch == ',' && depth == 0) { parts.add(inner.substring(start, i)); start = i + 1; }
        }
        parts.add(inner.substring(start));
        Set<String> here = new HashSet<>();
        for (String p : parts) here.addAll(collect(p, all));
        all.add(here);
        return here;
    }

    private void run(List<String> taxa, String t0newick, String t0lengths, int k, String dump) {
        int n = taxa.size();
        List<T> topos = allTopologies(taxa);
        KRegCCD ccd = new KRegCCD(List.of(new TreeParser(taxa, t0lengths, 1, false)),
                0, 0.05, 0.4, k, KRegCCD.TailMode.NONE);
        Set<Set<String>> c0 = clades(t0newick, n);
        java.io.PrintWriter dw = null;
        if (dump != null) {
            try { dw = new java.io.PrintWriter(new java.io.FileWriter(dump)); dw.println("newick\tprob\tnovel"); }
            catch (java.io.IOException e) { throw new RuntimeException(e); }
        }

        // group by RF distance to T0
        TreeMap<Integer, double[]> byRF = new TreeMap<>(); // rf -> {count, sumP}
        TreeMap<Integer, Integer> novelAtRF = new TreeMap<>();
        double sum = 0;
        for (T t : topos) {
            String nk = topo(t);
            double p = ccd.getProbabilityOfTree(new TreeParser(taxa, nk + ";", 1, false));
            sum += p;
            Set<Set<String>> ct = clades(nk, n);
            Set<Set<String>> sym = new HashSet<>(c0); sym.addAll(ct);
            Set<Set<String>> inter = new HashSet<>(c0); inter.retainAll(ct);
            int rf = sym.size() - inter.size();
            int novel = ccd.novelCladeCount(new TreeParser(taxa, nk + ";", 1, false));
            byRF.computeIfAbsent(rf, q -> new double[2]);
            byRF.get(rf)[0]++; byRF.get(rf)[1] += p;
            novelAtRF.put(rf, novel);
            if (dw != null) dw.printf("%s\t%.12g\t%d%n", nk, p, novel);
        }
        if (dw != null) dw.close();

        System.out.printf("%n=== %d taxa, single tree T0 = %s, k=%d, sum=%.6f ===%n", n, t0newick, k, sum);
        System.out.printf("%4s %6s %7s %14s %10s%n", "RF", "count", "novel", "mean P", "ratio/prev");
        double prevMean = Double.NaN;
        for (var e : byRF.entrySet()) {
            double mean = e.getValue()[1] / e.getValue()[0];
            double ratio = Double.isNaN(prevMean) ? Double.NaN : mean / prevMean;
            System.out.printf("%4d %6.0f %7d %14.4e %10.4f%n",
                    e.getKey(), e.getValue()[0], novelAtRF.get(e.getKey()), mean, ratio);
            prevMean = mean;
        }
    }

    @Test
    public void decay() {
        run(List.of("A", "B", "C", "D", "E"),
                "((((A,B),C),D),E)", "((((A:1,B:1):1,C:1):1,D:1):1,E:1):0;", 3, "doc/kreg-single5.tsv");
        run(List.of("A", "B", "C", "D", "E", "F"),
                "(((((A,B),C),D),E),F)", "(((((A:1,B:1):1,C:1):1,D:1):1,E:1):1,F:1):0;", 4, null);
    }
}
