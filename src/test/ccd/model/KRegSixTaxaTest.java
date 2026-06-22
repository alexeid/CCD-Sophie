package test.ccd.model;

import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import ccd.model.KRegCCD;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Six-taxon analogue of {@link KRegFiveTaxaTest}: add an (E,F) outgroup cherry to the two sampled
 * four-taxon trees and enumerate all (2*6-3)!! = 945 rooted topologies.
 *
 * Sample (each once): T1 = ((((A,B),C),D),(E,F)), T2 = ((((D,C),B),A),(E,F)).
 * Scored with TailMode.NONE at alpha=0.4, mu=0.05, reserve depth k=4 (root can reach order-4 novelty).
 * Dumps doc/kreg-six-taxa.tsv (newick, prob, novel) for the RF/MDS visualisations.
 */
public class KRegSixTaxaTest {

    private static final List<String> TAXA = Arrays.asList("A", "B", "C", "D", "E", "F");
    private static final double ALPHA = 0.4, MU = 0.05;

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

    @Test
    public void enumerate945() {
        List<T> topos = allTopologies(TAXA);
        System.out.println("topologies enumerated = " + topos.size());

        List<Tree> sample = new ArrayList<>();
        sample.add(new TreeParser(TAXA, "((((A:1,B:1):1,C:1):1,D:1):1,(E:1,F:1):1):0;", 1, false));
        sample.add(new TreeParser(TAXA, "((((D:1,C:1):1,B:1):1,A:1):1,(E:1,F:1):1):0;", 1, false));
        KRegCCD ccd = new KRegCCD(sample, 0, MU, ALPHA, 4, KRegCCD.TailMode.NONE);

        double sum = 0;
        java.util.Map<String, Integer> tierCount = new java.util.HashMap<>();
        try (java.io.PrintWriter w = new java.io.PrintWriter(new java.io.FileWriter("doc/kreg-six-taxa.tsv"))) {
            w.println("newick\tprob\tnovel");
            for (T t : topos) {
                Tree tree = new TreeParser(TAXA, topo(t) + ";", 1, false);
                double p = ccd.getProbabilityOfTree(tree);
                sum += p;
                tierCount.merge(String.format("%.12f", p), 1, Integer::sum);
                w.printf("%s\t%.12g\t%d%n", topo(t), p, ccd.novelCladeCount(tree));
            }
        } catch (java.io.IOException e) {
            throw new RuntimeException(e);
        }

        System.out.printf("P(T1) = %.12f   P(T2) = %.12f%n",
                ccd.getProbabilityOfTree(sample.get(0)), ccd.getProbabilityOfTree(sample.get(1)));
        System.out.printf("distinct probability values = %d%n", tierCount.size());
        System.out.printf("SUM over 945 trees = %.12f  (shortfall = %.6f)%n", sum, 1 - sum);
    }
}
