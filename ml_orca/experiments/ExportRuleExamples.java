// Optional offline adapter: reuse WeTune's parser and example-plan constructor.
// Run with a locally built WeTune classpath; no prover or optimizer policy changes.
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import wtune.superopt.constraint.Constraint;
import wtune.superopt.fragment.*;
import wtune.superopt.substitution.Substitution;
import wtune.superopt.substitution.SubstitutionSupport;
import static wtune.sql.plan.PlanSupport.translateAsAst;

class ExportRuleExamples {
  private static void checkTree(Op op) {
    switch (op.kind()) {
      case INPUT, PROJ -> {}
      case SET_OP -> {
        if (((Union) op).usesInputSymbols())
          throw new UnsupportedOperationException("explicit set-op input mapping");
      }
      case INNER_JOIN, LEFT_JOIN -> {
        Join join = (Join) op;
        if (!join.usesEqualitySymbols() || join.usesOutputSymbols()
            || join.usesPredicateSymbols())
          throw new UnsupportedOperationException("join symbol layout");
      }
      case IN_SUB_FILTER -> {
        if (((InSubFilter) op).usesPredicateSymbols())
          throw new UnsupportedOperationException("InSub predicate layout");
      }
      // ponytail: only lossless PlanTranslator layouts; extend the shared
      // translator before accepting Agg's fixed COUNT or opaque predicates.
      default -> throw new UnsupportedOperationException("operator " + op.kind());
    }
    for (Op child : op.predecessors()) checkTree(child);
  }

  private static void checkRule(Substitution rule) {
    checkTree(rule._0().root());
    checkTree(rule._1().root());
    for (Constraint c : rule.constraints()) {
      switch (c.kind()) {
        case TableEq, AttrsEq, SchemaEq -> {}
        case AttrsSub -> {
          Symbol.Kind domain = c.symbols()[1].kind();
          if (domain != Symbol.Kind.TABLE && domain != Symbol.Kind.SCHEMA && domain != Symbol.Kind.ATTRS)
            throw new UnsupportedOperationException("non-relational AttrsSub domain");
        }
        case Unique, NotNull -> {
          Symbol table = c.symbols()[0], attrs = c.symbols()[1];
          if (table.ctx() != rule._0().symbols() || table.kind() != Symbol.Kind.TABLE
              || attrs.ctx() != rule._0().symbols()
              || rule.constraints().sourceOf(attrs) != table)
            throw new UnsupportedOperationException("non-base integrity constraint");
        }
        default -> throw new UnsupportedOperationException("constraint " + c.kind());
      }
    }
  }

  private static String encode(String text) {
    return Base64.getEncoder().encodeToString(text.getBytes(StandardCharsets.UTF_8));
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 1) throw new IllegalArgumentException("require original rule file");
    boolean minimalInputs = false;
    int inputColumns = 1;
    for (int i = 1; i < args.length; ++i) {
      if (args[i].equals("--minimal-inputs")) minimalInputs = true;
      else if (args[i].startsWith("--input-columns="))
        inputColumns = Integer.parseInt(args[i].substring("--input-columns=".length()));
      else throw new IllegalArgumentException("unknown option: " + args[i]);
    }
    if (inputColumns < 1) throw new IllegalArgumentException("positive input width required");
    int lineNumber = 0;
    for (String raw : Files.readAllLines(Path.of(args[0]))) {
      ++lineNumber;
      String line = raw.strip();
      if (line.isEmpty() || line.startsWith("#")) continue;
      String stage = "parse";
      try {
        Substitution rule = Substitution.parse(line.split("\t", 2)[0]);
        stage = "supported_domain";
        checkRule(rule);
        stage = "instantiate";
        var pair = SubstitutionSupport.translateAsPlan(rule, minimalInputs, inputColumns);
        var lhs = pair.getLeft();
        var rhs = pair.getRight();
        if (lhs == null || rhs == null) throw new IllegalArgumentException("unbound plan");
        stage = "sql_export";
        String schema = lhs.schema().toDdl("mysql", new StringBuilder()).toString();
        String source = translateAsAst(lhs, lhs.root(), true).toString();
        String target = translateAsAst(rhs, rhs.root(), true).toString();
        System.out.println(lineNumber + "\tok\t" + encode(schema) + "\t"
            + encode(source) + "\t" + encode(target));
      } catch (RuntimeException | AssertionError ex) {
        System.out.println(lineNumber + "\t" + stage + "\t"
            + encode(ex.getClass().getSimpleName() + ": " + ex.getMessage()));
      }
    }
  }
}
