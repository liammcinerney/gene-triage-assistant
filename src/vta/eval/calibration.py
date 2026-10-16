"""
Judge calibration: does the judge catch errors we planted, and leave faithful
answers alone?

    python -m src.vta.eval.calibration [--repeats 3]

Method (like mutation testing): take real retrieved evidence, write answers
whose faults we KNOW because we put them there, and see whether the judge
finds them. Limits worth stating out loud: it only covers the fault types
seeded below, the answers are synthetic, and a handful of items is a sanity
check, not a validation study. Real hand-labelled agent answers would be better.
"""

import argparse
import json

from src.vta.eval.judge import evidence_from_trace, judge_answer

# Both chunks are verbatim retrieved text from earlier runs of literature_search.
CFTR_CHUNK = {
    "pmid": "40944809",
    "title": "Personalized Medicine in Cystic Fibrosis: Characterization of Eight Rare CFTR Variants",
    "chunk_text": (
        "Despite the approval of CFTR modulator (CFTRm) drugs for specific variants, many people with cystic fibrosis "
        "with non-eligible genotypes may still benefit from these medications. Indeed, recent studies show that some "
        "rare CFTR variants can be rescued by approved CFTRm drugs. we assessed the efficacy of CFTRm drugs on eight "
        "rare CFTR variants: p.Pro5Leu, p.Pro205Ser, p.Leu206Trp, p.Arg347Pro, p.Ile507del, p.Ser945Leu, p.Met1137Arg, "
        "and p.Asp1152His. Patient-derived intestinal organoids with these variants in heterozygosity with other cystic "
        "fibrosis-causing ones were analyzed by the forskolin-induced swelling assay. CFTR function in intestinal "
        "organoids with genotypes p.Asp1152His/p.Phe508del, p.Asp1152His/p.Asn1303Lys, p.Pro5Leu/p.Phe508del, "
        "p.Leu206Trp/p.Phe508del, p.Ser945Leu/p.Phe508del, p.Pro205Ser/p.Tyr1092Ter, and p.Met1137Arg/c.2657+5G>A was "
        "rescued by currently available CFTRm drugs, this was not observed for organoids with genotypes "
        "p.Arg347Pro/p.Phe508del (elexacaftor/tezacaftor/ivacaftor was not tested) and p.Ile507del/p.Gln890Ter. "
        "People with cystic fibrosis who, based on our data, started CFTRm therapy showed a clinical improvement, "
        "including an increase in lung function, and a reduction in sweat chloride levels."
    ),
}
HRT_CHUNK = {
    "pmid": "36637775",
    "title": "Hormone replacement therapy in BRCA mutation carriers: how shall we do no harm?",
    "chunk_text": (
        "Women with a BRCA mutation have an increased risk of developing breast and ovarian cancer. Bilateral "
        "salpingo-oophorectomy is the only effective strategy to reduce this risk. Hormone replacement therapy (HRT) is "
        "the only strategy able to significantly compensate for the loss of ovarian hormone production and counteract "
        "menopausal symptoms. There is strong evidence that short-term HRT use does not increase the risk of breast "
        "cancer among women with a BRCA1 mutation. Few data are available on BRCA2 mutation carriers. Therefore, BRCA "
        "mutation carriers require careful counseling about the outcomes of their RRSO, including menopausal symptoms "
        "and/or the fear associated with HRT use."
    ),
}

EVIDENCE = evidence_from_trace(
    [
        {
            "tool": "literature_search",
            "input": {"query": "CFTR rare variants modulators; BRCA HRT"},
            "output": {"found": True, "results": [CFTR_CHUNK, HRT_CHUNK]},
            "is_error": False,
        }
    ]
)
QUESTION = "Summarise the evidence on CFTR modulators in rare variants and HRT in BRCA carriers."

# expect: "clean" -> judge should flag nothing; otherwise the seeded fault type.
ITEMS = [
    ("clean_a", "clean", "Intestinal organoids with p.Asp1152His/p.Phe508del were rescued by approved CFTR modulators, whereas rescue was not observed for p.Ile507del/p.Gln890Ter (PMID 40944809)."),
    ("clean_b", "clean", "Organoid function was rescued for p.Leu206Trp/p.Phe508del (PMID 40944809). Separately, a review reports strong evidence that short-term HRT does not increase breast cancer risk in women with a BRCA1 mutation (PMID 36637775)."),
    ("fabricated_number", "unsupported", "Patients on modulator therapy saw their lung function rise by 12% on average (PMID 40944809)."),
    ("contradicted", "contradicted", "Organoids carrying p.Ile507del were successfully rescued by approved CFTR modulators (PMID 40944809)."),
    ("dropped_qualifier", "overstated", "The study showed that p.Arg347Pro does not respond to CFTR modulators (PMID 40944809)."),
    ("outside_knowledge", "unsupported", "CFTR encodes a chloride channel, and loss of its function causes thick mucus in the lungs (PMID 40944809)."),
    ("wrong_pmid", "unsupported", "Short-term hormone replacement therapy does not increase breast cancer risk in BRCA1 carriers (PMID 40944809)."),
    ("overstated_consensus", "overstated", "It is firmly established that HRT is safe for all BRCA mutation carriers (PMID 36637775)."),
]


def flagged(verdict: dict) -> bool:
    # Same rule the runner uses to fail a case: any non-supported claim rated major.
    return not verdict["grounded"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    tokens = 0
    rows = []
    for name, expect, answer in ITEMS:
        runs = [judge_answer(QUESTION, EVIDENCE, answer, use_cache=False) for _ in range(args.repeats)]
        tokens += sum(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in runs)
        flags = [flagged(r) for r in runs]
        labels = sorted({c["verdict"] for r in runs for c in r["claims"] if c["verdict"] != "supported"})
        unverified = sum(r["unverified_supported"] for r in runs)
        rows.append((name, expect, flags, labels, unverified, runs))

    print(f"\n{'item':<22}{'planted':<14}{'flagged':<10}{'labels the judge used':<32}{'unverified quotes'}")
    correct = 0
    for name, expect, flags, labels, unverified, _ in rows:
        want = expect != "clean"
        hits = sum(f == want for f in flags)
        correct += hits
        mark = "ok " if hits == len(flags) else ("~  " if hits else "BAD")
        print(f"{mark} {name:<19}{expect:<14}{sum(flags)}/{len(flags):<8}{','.join(labels) or '-':<32}{unverified}")

    n = len(ITEMS) * args.repeats
    seeded = [r for r in rows if r[1] != "clean"]
    clean = [r for r in rows if r[1] == "clean"]
    detect = sum(sum(r[2]) for r in seeded) / (len(seeded) * args.repeats)
    false_pos = sum(sum(r[2]) for r in clean) / (len(clean) * args.repeats)
    print(f"\nverdict correct in {correct}/{n} judgements | seeded faults caught {detect:.0%} | faithful answers wrongly flagged {false_pos:.0%}")
    print(f"unstable items (judge disagreed with itself across repeats): {[r[0] for r in rows if 0 < sum(r[2]) < len(r[2])] or 'none'}")
    print(f"judge tokens used: {tokens:,}")

    # Show the reasoning on anything the judge got wrong, so we can tell a bad
    # judge from a bad item.
    for name, expect, flags, labels, _, runs in rows:
        want = expect != "clean"
        if any(f != want for f in flags):
            print(f"\n--- {name} (planted: {expect}) first run claims ---")
            print(json.dumps([{k: c[k] for k in ("claim", "verdict", "explanation")} for c in runs[0]["claims"]], indent=1))


if __name__ == "__main__":
    main()
