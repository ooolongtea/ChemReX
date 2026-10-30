"""Route-assembly acceptance fixes (2026-10-30): a conflicted producer still
proves a material when the consumer prints the identical structure, and a
role prefix ("中间体006-8") is not part of a compound's name identity."""

from pipeline.document.routes import _name, assemble_routes_v2
from pipeline.validation.quality import QUALITY_VERSION

from test_route_repair import compound, reaction, record, scope_doc


SMILES = "C=C(Cl)/N=C(\\I)c1cn(C)c2ccccc12"


def conflicted(reaction_row):
    row = dict(reaction_row)
    row["quality"] = {**row["quality"], "structure_status": "conflict"}
    return row


def test_role_prefix_is_not_part_of_name_identity():
    assert _name("中间体006-8") == _name("006-8")
    assert _name("化合物15的甲磺酸盐") == _name("15的甲磺酸盐")
    assert _name("Intermediate 8") == _name("8")
    assert _name("中间体") == _name("中间体")  # bare role noun stays itself


def test_role_prefix_names_chain_procedure_steps():
    """The same batch is written "中间体006-8" as a product and "006-8" as the
    next step's reactant; without structures both sides must still chain."""
    early = record("example_6", [
        reaction(1, [compound("006-7", "CC")], [compound("中间体006-8")],
                 text="006-7 was reduced to give 中间体006-8."),
        reaction(2, [compound("006-8"), compound("丙烯酰氯", "C=CC(=O)Cl")],
                 [compound("化合物6", "C=CC(=O)NCC")],
                 text="006-8 was acylated with 丙烯酰氯 to give 化合物6."),
    ])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [early])])
    assert len(routes) == 1 and not fragments
    assert [e["basis"] for e in routes[0]["connections"]] == ["exact_name"]


def test_conflicted_producer_links_on_identical_structure():
    """example_6 step 1's product carries structure_status=conflict, yet the
    consumer prints the same structure: the material identity holds and the
    edge must form (the conflict flag stays on the producer for review)."""
    conflicted_step1 = conflicted(reaction(
        1, [compound("001-5", "Clc1ncc(Cl)nc1"), compound("006-1", "Cn1cccc1")],
        [compound("006-2", SMILES)], text="006-1 was coupled with 001-5."))
    chain = record("example_6", [
        conflicted_step1,
        reaction(2, [compound("006-2", SMILES), compound("006-4", "COc1cc(F)c(cc1)N")],
                 [compound("中间体006-5", "COc1cc(F)cc1Nc1nccc(Cl)n1")],
                 text="006-2 was condensed with 006-4 to give 中间体006-5."),
    ])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [chain])])
    assert len(routes) == 1 and not fragments
    assert [e["basis"] for e in routes[0]["connections"]] == ["canonical_smiles"]
    # The conflict itself is still reported, not silently dropped.
    assert any(u.reaction["quality"]["structure_status"] == "conflict"
               for u in routes[0]["selected"])


def test_conflicted_producer_name_only_match_stays_vetoed():
    """Without structures a conflicted producer proves nothing: a bare name
    match must not chain through it."""
    conflicted_step1 = conflicted(reaction(
        1, [compound("A", "CC")], [compound("006-2")], text="A gave 006-2."))
    chain = record("example_6", [
        conflicted_step1,
        reaction(2, [compound("006-2"), compound("B", "CO")],
                 [compound("006-5", "CCC")], text="006-2 reacted with B."),
    ])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [chain])])
    assert not routes and len(fragments) == 1
    assert any(b.get("reason") == "method_not_disclosed_in_scope"
               for u in fragments[0]["selected"] for b in u.boundaries)


def test_conflicted_consumer_stays_vetoed():
    """A conflicted consumer's own material claims are suspect: identical
    structures on both sides do not rescue an edge into it."""
    conflicted_step2 = conflicted(reaction(
        2, [compound("006-2", SMILES)], [compound("006-5", "CCC")],
        text="006-2 was converted to 006-5."))
    chain = record("example_6", [
        reaction(1, [compound("A", "CC")], [compound("006-2", SMILES)],
                 text="A gave 006-2."),
        conflicted_step2,
    ])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [chain])])
    assert not routes and len(fragments) == 1


def test_bare_marker_title_does_not_unconfirm_coverage():
    """A chunk titled only by a paragraph marker ("[0231]") has no titled
    terminal product to check; the vacuous unknown flag must not turn a
    connected route into a fragment."""
    producer = record("example_6", [
        reaction(1, [compound("001-5", "Clc1ncc(Cl)nc1")], [compound("006-5", "COc1cc(F)cc1Nc1nccc(Cl)n1")],
                 text="001-5 was coupled to give 006-5."),
    ])
    producer["terminal_product_coverage"] = "unknown"
    producer["source"]["title"] = "[0231]"
    consumer = record("example_15", [
        reaction(1, [compound("006-5", "COc1cc(F)cc1Nc1nccc(Cl)n1"), compound("015-1", "C1CCNCC1")],
                 [compound("化合物15的甲磺酸盐", "C=CC(=O)NCC")],
                 text="006-5 reacted with 015-1 to give 化合物15的甲磺酸盐."),
    ])
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [producer, consumer])])
    assert len(routes) == 1 and not fragments
    # A real title still carries the coverage obligation.
    producer["source"]["title"] = "实施例6"
    routes, fragments = assemble_routes_v2([scope_doc("doc.pdf", [producer, consumer])])
    assert not routes and len(fragments) == 1
    assert "terminal_product_coverage_unknown" in [g["code"] for g in fragments[0]["gaps"]]
