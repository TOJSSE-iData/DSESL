from typing import Dict, List, Tuple

ENTITY_DICT = {
    2: "Gene",
    3: "BiologicalProcess",
    1: "SideEffect",
    5: "MolecularFunction",
    7: "Compound",
    4: "Pathway",
    8: "CellularComponent",
    0: "Symptom",
    10: "Anatomy",
    6: "PharmacologicClass",
    9: "Disease",
}
RELATION_DICT = {
    1: ("Gene", "PARTICIPATES", "BiologicalProcess"),
    2: ("Anatomy", "EXPRESSES", "Gene"),
    3: ("Gene", "REGULATES", "Gene"),
    4: ("Gene", "INTERACTS", "Gene"),
    10: ("Compound", "CAUSES", "SideEffect"),
    6: ("Gene", "PARTICIPATES", "MolecularFunction"),
    0: ("Gene", "PARTICIPATES", "CellularComponent"),
    7: ("Gene", "COVARIES", "Gene"),
    8: ("Gene", "PARTICIPATES", "Pathway"),
    5: ("Disease", "ASSOCIATES", "Gene"),
    19: ("Compound", "DOWNREGULATES", "Gene"),
    18: ("Compound", "UPREGULATES", "Gene"),
    11: ("Compound", "BINDS", "Gene"),
    9: ("Disease", "UPREGULATES", "Gene"),
    17: ("Disease", "DOWNREGULATES", "Gene"),
    20: ("Compound", "RESEMBLES", "Compound"),
    13: ("Disease", "PRESENTS", "Symptom"),
    14: ("Disease", "LOCALIZES", "Anatomy"),
    21: ("PharmacologicClass", "INCLUDES", "Compound"),
    15: ("Compound", "TREATS", "Disease"),
    16: ("Disease", "RESEMBLES", "Disease"),
    22: ("Compound", "PALLIATES", "Disease"),
    23: ("Anatomy", "DOWNREGULATES", "Gene"),
    12: ("Anatomy", "UPREGULATES", "Gene"),
}


def _entity_set_from_triplet(triplet_set: set):
    result = set([x[0] for x in triplet_set])
    result.update(set([x[2] for x in triplet_set]))
    return result


def _yield_read_raw_triplet(data_path: str, entity_type_dict: Dict[int, int]):
    with open(data_path, "rt") as f:
        for line in f.readlines():
            # read line
            head, relation, tail = [int(x.strip()) for x in line.split("\t") if x.strip()]
            # build KG triplet
            if ENTITY_DICT[entity_type_dict[head]] == RELATION_DICT[relation][0]:
                _triplet = (head, relation, tail)
            else:
                assert ENTITY_DICT[entity_type_dict[tail]] == RELATION_DICT[relation][0]
                _triplet = (tail, relation, head)
            yield _triplet
    return None


def load_kg_neighbor_dict(data_path: str, entity_type_dict: Dict[int, int]) -> Dict[int, List[Tuple[int, int, int]]]:
    result = dict()
    for _triplet in _yield_read_raw_triplet(data_path, entity_type_dict):
        # record neighbor info
        for key in (_triplet[0], _triplet[2]):
            result[key] = result.get(key, set())
            result[key].add(_triplet)
    return result


def load_entity_type_dict(data_path: str) -> Dict[int, int]:
    result = dict()
    with open(data_path, "rt") as f:
        for entity_id, _type in enumerate(f.readlines()):
            result[entity_id] = int(_type)
    return result
