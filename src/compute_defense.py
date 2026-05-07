from collections import defaultdict

def calculate_dmrs_weights(seeker_turn, dmrs_items_list, nli_model):
    """
    seeker_turn: "I don't care that he left, he was useless anyway."
    dmrs_items_list: ["Devaluation of others", "Denial of feelings", ...]
    nli_model: The natural language inference model to use
    """
    # The model "weights" the seeker's sentence against each item
    result = nli_model(seeker_turn, dmrs_items_list, multi_label=True)
    
    # This returns a dictionary of items and their "weights" (0.0 to 1.0)
    item_weights = dict(zip(result['labels'], result['scores']))
    return item_weights


def calculate_dmrs_mechanism(item_weights):
    grouped = defaultdict(list)

    for key, value in item_weights.items():
        prefix = key.rsplit("_", 1)[0]
        grouped[prefix].append(value)

    return {
        defense: (sum(values) - 5) * 100 / 234
        for defense, values in grouped.items()
    }

def compute_defense_score(mechanism_scores):
    level_scores = {
        7: sum(mechanism_scores.get(m, 0) for m in ['Affiliation', 'Altruism', 'Anticipation', 'Humor', 'Self Assertion', 'Self Observation', 'Sublimation', 'Suppression']),
        6: sum(mechanism_scores.get(m, 0) for m in ['Undoing', 'Intellectualization', 'Isolation of Affect']),
        5: sum(mechanism_scores.get(m, 0) for m in ['Displacement', 'Reaction Formation', 'Dissociation', 'Repression']),
        4: sum(mechanism_scores.get(m, 0) for m in ['Omnipotence', 'Idealization Self', 'Idealization Other', 'Devaluation Self', 'Devaluation Other']),
        3: sum(mechanism_scores.get(m, 0) for m in ['Autistic Fantasy', 'Projection', 'Rationalization', 'Denial']),
        2: sum(mechanism_scores.get(m, 0) for m in ['Splitting Self', 'Splitting Other', 'Projective Identification']),
        1: sum(mechanism_scores.get(m, 0) for m in ['Passive Aggression', 'Help Rejecting Complaining', 'Acting Out']),
    }
    # get argmax
    max_level = max(level_scores, key=level_scores.get)
    return level_scores, max_level