import json

# Define category IDs from the ontology
MUSIC_GENRE_ID = "/m/0kpv1t"  # "Musical genre"
MUSIC_INSTRUMENT_ID = "/m/04szw"  # "Musical instrument"
MOOD_ID = "/t/dd00030"  # "Mood"


def load_ontology(file_path):
    """
    Loads the AudioSet ontology from a local JSON file.

    :param file_path: Path to the ontology JSON file.
    :return: Parsed JSON data (list of dicts).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_id_to_node_map(ontology):
    """
    Build a dictionary mapping each node's 'id' to the node object itself.

    :param ontology: List of ontology nodes.
    :return: Dictionary mapping node 'id' -> node dict.
    """
    return {node['id']: node for node in ontology}


def get_first_level_children(ontology, parent_id):
    """
    Gets the first-level (direct) children of a given parent category
    that are NOT marked as 'abstract'.

    :param ontology: The full ontology as a list of dicts.
    :param parent_id: The ID of the parent category (e.g., "Musical genre").
    :return: A list of child node names that are non-abstract.
    """
    id_to_node = build_id_to_node_map(ontology)

    if parent_id not in id_to_node:
        return []

    parent_node = id_to_node[parent_id]

    first_level_children = []
    for child_id in parent_node.get("child_ids", []):
        child_node = id_to_node.get(child_id)
        if child_node and "abstract" not in child_node.get("restrictions", []):
            first_level_children.append(child_node["name"])

    return first_level_children


if __name__ == "__main__":
    ontology_file = "ontology.json"  # Replace with your local file path

    # Load ontology from file
    ontology = load_ontology(ontology_file)

    # Extract first-level children for each category
    musical_genres = get_first_level_children(ontology, MUSIC_GENRE_ID)
    musical_instruments = get_first_level_children(ontology, MUSIC_INSTRUMENT_ID)
    moods = get_first_level_children(ontology, MOOD_ID)

    # Print the extracted categories
    print("\nMusical Genres:")
    print(musical_genres)

    print("\nMusical Instruments:")
    print(musical_instruments)

    print("\nMoods:")
    print(moods)
