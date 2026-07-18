from src.rpr import ReferenceProfileRegistry
import random 

def sample_reference_profile_subset(
    rpr: ReferenceProfileRegistry,
    docid,
    doc_type: str = None,
    length: int = None,
    min_length: int = None,
    max_length: int = None,
    seed: int = None,
) -> ReferenceProfileRegistry:
    """
    Build a random subset of `rpr` as a new ReferenceProfileRegistry.

    Parameters
    ----------
    rpr : ReferenceProfileRegistry
        The full list of profiles to sample from.
    docid :
        The docid we care about when deciding inclusion.
    doc_type : str, optional
        If given, only profiles with this `doc_type` are considered for the subset.
    include_docid : {"yes", "no", "random"}
        - "yes":    the profile with `docid` is forced into the subset.
        - "no":     the profile with `docid` is forced OUT of the subset.
        - "random": the profile with `docid` is included with probability `p`.
    length : int, optional
        Exact size of the returned subset. If given, takes priority over
        min_length/max_length.
    min_length, max_length : int, optional
        If `length` is not given, the subset size is drawn uniformly from
        [min_length, max_length] (inclusive), using `seed`.
    seed : int, optional
        Seed for all the random choices made in this function (size choice,
        whether to include the target docid in "random" mode, and which
        other profiles fill the rest of the subset). Uses a local
        random.Random instance, so global random state is untouched.
    p : float, default 0.7
        Probability of including the target docid's profile when
        include_docid="random". Ignored otherwise.

    Returns
    -------
    ReferenceProfileRegistry
        A new list containing the sampled subset of profiles.
    """

    rng = random.Random(seed)

    all_profiles = list(rpr)
    if doc_type is not None:
        all_profiles = [prof for prof in all_profiles if prof.doc_type == doc_type]

    target_profile = rpr.get_profile_by_docid(docid)


    # Pool of profiles eligible to fill the "free" slots of the subset
    # (everything except the target profile, which is handled separately).
    other_profiles = [prof for prof in all_profiles if prof is not target_profile]


    if length is not None:
        subset_size = length
    else:
        if min_length is None or max_length is None:
            raise ValueError(
                "Either `length`, or both `min_length` and `max_length`, must be provided"
            )
        if min_length > max_length:
            raise ValueError("min_length cannot be greater than max_length")
        subset_size = rng.randint(min_length, max_length)

    if subset_size < 0:
        raise ValueError("Computed subset size is negative")

    # How many additional (non-target) profiles do we need to fill the subset?
    remaining_slots = subset_size - 1 if target_profile else subset_size
    remaining_slots = max(remaining_slots, 0)

    chosen_others = rng.sample(other_profiles, min(remaining_slots, len(other_profiles))) if remaining_slots > 0 else []

    subset_profiles = list(chosen_others)
    if target_profile is not None:
        subset_profiles.append(target_profile)

    # Shuffle so the target profile (if forced in) isn't always last.
    rng.shuffle(subset_profiles)

    result = ReferenceProfileRegistry()
    for prof in subset_profiles:
        result.add_profile(prof)

    return result


def format_profile_for_prompt(profile_dict: dict, max_fragments: int = 10) -> dict:
    """
    Take one profile's to_dict() output (with tracked fields still in
    {value: first_seen_id} form) and turn it into a prompt-friendly dict:
    - tracked fields become plain lists of their keys (ids dropped)
    - empty lists are omitted entirely
    - fragments_mentioned is truncated to the last `max_fragments` items
    """
    tracked_fields = ("alternative_titles", "citations", "fragments_mentioned", "authors")

    formatted = {}
    for key, value in profile_dict.items():
        if key in tracked_fields:
            values_list = list(value.keys()) if isinstance(value, dict) else list(value)
            if key == "fragments_mentioned":
                values_list = values_list[-max_fragments:]
            if not values_list:
                continue  # drop empty lists
            formatted[key] = values_list
        else:
            formatted[key] = value

    return formatted



def example_to_string(example_input: dict, docid: str, doctype: str, max_fragments: int = 10) -> str:
    """
    Given one fewshot example's `input` dict (with keys "input_mention",
    "context", "profileRegistry"), build a single formatted string
    describing the input. Output mention is intentionally not included.
    """
    rpr = ReferenceProfileRegistry.from_dict(example_input["profileRegistry"])

    attributes = ["docid", "alternative_titles",
                  "citations", "fragments_mentioned", "authors"]

    
    filtered_rpr = sample_reference_profile_subset(
        rpr=rpr,
        docid=docid,
        doc_type=doctype,
        min_length = 1,
        max_length = 10,
        seed=None,
    )
            
    profiles_formatted = [
        format_profile_for_prompt(profile.to_dict(attributes=attributes), max_fragments=max_fragments)
        for profile in filtered_rpr
    ]

    lines = []
    lines.append(f"Input mention: {example_input['input_mention']}")
    lines.append(f"Context: {example_input['context']}")
    lines.append("Reference Profile Registry:")
    for i, profile in enumerate(profiles_formatted):
        lines.append(f"  Profile {i}: {profile}")

    return "\n".join(lines)