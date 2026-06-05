from .sublabel_definitions import SUBLABEL_DEFINITIONS_V1, SUBLABEL_DEFINITIONS_V2
from .guidelines import ANNOTATION_GUIDELINES   
from src.tokenizer_utils import decode




def build_sublabel_definitions(keep_labels, sublabel_definitions=SUBLABEL_DEFINITIONS_V1):
    missing = [lbl for lbl in keep_labels if lbl not in sublabel_definitions]
    if missing:
        raise ValueError(f"Unknown sublabels: {missing}")

    return "\n\n\n".join(
        f"- {sublabel_definitions[label]}"
        for label in keep_labels
    )

def get_prompt_sublabel_extraction(prompt_path, keep_labels, few_shot_examples=None, sublabel_definitions=SUBLABEL_DEFINITIONS_V1, with_context_bool=True):
    """
    Generate dynamic prompt for sublabel extraction using a TXT template.

    Args:
        keep_labels (List[str]): Sublabels to extract (e.g. ["title", "citation"])

    Returns:
        Tuple[str, str]: (system_prompt, user_prompt_template)
    """

    with open(prompt_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # --- Build dynamic fields ---
    sublabels_str = ", ".join(keep_labels)
    sublabels_definition = build_sublabel_definitions(keep_labels, sublabel_definitions=sublabel_definitions)

    # --- Fill template ---
    system_prompt = template.format(
        sublabels=sublabels_str,
        sublabels_definition=sublabels_definition,
    )

    # Add few-shot examples if provided
    if few_shot_examples:
        system_prompt += "\n\nHere are some examples:\n"
        for i, (input_text, expected_output) in enumerate(few_shot_examples, 1):
            system_prompt += f"\nExample {i}:\n"
            system_prompt += f"<ORIGINAL_TEXT>{input_text}<END_ORIGINAL_TEXT>\n"
            system_prompt += f"<EXPECTED_OUTPUT>{expected_output}<END_EXPECTED_OUTPUT>\n"
    
    if with_context_bool:
        user_prompt_template = """Here is the legal context to consider for annotation:\n{context}\n\nNow, please annotate the following legal text with the appropriate sublabels tags:"""
    else:
        user_prompt_template = """Please annotate the following legal text with the appropriate sublabels tags:"""

    user_prompt_template += """

    <ORIGINAL_TEXT>{text}<END_ORIGINAL_TEXT>
    
    OUTPUT:"""

    return system_prompt, user_prompt_template

