# based on https://explosion.ai/blog/coref and
# https://gist.github.com/thomashacker/b5dd6042c092e0a22c2b9243a64a2466
import spacy
from spacy.tokens import Doc
nlp = spacy.load("en_coreference_web_trf")
from loggerLog import *

# use the nlp model to process text and get the coreferences, then apply function resolve_references to resolve them
def spacyExperimentalCoref(text):
    doc = nlp(text)
    #print(doc.spans)
    try:
        textCoref = resolve_references(doc)
    except Exception as e:
        logger.error("Exception in function resolve_references in spacyExperimentalCoref (corefResol.py): %s", e)
    return textCoref


# define lightweight function for resolving references in text
def resolve_references(doc: Doc) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_
            
            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string