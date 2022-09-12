import numpy as np
import itertools
from string import ascii_uppercase
from typing import List, Tuple, Dict
import random
from flair.data import Sentence
from flair.models import SequenceTagger
import stopwatch
import Levenshtein
import itertools

sw = stopwatch.StopWatch()


def pseudonymize(text: str, tagger: SequenceTagger) -> Tuple[str, str]:
    """
    Perform the pseudonymization action and return both the tagged version (see function "tag_entities") and the pseudonymized version

    Args:
        text (str): the input text to pseudonymize
        tagger (SequenceTagger): the flair model for NER

    Returns:
        Tuple[str, str]: the original text with tags, and the pseudonymized text
    """
    with sw.timer("root"):
        text_sentences = [Sentence(t.strip()) for t in text.split("\n") if t.strip()]
        with sw.timer("model_annotation"):
            # inplace function
            tagger.predict(
                sentences=text_sentences,
                mini_batch_size=32,
                embedding_storage_mode="none",
                verbose=True,
            )
        return tag_entities(sentences=text_sentences)


def get_replacement_stock() -> List[str]:
    """
    A list of faked names to replace the information you want to hide
    """
    stock = [f"{letter}..." for letter in ascii_uppercase] + [
        f"{a}{b}..." for a, b in list(itertools.combinations(ascii_uppercase, 2))
    ]
    random.shuffle(stock)
    return stock


def old_tag_entities_sentence(
    sentence: Sentence,
    pseudo_from: int = 0,
    replacements: List[str] = get_replacement_stock(),
) -> str:
    """
    OLD Function to be remove in future
    Args:
        sentence (Sentence): flair.data.Sentence after the running of NER task
        pseudo_from (int, optional): count of already pseudonymized entities. Used to know how to slice the pseudo name stock. Defaults to 0.
    Returns:
        str, str: a text where the entities have a XML tag, and a text where entities have been (poorly) pseudonymized
    """
    # let us assume there is at most one prediction per span
    spans = sentence.get_spans("ner")
    ## WARNING: don't use sentence.text, because there is a shift in characters positions
    #  due to the adding by .text of blanks characters around tokens!
    original_text = sentence.to_plain_string()
    tagged_sentence = original_text
    pseudo_sentence = (
        original_text  # these copies are independent because strings are immutable
    )
    found_entities = 0
    shift_tags_start, shift_tags_end = 0, 0  # shift due to the add of tags
    shift_pseudo_start, shift_pseudo_end = 0, 0
    for span in spans:
        if span.tag in ["PER", "ORG", "LOC"]:
            start, end = span.start_position, span.end_position
            repl = replacements[(pseudo_from + found_entities) % len(replacements)]
            pseudo_sentence = (
                pseudo_sentence[: start + shift_pseudo_start]
                + repl
                + pseudo_sentence[end + shift_pseudo_end :]
            )
            shift_pseudo_start += len(repl) - (end - start)
            shift_pseudo_end += len(repl) - (end - start)
            found_entities += 1
            tagged_sentence = (
                tagged_sentence[: start + shift_tags_start]
                + "</a>"
                + f"<{str(span.tag)}>"
                + original_text[start:end]
                + f"</{str(span.tag)}>"
                + "<a>"
                + tagged_sentence[end + shift_tags_end :]
            )
            shift_tags_start += (
                5 + 6 + 3 + 4
            )  # 5 characters for tag <PER> (or LOC or ORG) + 6 for </PER> + 3 for <a> and 4 for </a>
            shift_tags_end += (
                5 + 6 + 3 + 4
            )  # 5 characters for tag <PER> (or LOC or ORG) + 6 for </PER> + 3 for <a> and 4 for </a>
    tagged_sentence = "<a>" + tagged_sentence + "</a>"
    tagged_sentence = tagged_sentence.replace("<a></a>", "")
    return (f"<sentence>{tagged_sentence}</sentence>", pseudo_sentence, found_entities)


def tag_entities(sentences: List[Sentence]) -> Tuple[str, str]:
    """
    Tag and replace each PERSON pame, ORGANIZATION name or LOCATION name detected with NER (model "ner" of flair. There are also some models with more classes)
    The tags will be <LOC> (for spans about location), <PER> (for persons) and <ORG> (for organization), <a> (=NO TAG for this span).
    sentence are bounded with tags <sentence> and there is a tag <text> around the whole text.

    TODO : enable entity linking before pseudonymization to perform a better pseudo task

    Args:
        sentences (List[Sentence]): the flair.data.Sentence objects after a NER task have been performed with flair model

    Returns:
        str, str: a text where the entities have XML tags, and a text where entities have been (poorly) pseudonymized
    """

    all_starts, all_ends, all_tags, all_entities = list(), list(), list(), list()
    all_texts = list()
    count_entities = 0
    pseudo_replacements = dict()
    replacement_stock = (
        get_replacement_stock()
    )  # where replacement values are sampled from

    # save the position, tag values and entity texts for every sentence
    for sentence in sentences:
        starts, ends, tags, entities, text = apply_ner_sentence(sentence)
        all_starts.append(starts)  # type: List[List[int]]
        all_ends.append(ends)
        all_tags.append(tags)
        all_entities.append(entities)
        all_texts.append(text)

        # generate a replacement token for each detected entity
        for i, entity in enumerate(entities):
            # Remark: the indice (i+count_entities) is unique for each entity
            pseudo_replacements[entity] = replacement_stock[i + count_entities]
        count_entities += len(entities)

    # Prepare replacement dictionary
    normalized_entities = normalize_entities(
        entities=list(np.concatenate(all_entities)),
        tags=list(np.concatenate(all_tags)),
        distance_threshold=2,
    )

    for entities in all_entities:
        for entity in entities:
            if entity in normalized_entities:
                pseudo_replacements[entity] = pseudo_replacements[
                    normalized_entities[entity]
                ]

    # Finally tag and pseudonymize sentences
    tagged_text, pseudo_text = "", ""

    for ind, text in enumerate(all_texts):
        tagged_sentence, pseudo_sentence = apply_tagging_sentence(
            starts=all_starts[ind],
            ends=all_ends[ind],
            tags=all_tags[ind],
            entities=all_entities[ind],
            plain_text=text,
            replacement_dict=pseudo_replacements,
        )
        pseudo_text += pseudo_sentence
        tagged_text += tagged_sentence

    return (
        "<text>" + tagged_text.replace("<sentence></sentence>", "") + "</text>",
        pseudo_text,
    )


def apply_ner_sentence(
    sentence: Sentence,
) -> Tuple[List[int], List[int], List[str], List[str], str]:
    """
    For one sentence, return the inner text, the starting positions, the ending positions and the tags of every recognized entity

    Args:
        sentence (Sentence): the flair.data.Sentence after a NER task have been performed with flair model

    Returns:
        List[int], List[int], List[str], List[str], str: starts, ends, tags, entity texts of the entities found in the sentence + the text of the sentence
    """

    spans = sentence.get_spans("ner")
    starts, ends, tags, entities = list(), list(), list(), list()
    text = sentence.to_plain_string()
    for span in spans:
        if span.tag in ["PER", "ORG", "LOC"]:
            # beware span.text does not respect the initial punctuations: it's why we need to use plain string form
            start, end, tag = span.start_position, span.end_position, str(span.tag)
            starts.append(start)
            ends.append(end)
            tags.append(tag)
            entities.append(text[start:end])
    return starts, ends, tags, entities, text


# perform the replacement and the add of tags
def apply_tagging_sentence(
    starts: List[int],
    ends: List[int],
    tags: List[str],
    entities: List[str],
    plain_text: str,
    replacement_dict: Dict[str, str],
) -> Tuple[str, str]:
    """
    Args:
        starts, ends, tags, entity texts of the entities found in the sentence + the text of the sentence + the prepared replacement dictionary for pseudo
    Returns:
        str, str: a text where the entities have a XML tag, and a text where entities have been pseudonymized
    """

    assert (
        len(starts) == len(ends) == len(tags) == len(entities)
    ), "Input lists mast be of the same length"
    shift_tags_start, shift_tags_end = 0, 0  # shift due to the add of tags
    shift_pseudo_start, shift_pseudo_end = 0, 0
    tagged_sentence, pseudo_sentence = plain_text, plain_text
    n_entities = len(start)

    for i in range(n_entities):
        start, end, entity, tag = starts[i], ends[i], entities[i], tags[i]
        replacement = replacement_dict[entity]

        pseudo_sentence = (
            pseudo_sentence[: start + shift_pseudo_start]
            + replacement
            + pseudo_sentence[end + shift_pseudo_end :]
        )
        shift_pseudo_start += len(replacement) - (end - start)
        shift_pseudo_end += len(replacement) - (end - start)
        tagged_sentence = (
            tagged_sentence[: start + shift_tags_start]
            + "</a>"
            + f"<{tag}>"
            + plain_text[start:end]
            + f"</{tag}>"
            + "<a>"
            + tagged_sentence[end + shift_tags_end :]
        )
    tagged_sentence = "<a>" + tagged_sentence + "</a>"
    tagged_sentence = tagged_sentence.replace("<a></a>", "")
    return (
        f"<sentence>{tagged_sentence}</sentence>",
        pseudo_sentence,
    )


def normalize_entities(
    entities: List[str], tags: List[str], distance_threshold: int = 2
) -> Dict[str, str]:
    """
    Analyze a list of entities, determine if they are similar AND share the same tag, and return a dictionary where
    a key is an entity from the input list, and a value is the indice of the first found similar entity
    (if an entity as no alter ego, it does not appear in output dictionary)

    Warning: this function take into account the value of the tag

    Args:
        entities (List[str]): list of entities extracted with NER
        distance_threshold (int, optional): Under this Levenshtein distance, two entities are considered as similar. Defaults to 2.

    Returns:
        Dict[str, str]: correspondance between input entities and normalized entities
    """

    def is_similar(a: str, b: str, distance_threshold: int) -> bool:
        similarity = Levenshtein.distance(a, b)
        return 0 < similarity < distance_threshold

    correspondances = {}

    for i, entity_i in enumerate(entities):
        for j, entity_j in enumerate(entities):
            if j > i:
                if is_similar(entity_i, entity_j, distance_threshold) and (
                    tags[i] == tags[j]
                ):
                    correspondances[entity_j] = entity_i
                    break  # go back to external loop
    return correspondances
