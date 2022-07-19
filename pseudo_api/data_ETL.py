import copy
import itertools
from collections import defaultdict, Counter
from string import ascii_uppercase
from typing import List, Tuple, Dict
import random
from flair.data import Sentence
from flair.models import SequenceTagger
import stopwatch

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


def update_stats(
    analysis_stats: dict,
    analysis_ner_stats: dict,
    time_info: stopwatch.AggregatedReport,
    output_type: str,
):
    def update_averages(avg: float, size: int, value: float):
        return (size * avg + value) / (size + 1)

    def update_dict_values(old_dict: dict, new_dict: dict):
        for k, v in new_dict.items():
            if k in old_dict:
                old_dict[k] += v
            else:
                old_dict[k] = v
        return old_dict

    # get previous values
    old_nb_analyzed_documents = analysis_stats.get("nb_analyzed_documents", 0)
    old_nb_analyzed_sentences = analysis_stats.get("nb_analyzed_sentences", 0)
    old_output_types_freq = analysis_stats.get(f"output_type_{output_type}", 0)
    old_avg_time = analysis_stats.get("avg_time_per_doc", 0)
    old_avg_time_per_sent = analysis_stats.get("avg_time_per_sentence", 0)

    analysis_stats["nb_analyzed_documents"] = old_nb_analyzed_documents + 1
    analysis_stats[
        "nb_analyzed_sentences"
    ] = old_nb_analyzed_sentences + analysis_ner_stats.pop("nb_analyzed_sentences")

    # add entities tags freqs
    analysis_stats = update_dict_values(analysis_stats, analysis_ner_stats)

    # deal with time stats
    delta_ms, _, _ = time_info.aggregated_values["root"]
    analysis_stats["avg_time_per_doc"] = update_averages(
        old_avg_time, old_nb_analyzed_documents, delta_ms
    )
    analysis_stats["avg_time_per_sentence"] = update_averages(
        old_avg_time_per_sent,
        old_nb_analyzed_sentences,
        delta_ms / analysis_stats["nb_analyzed_sentences"],
    )

    analysis_stats[f"output_type_{output_type}"] = old_output_types_freq + 1


def get_replacement_stock() -> List[str]:
    """
    A list of faked names to replace the information you want to hide
    """
    stock = [f"{letter}..." for letter in ascii_uppercase] + [
        f"{a}{b}..." for a, b in list(itertools.combinations(ascii_uppercase, 2))
    ]
    random.shuffle(stock)
    return stock


def tag_entities(sentences: List[Sentence]) -> Tuple[str, str]:
    """
    Tag and replace each PERSON pame, ORGANIZATION name or LOCATION name detected with NER
    The tags will be <LOC> (for spans about location), <PER> (for persons) and <ORG> (for organization), <a> (=NO TAG for this span).
    sentence are bounded with tags <sentence> and there is a tag <text> around the whole text.

    TODO : enable entity linking before pseudonymization to perform a better pseudo task

    Args:
        sentences (List[Sentence]): the flair.data.Sentence objects after a NER task have been performed with flair model

    Returns:
        str, str: a text where the entities have XML tags, and a text where entities have been (poorly) pseudonymized
    """
    replacements = get_replacement_stock()

    def tag_entities_one_sentence(sentence: Sentence, pseudo_from: int = 0) -> str:
        """
        Args:
            sentence (Sentence): flair.data.Sentence after the running of NER task
            pseudo_from (int, optional): count of already pseudonymized entities. Used to know how to slice the pseudo name stock. Defaults to 0.
        Returns:
            str, str: a text where the entities have a XML tag, and a text where entities have been (poorly) pseudonymized
        """
        # let us assume there is at most one prediction per span
        spans = sentence.get_spans("ner")
        tagged_sentence = sentence.text
        pseudo_sentence = (
            sentence.text
        )  # these copies are independent because strings are immutable
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
                    + sentence.text[start : end]
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
        return (
            f"<sentence>{tagged_sentence}</sentence>",
            pseudo_sentence,
            found_entities,
        )

    tagged_text, pseudo_text = "", ""

    # "total_found_entities" is used to consume the replacement stock from the index we stopped
    total_found_entities = 0
    for _, sentence in enumerate(sentences):
        tagged_sentence, pseudo_sentence, found_entities = tag_entities_one_sentence(
            sentence, pseudo_from=total_found_entities
        )
        total_found_entities += found_entities
        pseudo_text += pseudo_sentence
        tagged_text += tagged_sentence
    return (
        "<text>" + tagged_text.replace("<sentence></sentence>", "") + "</text>",
        pseudo_text,
    )
