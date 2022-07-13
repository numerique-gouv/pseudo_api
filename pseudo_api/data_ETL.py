import copy
import itertools
from collections import defaultdict, Counter
from string import ascii_uppercase
from typing import Callable, List, Tuple

from flair.data import Token, Sentence
from flair.models import SequenceTagger
import stopwatch

sw = stopwatch.StopWatch()


def create_conll_output(sentences_tagged: List[Sentence]):
    # TODO : fix type
    tags = []
    conll_str: str = ""
    for sent_pred in sentences_tagged:
        tags.extend([s for s in sent_pred.get_spans("ner")])
        for tok_pred in sent_pred:
            result_str = f"{tok_pred.text}\t{tok_pred}\t" \
                         f"{tok_pred.start_pos}\t{tok_pred.end_pos}"
            conll_str += result_str + "\n"
        conll_str += "\n"
    return conll_str #, Counter(tags)


def prepare_output(text: str, tagger: SequenceTagger, output_type: str = "pseudonymized"):
    stats_dict = {}
    with sw.timer("root"):
        text_sentences = [Sentence(t.strip(), use_tokenizer=True)
                          for t in text.split("\n") if t.strip()]
        with sw.timer('model_annotation'):
            tagger.predict(sentences=text_sentences,
                                              mini_batch_size=32,
                                              embedding_storage_mode="none",
                                              verbose=True)

        if output_type == "conll":
            api_output = create_conll_output(sentences_tagged=text_sentences)
        elif output_type == "tagged":
            api_output = create_tagged_text(sentences_tagged=text_sentences)
        elif output_type == "pseudonymized":
            api_output = create_pseudonymized_text(sentences_tagged=text_sentences)

        # deal with stats
        stats_dict["nb_analyzed_sentences"] = len(text)
        return api_output, stats_dict


def update_stats(analysis_stats: dict, analysis_ner_stats: dict, time_info: stopwatch.AggregatedReport,
                 output_type: str):
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
    analysis_stats["nb_analyzed_sentences"] = old_nb_analyzed_sentences + analysis_ner_stats.pop(
        "nb_analyzed_sentences")

    # add entities tags freqs
    analysis_stats = update_dict_values(analysis_stats, analysis_ner_stats)

    # deal with time stats
    delta_ms, _, _ = time_info.aggregated_values["root"]
    analysis_stats["avg_time_per_doc"] = update_averages(old_avg_time,
                                                         old_nb_analyzed_documents, delta_ms)
    analysis_stats["avg_time_per_sentence"] = update_averages(old_avg_time_per_sent,
                                                              old_nb_analyzed_sentences,
                                                              delta_ms / analysis_stats[
                                                                  "nb_analyzed_sentences"])

    analysis_stats[f"output_type_{output_type}"] = old_output_types_freq + 1


def create_tagged_text(sentences_tagged: List[Sentence]):
    # Iterate over the modified sentences to recreate the text (tagged)
    tagged_str = ""
    tags = []
    for sent in sentences_tagged:
        tags.extend([s.tag for s in sent.get_spans("ner")])
        temp_str = sent.to_tagged_string()
        tagged_str += temp_str + "\n\n"

    return tagged_str #, Counter(tags)


def create_pseudonymized_text(sentences_tagged: List[Sentence]):
    singles = [f"{letter}..." for letter in ascii_uppercase]
    doubles = [f"{a}{b}..." for a, b in list(itertools.combinations(ascii_uppercase, 2))]
    pseudos = singles + doubles
    pseudo_entity_dict = {}
    sentences_pseudonymized = copy.deepcopy(sentences_tagged)
    tag_stats = defaultdict(int)

    # Replace the entities within the sentences themselves
    for id_sn, sent in enumerate(sentences_pseudonymized):
        for sent_span in sent.get_spans("ner"):
            if "LOC" in sent_span.tag:
                for id_tok in range(len(sent_span.tokens)):
                    sent_span.tokens[id_tok].text = "..."
            else:
                for id_tok, token in enumerate(sent_span.tokens):
                    replacement = pseudo_entity_dict.get(token.text.lower(), pseudos.pop(0))
                    pseudo_entity_dict[token.text.lower()] = replacement
                    sent_span.tokens[id_tok].text = replacement

    # Iterate over the modified sentences to recreate the text (pseudonymized)
    pseudonymized_str = ""
    for sent in sentences_pseudonymized:
        detokenized_str = " ".join([t.text for t in sent.tokens]),
        pseudonymized_str += detokenized_str + "\n\n"

    return pseudonymized_str, tag_stats


def create_api_output(sentences_tagged: List[Sentence]) -> Tuple[str, str]:
    "We create two output texts: tagged and pseudonyimzed"
    tagged_str = create_tagged_text(sentences_tagged=sentences_tagged)
    pseudonymized_str = create_pseudonymized_text(sentences_tagged=sentences_tagged)

    return tagged_str, pseudonymized_str
