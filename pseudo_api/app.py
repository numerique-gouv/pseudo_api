import logging

import stopwatch
from flair.models import SequenceTagger
from flask import Flask
from flask import request, jsonify

from data_ETL import pseudonymize, sw

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

server = Flask(__name__)

# Env variables
PSEUDO_MODEL_PATH = "flair/ner-french"
TAGGER = SequenceTagger.load(PSEUDO_MODEL_PATH)


def run_pseudonymize_request(return_tags: bool = False):
    data = {"success": False}
    try:
        if request.form.get("text"):
            text = request.form.get("text")
            logging.info("Tagging text with model...")
            tagged_text, pseudo_text = pseudonymize(text=text, tagger=TAGGER)
            data["pseudo"] = pseudo_text
            if return_tags:
                data["tags"] = tagged_text
            data["success"] = True
    except Exception as e:
        logger.error(e)
    # finally:
    #    logger.info(stopwatch.format_report(sw.get_last_aggregated_report()))
    return jsonify(data)


@server.route("/", methods=["GET", "POST"])
def api_pseudonymize():
    if request.method == "GET":
        return "The model is up and running. Send a POST request"
    else:
        return run_pseudonymize_request(return_tags=False)


@server.route("/tags/", methods=["GET", "POST"])
def api_tags():
    if request.method == "GET":
        return "The model is up and running. Send a POST request"
    else:
        return run_pseudonymize_request(return_tags=True)
