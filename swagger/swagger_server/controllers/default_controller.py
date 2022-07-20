import connexion
import six

from swagger_server.models.inline_response200 import InlineResponse200  # noqa: E501
from swagger_server.models.inline_response2001 import InlineResponse2001  # noqa: E501
from swagger_server import util


def root_post(text):  # noqa: E501
    """Renvoie un json contenant le texte pseudonymisé.

     # noqa: E501

    :param text: 
    :type text: str

    :rtype: InlineResponse200
    """
    return 'do some magic!'


def tags_post(text):  # noqa: E501
    """Renvoie un json contenant le texte pseudonymisé et une version du texte où des balises XML ont été ajoutées autour des entités reconnues

     # noqa: E501

    :param text: 
    :type text: str

    :rtype: InlineResponse2001
    """
    return 'do some magic!'
