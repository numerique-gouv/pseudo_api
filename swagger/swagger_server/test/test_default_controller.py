# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.inline_response200 import InlineResponse200  # noqa: E501
from swagger_server.models.inline_response2001 import InlineResponse2001  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_root_post(self):
        """Test case for root_post

        Renvoie un json contenant le texte pseudonymisé.
        """
        data = dict(text='text_example')
        response = self.client.open(
            '/pseudo/api//',
            method='POST',
            data=data,
            content_type='application/x-www-form-urlencoded')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_tags_post(self):
        """Test case for tags_post

        Renvoie un json contenant le texte pseudonymisé et une version du texte où des balises XML ont été ajoutées autour des entités reconnues
        """
        data = dict(text='text_example')
        response = self.client.open(
            '/pseudo/api//tags/',
            method='POST',
            data=data,
            content_type='application/x-www-form-urlencoded')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
