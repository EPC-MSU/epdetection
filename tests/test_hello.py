from unittest import TestCase
from hello_world import say_hello


class TestHello(TestCase):
    def test_hello_say(self):
        self.assertTrue(say_hello() == "hello")
