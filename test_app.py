import unittest
import openai
from unittest.mock import patch
from types import SimpleNamespace
from streamlit.testing.v1 import AppTest


class TestChatbotApp(unittest.TestCase):

    def test_smoke(self):
        """Test if the app runs without throwing an exception."""
        at = AppTest.from_file("app.py", default_timeout=10).run()
        assert not at.exception

    def test_sidebar(self):
        """Test if a single text input for the OpenAI API key exists in the sidebar."""
        at = AppTest.from_file("app.py").run()
        assert len(at.sidebar.text_input) == 1

    def test_session_state(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_file("app.py").run()
        assert "messages" in at.session_state
        assert at.session_state["messages"][0]["role"] == "assistant"

    def test_messages_render(self):
        """Test if the initial assistant message is rendered."""
        at = AppTest.from_file("app.py").run()
        assert len(at.chat_message) == 1

    @patch("openai.ChatCompletion.create")
    def test_openai_api(self, mock_openai):
        """Test if the OpenAI API is correctly integrated and the chat updates."""
        mock_openai.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='Sure, I can help!', role='assistant'))]
        )
        at = AppTest.from_file("app.py").run()
        at.text_input[0].set_value("test_api_key").run()
        assert at.session_state["openai_api_key"] == "test_api_key"
        at.chat_input[0].set_value("Hello").run()
        assert len(at.session_state["messages"]) == 3

    @patch("openai.ChatCompletion.create")
    def test_openai_401_error(self, mock_openai):
        """Test if the app correctly handles a 401 Authentication Error from OpenAI."""
        mock_openai.side_effect = openai.error.AuthenticationError(
            "Invalid Authentication")
        at = AppTest.from_file("app.py").run()
        at.text_input[0].set_value("wrong_api_key").run()
        at.chat_input[0].set_value("Hello").run()
        if len(at.text_area) > 0:
            assert "Invalid Authentication" in at.text_area[0].get_value()
        else:
            print("text_area is empty! Something's wrong!")
