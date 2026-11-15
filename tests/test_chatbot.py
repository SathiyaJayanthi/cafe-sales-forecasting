"""
Tests for CafeAssistantBot in src.chatbot.
"""

import unittest
from src.chatbot import CafeAssistantBot, CoffeeSenseBot


class TestCafeAssistantBot(unittest.TestCase):
    def setUp(self):
        self.bot = CafeAssistantBot()

    def test_greeting_only_on_pure_greetings(self):
        # Pure greeting
        resp = self.bot.get_response("hello")
        self.assertIn("Welcome to Cafe Assistant", resp)

        resp2 = self.bot.get_response("hi")
        self.assertIn("Welcome to Cafe Assistant", resp2)

    def test_most_selling_item_query(self):
        # Query that previously broke due to 'hi' in 'which'
        resp = self.bot.get_response("which is most selling item in cafe")
        self.assertNotIn("Welcome to Cafe Assistant", resp)
        self.assertIn("Latte", resp)
        self.assertIn("1,085 orders", resp)
        self.assertIn("₹40,754.00", resp)

    def test_least_selling_item_query(self):
        resp = self.bot.get_response("what is the least selling coffee?")
        self.assertIn("Double Vanilla", resp)
        self.assertIn("187 orders", resp)

    def test_total_revenue_query(self):
        resp = self.bot.get_response("what was the total revenue in 2025?")
        self.assertIn("217,260.00", resp)

    def test_payment_methods_query(self):
        resp = self.bot.get_response("how do customers pay?")
        self.assertIn("Card", resp)
        self.assertIn("69.0%", resp)

    def test_info_not_available_when_out_of_scope(self):
        resp = self.bot.get_response("who is the store owner?")
        self.assertIn("this information is not available with the provided dataset", resp)

        resp2 = self.bot.get_response("what is the wifi password?")
        self.assertIn("this information is not available with the provided dataset", resp2)

        resp3 = self.bot.get_response("do you sell pizza?")
        self.assertIn("this information is not available with the provided dataset", resp3)

    def test_alias_backward_compatibility(self):
        bot2 = CoffeeSenseBot()
        resp = bot2.get_response("which is most selling item in cafe")
        self.assertIn("Latte", resp)


if __name__ == "__main__":
    unittest.main()
