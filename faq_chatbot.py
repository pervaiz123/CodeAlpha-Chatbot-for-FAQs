"""
FAQ Chatbot — NLP-powered question matching
==========================================
Requirements:  pip install nltk scikit-learn colorama
First run:     python faq_chatbot.py --download   (fetches NLTK data once)

Architecture
------------
1.  FAQs are stored as (question, answer) pairs.
2.  At startup every FAQ question is preprocessed:
        tokenise → lowercase → strip punctuation → remove stopwords → lemmatise
3.  A TF-IDF matrix is built over the preprocessed questions.
4.  At query time the user's question goes through the same pipeline and is
    compared against every FAQ vector using cosine similarity.
5.  The best match above a configurable threshold is returned.
"""

import sys
import re
import string
import argparse

# ── third-party ──────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    print("Missing dependencies.  Run:  pip install nltk scikit-learn")
    sys.exit(1)

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

    class _Dummy:
        def __getattr__(self, _): return ""
    Fore = Style = _Dummy()


# ── FAQ dataset ───────────────────────────────────────────────────────────────
# Topic: E-commerce / online shopping support
FAQS = [
    ("How do I track my order?",
     "You can track your order by visiting the 'My Orders' section in your account "
     "and clicking 'Track Shipment'.  A live map and estimated delivery date will be shown."),

    ("What is your return policy?",
     "We accept returns within 30 days of delivery for unused items in original packaging.  "
     "Start a return from 'My Orders' → 'Return Item' and we'll email you a prepaid label."),

    ("How long does shipping take?",
     "Standard shipping takes 5–7 business days.  Express shipping (1–2 days) is available "
     "at checkout for an additional fee.  International orders may take 10–15 business days."),

    ("Can I change or cancel my order?",
     "Orders can be modified or cancelled within 1 hour of placement.  "
     "After that the order enters fulfilment.  Contact support immediately for urgent changes."),

    ("What payment methods do you accept?",
     "We accept Visa, Mastercard, American Express, PayPal, Apple Pay, Google Pay, "
     "and bank transfer.  All transactions are secured with 256-bit SSL encryption."),

    ("How do I reset my password?",
     "Click 'Forgot Password' on the login page, enter your email address, "
     "and check your inbox for a reset link.  The link expires after 15 minutes."),

    ("Do you offer international shipping?",
     "Yes!  We ship to 50+ countries.  Duties and taxes are calculated at checkout "
     "and vary by destination.  Some remote regions may not be covered — "
     "enter your address to check availability."),

    ("How do I apply a discount code?",
     "Add items to your cart, proceed to checkout, and enter your code in the "
     "'Promo Code' field.  The discount is applied instantly.  Only one code per order."),

    ("Is my personal information secure?",
     "Absolutely.  We never sell your data.  All information is encrypted in transit "
     "and at rest.  You can request a copy or deletion of your data from Account Settings."),

    ("How do I contact customer support?",
     "Reach us 24/7 via live chat (bottom-right button on any page), "
     "email support@example.com, or call +1-800-555-0100 (Mon–Fri 9 AM–6 PM EST)."),

    ("What if I received a damaged item?",
     "We're sorry to hear that!  Take a photo and contact us within 7 days of delivery.  "
     "We'll arrange a free replacement or full refund — whichever you prefer."),

    ("Can I exchange a product for a different size or colour?",
     "Yes.  Start an exchange from 'My Orders', choose the new variant, "
     "and ship the original back using our free label.  "
     "The replacement ships once we receive your return."),

    ("When will I receive my refund?",
     "Refunds are processed within 3–5 business days after we receive your return.  "
     "Credit card refunds may take an additional 5–7 days to appear on your statement."),

    ("Do you have a loyalty or rewards programme?",
     "Yes!  Join ShopRewards for free.  Earn 1 point per $1 spent; "
     "100 points = $1 reward credit.  VIP tiers unlock free express shipping and exclusive deals."),

    ("How do I unsubscribe from marketing emails?",
     "Click 'Unsubscribe' at the bottom of any marketing email, "
     "or go to Account Settings → Email Preferences and turn off promotional emails."),
]


# ── NLP preprocessing ─────────────────────────────────────────────────────────
def download_nltk_data():
    """Download required NLTK corpora (run once)."""
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    print("NLTK data downloaded successfully.")


STOP_WORDS = None   # populated lazily after NLTK data is confirmed present
LEMMATIZER = None


def _ensure_nltk():
    """Download NLTK data automatically on first run, then load into globals."""
    global STOP_WORDS, LEMMATIZER
    if STOP_WORDS is None:
        for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
            nltk.download(pkg, quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
        LEMMATIZER = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """
    Pipeline:
        1. Lowercase
        2. Remove punctuation
        3. Tokenise
        4. Remove stopwords
        5. Lemmatise
    Returns a single whitespace-joined string ready for TF-IDF.
    """
    _ensure_nltk()
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ── Chatbot core ──────────────────────────────────────────────────────────────
class FAQChatbot:
    """TF-IDF + cosine-similarity FAQ matcher."""

    THRESHOLD = 0.15        # minimum similarity to return a match
    NO_MATCH_MSG = (
        "I'm sorry, I don't have an answer for that right now.  "
        "Please contact our support team at support@example.com."
    )

    def __init__(self, faqs: list[tuple[str, str]]):
        self.faqs = faqs
        self._questions_raw  = [q for q, _ in faqs]
        self._answers        = [a for _, a in faqs]

        # Build corpus from preprocessed questions
        self._corpus = [preprocess(q) for q in self._questions_raw]

        self._vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self._vectorizer.fit_transform(self._corpus)

    def get_answer(self, user_question: str) -> tuple[str, float, str]:
        """
        Returns (answer, similarity_score, matched_faq_question).
        If no match exceeds the threshold, returns (NO_MATCH_MSG, 0.0, '').
        """
        processed = preprocess(user_question)
        if not processed.strip():
            return self.NO_MATCH_MSG, 0.0, ""

        user_vec = self._vectorizer.transform([processed])
        scores   = cosine_similarity(user_vec, self._tfidf_matrix).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < self.THRESHOLD:
            return self.NO_MATCH_MSG, best_score, ""

        return (
            self._answers[best_idx],
            best_score,
            self._questions_raw[best_idx],
        )


# ── Terminal UI ───────────────────────────────────────────────────────────────
BANNER = r"""
  ___  ___  ___     ___ _         _   _          _
 | __|| _ \| __|   / __| |_  __ _| |_| |__  ___ | |_
 | _| |   /| _|   | (__| ' \/ _` |  _| '_ \/ _ \|  _|
 |_|  |_|_\|___|   \___|_||_\__,_|\__|_.__/\___/ \__|
"""

def print_banner():
    print(Fore.CYAN + BANNER + Style.RESET_ALL)
    print(f"{Fore.WHITE}  NLP-powered FAQ Assistant  |  type 'quit' to exit\n")


def run_cli(bot: FAQChatbot, verbose: bool = False):
    print_banner()
    while True:
        try:
            user_input = input(f"{Fore.GREEN}You › {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print(f"{Fore.CYAN}Bot › {Style.RESET_ALL}Thanks for chatting!  Bye 👋")
            break

        answer, score, matched_q = bot.get_answer(user_input)

        print(f"\n{Fore.CYAN}Bot › {Style.RESET_ALL}{answer}\n")
        if verbose:
            if matched_q:
                print(f"  {Fore.YELLOW}[matched: '{matched_q}'  |  score: {score:.3f}]{Style.RESET_ALL}\n")
            else:
                print(f"  {Fore.YELLOW}[no match above threshold  |  best score: {score:.3f}]{Style.RESET_ALL}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NLP FAQ Chatbot")
    parser.add_argument("--download", action="store_true",
                        help="Download required NLTK data and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show match score and matched question")
    parser.add_argument("--query", "-q", type=str,
                        help="Answer a single question and exit (non-interactive)")
    args = parser.parse_args()

    if args.download:
        download_nltk_data()
        return

    bot = FAQChatbot(FAQS)

    if args.query:
        answer, score, matched_q = bot.get_answer(args.query)
        print(f"Answer: {answer}")
        if args.verbose:
            print(f"Matched: {matched_q!r}  |  Score: {score:.3f}")
    else:
        run_cli(bot, verbose=args.verbose)


if __name__ == "__main__":
    main()